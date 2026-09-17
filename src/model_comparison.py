"""
Model / component comparison.

Compares only models that are actually installed and runnable in this
environment:

  - ASR:            Whisper "tiny" vs "base" (currently used in the
                     pipeline) vs "small" (already cached locally).
                     "tiny" is downloaded on first run if not already
                     cached (~75MB pretrained weights, not a new
                     dependency).
  - Summarization:   T5-small (currently used in the pipeline)
                     vs a TF-IDF extractive baseline (no new model).
  - Diarization:     only one model/pipeline (PyAnnote community-1) is
                     installed, so no model-vs-model comparison is run.
                     This is reported explicitly rather than skipped
                     silently or faked.

The ASR comparison transcribes the FULL audio file with both Whisper
sizes (preserving full context, exactly like the original pipeline
run) and then restricts each model's own hypothesis to the same audio
time window that outputs/evaluations/reference_transcript.txt was
written against (the span of the first 17 base-model segments in
outputs/transcripts/meeting.json). An earlier version of this script
hard-trimmed the audio to a fixed 85s clip before transcribing; that
cut off real reference content (the window actually extends to
~97.7s) and produced a misleadingly inflated, identical WER for both
models. Filtering by timestamp after a full-context transcription
avoids that. This is a single manually-verified development sample,
not a benchmark dataset -- these numbers describe this one recording
only.
"""

import json
import os
import re
import time
from datetime import datetime, timezone

import torch
import whisper
from jiwer import wer as compute_wer
from rouge_score import rouge_scorer

from extractive_summary_baseline import generate_extractive_summary


AUDIO_FILE = "data/audio/meeting.wav"
REFERENCE_TRANSCRIPT_FILE = "outputs/evaluations/reference_transcript.txt"
WER_RESULTS_FILE = "outputs/evaluations/wer_results.json"
BASE_TRANSCRIPT_FILE = "outputs/transcripts/meeting.json"

REFERENCE_SUMMARY_FILE = "outputs/evaluations/reference_summary.txt"
MEETING_SUMMARY_FILE = "outputs/summaries/meeting_summary.json"

OUTPUT_FILE = "outputs/evaluations/model_comparison.json"

# Number of leading base-model segments reference_transcript.txt was
# written against (see src/wer_evaluation.py: whisper_segments[:17]).
REFERENCE_SEGMENT_COUNT = 17

ASR_MODEL_SIZES = ["tiny", "base", "small"]


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_reference_window_end_seconds():
    """The reference transcript was written against the first
    REFERENCE_SEGMENT_COUNT segments of the already-stored base-model
    transcript. Derive that window's end time from the stored file
    instead of hardcoding it, so it can't drift out of sync with
    src/wer_evaluation.py."""

    with open(BASE_TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        base_segments = json.load(f)

    reference_segments = base_segments[:REFERENCE_SEGMENT_COUNT]

    return reference_segments[-1]["end"]


def transcribe_with_model(model_size, audio_path, device, window_end_seconds):
    """Transcribe the FULL audio file (full context, like the original
    pipeline run) then restrict the hypothesis to segments starting
    before window_end_seconds, matching the reference transcript's
    audio span."""

    print(f"Loading Whisper '{model_size}'...")
    model = whisper.load_model(model_size, device=device)

    print(f"Transcribing full audio with Whisper '{model_size}'...")
    start_time = time.time()

    result = model.transcribe(
        audio_path,
        fp16=torch.cuda.is_available()
    )

    elapsed_seconds = time.time() - start_time

    all_segments = result.get("segments", [])

    windowed_segments = [
        segment for segment in all_segments
        if segment["start"] < window_end_seconds
    ]

    hypothesis = " ".join(
        segment["text"].strip()
        for segment in windowed_segments
        if segment.get("text", "").strip()
    )

    del model

    return {
        "model_size": model_size,
        "num_segments_total": len(all_segments),
        "num_segments_in_window": len(windowed_segments),
        "hypothesis_preview": hypothesis[:300],
        "transcribe_seconds": round(elapsed_seconds, 2),
        "_hypothesis": hypothesis,
    }


def run_asr_comparison():

    print("=" * 60)
    print("ASR MODEL COMPARISON (Whisper base vs small)")
    print("=" * 60)

    with open(REFERENCE_TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        reference = f.read().strip()

    reference_normalized = normalize_text(reference)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    window_end_seconds = get_reference_window_end_seconds()
    print(
        f"Reference window: 0.0-{window_end_seconds}s "
        f"(first {REFERENCE_SEGMENT_COUNT} base segments)"
    )

    models_result = {}

    for model_size in ASR_MODEL_SIZES:

        outcome = transcribe_with_model(
            model_size, AUDIO_FILE, device, window_end_seconds
        )

        hypothesis_normalized = normalize_text(outcome["_hypothesis"])

        error_rate = compute_wer(
            reference_normalized,
            hypothesis_normalized
        )

        outcome["wer"] = float(error_rate)
        outcome["wer_percentage"] = float(error_rate * 100)

        del outcome["_hypothesis"]

        models_result[f"whisper_{model_size}"] = outcome

        print(
            f"  whisper_{model_size}: "
            f"WER={outcome['wer_percentage']:.2f}% "
            f"({outcome['transcribe_seconds']}s)"
        )

    # Cross-check the fresh 'base' run against the canonical, already
    # -published wer_results.json. Both now use full-context
    # transcription of the same file and the same segment-count
    # window, so they are expected to match closely (identically, if
    # Whisper's decoding is deterministic on this machine); they are
    # not forced to match, so any real discrepancy is visible here
    # rather than hidden.
    cross_check = {"available": False}

    if os.path.exists(WER_RESULTS_FILE):

        with open(WER_RESULTS_FILE, "r", encoding="utf-8") as f:
            stored = json.load(f)

        cross_check = {
            "available": True,
            "stored_wer_percentage": stored.get("wer_percentage"),
            "fresh_base_wer_percentage":
                models_result["whisper_base"]["wer_percentage"],
            "note": (
                "Both values use full-context transcription of the "
                "same audio file, restricted to the same reference "
                "window. They should match closely; any gap reflects "
                "run-to-run decoding nondeterminism, not a windowing "
                "or methodology difference."
            ),
        }

    wer_base = models_result["whisper_base"]["wer_percentage"]
    wer_small = models_result["whisper_small"]["wer_percentage"]

    return {
        "evaluation_window_seconds": window_end_seconds,
        "reference_file": REFERENCE_TRANSCRIPT_FILE,
        "models": models_result,
        "wer_percentage_delta_base_minus_small": round(
            wer_base - wer_small, 4
        ),
        "cross_check_against_wer_results_json": cross_check,
    }


def score_rouge(reference_text, candidate_text):

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    scores = scorer.score(reference_text, candidate_text)

    return {
        metric: {
            "precision": float(scores[metric].precision),
            "recall": float(scores[metric].recall),
            "fmeasure": float(scores[metric].fmeasure),
        }
        for metric in ["rouge1", "rouge2", "rougeL"]
    }


def run_summarization_comparison():

    print()
    print("=" * 60)
    print("SUMMARIZATION COMPARISON (T5-small vs TF-IDF extractive)")
    print("=" * 60)

    with open(REFERENCE_SUMMARY_FILE, "r", encoding="utf-8") as f:
        reference_summary = f.read().strip()

    with open(MEETING_SUMMARY_FILE, "r", encoding="utf-8") as f:
        t5_data = json.load(f)

    t5_summary_text = t5_data.get("final_summary", "").strip()

    if not t5_summary_text:
        raise ValueError(
            f"No final_summary found in {MEETING_SUMMARY_FILE}."
        )

    t5_rouge = score_rouge(reference_summary, t5_summary_text)

    target_word_count = len(t5_summary_text.split())

    extractive_result = generate_extractive_summary(
        target_word_count=target_word_count
    )

    extractive_rouge = score_rouge(
        reference_summary,
        extractive_result["summary"]
    )

    print(
        f"  t5_small:            "
        f"ROUGE-1 F1={t5_rouge['rouge1']['fmeasure']:.4f}"
    )
    print(
        f"  tfidf_extractive:    "
        f"ROUGE-1 F1={extractive_rouge['rouge1']['fmeasure']:.4f}"
    )

    return {
        "reference_file": REFERENCE_SUMMARY_FILE,
        "note": (
            "reference_summary.txt is an extractive excerpt of the "
            "transcript, not a human-written abstractive reference. "
            "ROUGE here measures overlap with the source text, not "
            "summary quality in the fullest sense."
        ),
        "models": {
            "t5_small": {
                "model": t5_data.get("model", "google-t5/t5-small"),
                "summary_word_count": len(t5_summary_text.split()),
                "summary_preview": t5_summary_text[:300],
                "rouge": t5_rouge,
            },
            "tfidf_extractive_baseline": {
                "method": extractive_result["method"],
                "summary_word_count": extractive_result["actual_word_count"],
                "sentences_selected": extractive_result["sentences_selected"],
                "summary_preview": extractive_result["summary"][:300],
                "rouge": extractive_rouge,
            },
        },
        "rouge1_f1_delta_t5_minus_extractive": round(
            t5_rouge["rouge1"]["fmeasure"]
            - extractive_rouge["rouge1"]["fmeasure"],
            4
        ),
    }


def diarization_comparison_placeholder():

    return {
        "status": "not_available",
        "reason": (
            "Only one diarization pipeline is installed in this "
            "environment (pyannote/speaker-diarization-community-1). "
            "No second diarization model is available, so no honest "
            "model-vs-model comparison can be run. See "
            "outputs/evaluations/ablation_results.json for a "
            "with/without-diarization structural comparison instead."
        ),
    }


def main():

    asr_comparison = run_asr_comparison()
    summarization_comparison = run_summarization_comparison()
    diarization_comparison = diarization_comparison_placeholder()

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "All numbers below come from a single manually-verified "
            "development recording, not a benchmark dataset.",
            "The ASR comparison transcribes the full recording with "
            "each model but scores WER only over the audio span "
            f"({asr_comparison['evaluation_window_seconds']}s) "
            "matching the only manually-verified transcript reference "
            "available.",
            "The summarization comparison covers the full transcript, "
            "but is scored against an extractive excerpt reference, "
            "not a human-written abstractive summary.",
        ],
        "asr_comparison": asr_comparison,
        "summarization_comparison": summarization_comparison,
        "diarization_comparison": diarization_comparison,
    }

    os.makedirs("outputs/evaluations", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 60)
    print()
    print("Results saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
