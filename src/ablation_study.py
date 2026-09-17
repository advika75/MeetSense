"""
Ablation study: Whisper only -> +VAD -> +diarization -> full pipeline.

Each stage reuses the existing production output where possible
(diarization in particular is expensive and HF-token gated, so it is
never re-run here) and only performs fresh inference where a stage
genuinely needs a result that doesn't already exist -- currently just
"whisper_plus_vad", which transcribes the already-computed VAD
speech-only audio (outputs/vad/speech_only.wav) to isolate VAD's
effect on ASR.

Only metrics that are scientifically meaningful for a given stage are
reported; everything else is explicitly marked "not applicable" with a
reason, rather than fabricated or silently omitted.

Important existing-architecture note (found during the Stage 1 audit,
not introduced by this script): src/transcription.py and
src/diarization.py both read data/audio/meeting.wav directly. VAD's
output is not currently wired into either step. So in the current
production pipeline, "whisper_only" and "whisper_vad_diarization"
share the exact same underlying transcript -- diarization adds speaker
labels, it does not re-transcribe. This script measures VAD's effect
on ASR in isolation (the "whisper_plus_vad" stage) rather than
pretending the production pipeline already chains VAD into ASR, which
it does not.

This is a single manually-verified development recording, not a
benchmark dataset -- see the "limitations" field in the output JSON.
"""

import json
import os
import re
import time
from datetime import datetime, timezone

import torch
import whisper
from jiwer import wer as compute_wer


VAD_SPEECH_ONLY_AUDIO = "outputs/vad/speech_only.wav"
VAD_STATS_FILE = "outputs/vad/vad_statistics.json"

BASE_TRANSCRIPT_FILE = "outputs/transcripts/meeting.json"
SPEAKER_TRANSCRIPT_FILE = "outputs/transcripts/speaker_transcript.json"
DIARIZATION_FILE = "outputs/transcripts/diarization.json"

REFERENCE_TRANSCRIPT_FILE = "outputs/evaluations/reference_transcript.txt"
WER_RESULTS_FILE = "outputs/evaluations/wer_results.json"
DER_RESULTS_FILE = "outputs/evaluations/der_results.json"
ACTION_ITEM_RESULTS_FILE = "outputs/evaluations/action_item_results.json"
ROUGE_RESULTS_FILE = "outputs/evaluations/rouge_results.json"

OUTPUT_FILE = "outputs/evaluations/ablation_results.json"

# Matches the production model (src/transcription.py: whisper.load_model("base")).
WHISPER_MODEL_SIZE = "base"


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def select_by_word_budget(segment_texts, target_word_count):
    """Accumulate segment texts in order until the target word count is
    reached. Used instead of timestamp windowing because VAD removes
    silence and shifts the internal timeline of its output audio, so a
    time-based cutoff would no longer point at the same spoken
    content."""

    selected = []
    word_total = 0

    for text in segment_texts:

        if word_total >= target_word_count:
            break

        selected.append(text)
        word_total += len(text.split())

    return " ".join(selected), word_total


def stage_whisper_only():
    """Reuse the existing production transcript (Whisper base on the
    raw audio) and the already-published WER result computed from it.
    No new inference."""

    with open(WER_RESULTS_FILE, "r", encoding="utf-8") as f:
        stored_wer = json.load(f)

    return {
        "stage": "whisper_only",
        "description": (
            "Whisper base ASR on the original raw audio. No VAD "
            "filtering, no diarization, no downstream NLP."
        ),
        "source": (
            f"Reused from {BASE_TRANSCRIPT_FILE} and {WER_RESULTS_FILE} "
            "(already the production Whisper-only transcript and its "
            "WER; not re-run here)."
        ),
        "metrics": {
            "wer_percentage": stored_wer["wer_percentage"],
            "der_percentage": None,
            "action_item_f1": None,
            "rouge1_f1": None,
        },
        "not_applicable": {
            "der_percentage": "No diarization has been applied yet.",
            "action_item_f1": (
                "Action-item extraction is evaluated once the "
                "speaker-aware transcript exists (see 'full_pipeline')."
            ),
            "rouge1_f1": "Summarization has not run yet at this stage.",
        },
    }


def stage_whisper_plus_vad(reference_normalized, reference_word_budget, device):
    """Transcribe the already-computed VAD speech-only audio
    (produced by the existing src/vad.py, not re-run here) fresh with
    the same Whisper size used in production, to isolate VAD's effect
    on ASR."""

    print(f"Loading Whisper '{WHISPER_MODEL_SIZE}' for VAD-filtered audio...")
    model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)

    print("Transcribing VAD speech-only audio...")
    start_time = time.time()

    result = model.transcribe(
        VAD_SPEECH_ONLY_AUDIO,
        fp16=torch.cuda.is_available()
    )

    elapsed_seconds = time.time() - start_time

    del model

    segment_texts = [
        segment["text"].strip()
        for segment in result.get("segments", [])
        if segment.get("text", "").strip()
    ]

    hypothesis_text, hypothesis_word_count = select_by_word_budget(
        segment_texts, reference_word_budget
    )

    hypothesis_normalized = normalize_text(hypothesis_text)

    error_rate = compute_wer(reference_normalized, hypothesis_normalized)
    wer_percentage = float(error_rate * 100)

    vad_stats = {}

    if os.path.exists(VAD_STATS_FILE):
        with open(VAD_STATS_FILE, "r", encoding="utf-8") as f:
            vad_stats = json.load(f)

    return {
        "stage": "whisper_plus_vad",
        "description": (
            "Whisper base ASR on VAD-filtered speech-only audio "
            "(Silero VAD, silence removed) instead of the raw file. "
            "Diarization and downstream NLP not yet applied."
        ),
        "source": (
            f"Fresh transcription of {VAD_SPEECH_ONLY_AUDIO} performed "
            "by this script; the VAD filtering itself is reused from "
            "the existing outputs/vad/ output (src/vad.py), not "
            "re-run."
        ),
        "vad_stats": vad_stats,
        "transcribe_seconds": round(elapsed_seconds, 2),
        "hypothesis_word_count_used": hypothesis_word_count,
        "hypothesis_preview": hypothesis_text[:300],
        "metrics": {
            "wer_percentage": wer_percentage,
            "der_percentage": None,
            "action_item_f1": None,
            "rouge1_f1": None,
        },
        "not_applicable": {
            "der_percentage": "No diarization has been applied yet.",
            "action_item_f1": (
                "Action-item extraction is evaluated once the "
                "speaker-aware transcript exists (see 'full_pipeline')."
            ),
            "rouge1_f1": "Summarization has not run yet at this stage.",
        },
        "note": (
            "WER is computed over a word-budget-matched prefix of the "
            "VAD-filtered hypothesis (not a timestamp window), because "
            "removing silence shifts the internal timeline of the VAD "
            "output relative to the original audio."
        ),
    }


def stage_plus_diarization(whisper_only_wer_percentage):
    """Reuse the existing production diarization + merge output.
    Diarization is expensive (HF-token-gated PyAnnote pipeline) and
    already computed; it is not re-run here."""

    with open(DER_RESULTS_FILE, "r", encoding="utf-8") as f:
        stored_der = json.load(f)

    return {
        "stage": "whisper_vad_diarization",
        "description": (
            "Adds PyAnnote speaker diarization and speaker/transcript "
            "alignment on top of the Whisper-only transcript, "
            "producing a speaker-aware transcript."
        ),
        "source": (
            f"Reused from {DIARIZATION_FILE}, {SPEAKER_TRANSCRIPT_FILE} "
            f"and {DER_RESULTS_FILE} (already the production "
            "diarization output; not re-run here)."
        ),
        "metrics": {
            "wer_percentage": whisper_only_wer_percentage,
            "der_percentage": stored_der["der_percentage"],
            "action_item_f1": None,
            "rouge1_f1": None,
        },
        "not_applicable": {
            "action_item_f1": (
                "Reserved for 'full_pipeline', where action-item "
                "extraction is evaluated as part of the complete "
                "downstream NLP stage."
            ),
            "rouge1_f1": "Summarization has not run yet at this stage.",
        },
        "note": (
            "wer_percentage is inherited from 'whisper_only': in the "
            "current production pipeline, diarization and merging "
            "operate on the same raw-audio Whisper transcript "
            "(src/transcription.py and src/diarization.py both read "
            "data/audio/meeting.wav directly; VAD output is not wired "
            "into either). Merging speaker labels onto the transcript "
            "does not change the transcribed words."
        ),
    }


def stage_full_pipeline(whisper_only_wer_percentage, der_percentage):
    """Reuse the existing production evaluation outputs for the
    complete pipeline. No new inference."""

    with open(ACTION_ITEM_RESULTS_FILE, "r", encoding="utf-8") as f:
        stored_action = json.load(f)

    with open(ROUGE_RESULTS_FILE, "r", encoding="utf-8") as f:
        stored_rouge = json.load(f)

    return {
        "stage": "full_pipeline",
        "description": (
            "Complete production pipeline: Whisper ASR, diarization, "
            "speaker-aware transcript, NLP preprocessing, TF-IDF "
            "keywords, LDA topics, rule-based action-item/decision "
            "extraction, and T5-small summarization."
        ),
        "source": (
            f"Reused from {ACTION_ITEM_RESULTS_FILE} and "
            f"{ROUGE_RESULTS_FILE} (already the production evaluation "
            "outputs; not re-run here)."
        ),
        "metrics": {
            "wer_percentage": whisper_only_wer_percentage,
            "der_percentage": der_percentage,
            "action_item_f1": stored_action["f1"],
            "rouge1_f1": stored_rouge["rouge1"]["fmeasure"],
        },
        "note": (
            "wer_percentage and der_percentage are inherited from "
            "earlier stages -- neither the transcribed words nor the "
            "speaker assignments change during NLP/summarization."
        ),
    }


def main():

    print("=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)

    with open(REFERENCE_TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        reference = f.read().strip()

    reference_normalized = normalize_text(reference)
    reference_word_budget = len(reference.split())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Reference word budget: {reference_word_budget} words")

    print()
    print("Stage 1/4: whisper_only (reusing existing production output)")
    stage_a = stage_whisper_only()
    print(f"  WER: {stage_a['metrics']['wer_percentage']:.2f}%")

    print()
    print("Stage 2/4: whisper_plus_vad (fresh transcription)")
    stage_b = stage_whisper_plus_vad(
        reference_normalized, reference_word_budget, device
    )
    print(f"  WER: {stage_b['metrics']['wer_percentage']:.2f}%")

    print()
    print(
        "Stage 3/4: whisper_vad_diarization "
        "(reusing existing production output)"
    )
    stage_c = stage_plus_diarization(stage_a["metrics"]["wer_percentage"])
    print(f"  DER: {stage_c['metrics']['der_percentage']:.2f}%")

    print()
    print("Stage 4/4: full_pipeline (reusing existing production output)")
    stage_d = stage_full_pipeline(
        stage_a["metrics"]["wer_percentage"],
        stage_c["metrics"]["der_percentage"],
    )
    print(f"  Action-item F1: {stage_d['metrics']['action_item_f1']:.4f}")
    print(f"  ROUGE-1 F1: {stage_d['metrics']['rouge1_f1']:.4f}")

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "All numbers below come from a single manually-verified "
            "development recording, not a benchmark dataset.",
            "WER is computed over a fixed-size word budget matched to "
            "the manually-verified reference transcript, not the full "
            "meeting.",
            "'whisper_only' and 'whisper_vad_diarization' share the "
            "same underlying transcript in the current production "
            "pipeline, because VAD output is not currently wired into "
            "src/transcription.py or src/diarization.py. This ablation "
            "measures VAD's effect on ASR in isolation ("
            "'whisper_plus_vad'), not VAD's effect on the production "
            "pipeline as currently wired.",
        ],
        "stages": [stage_a, stage_b, stage_c, stage_d],
    }

    os.makedirs("outputs/evaluations", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("ABLATION STUDY COMPLETE")
    print("=" * 60)
    print()
    print("Results saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
