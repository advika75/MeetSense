"""
Structured error analysis across ASR, diarization, NLP, action-item
extraction, and summarization.

Every finding here is computed directly from existing output files
(outputs/transcripts/, outputs/nlp/, outputs/summaries/,
outputs/evaluations/) or from the actual regex patterns used in
src/action_item_extraction.py. Where a check relies on a curated word
list (filler/discourse markers used for the NER and keyword-noise
checks), that list is heuristic and is labeled as such in the output
-- it is a lexical approximation, not a human-verified judgment, and
is documented as a "methodology" string next to the finding it
supports.

Nothing here is estimated or invented: every count traces back to a
specific source file, and any question that can't be answered from
existing data (e.g. whether the 34 action items outside the verified
window are correct) is reported as "unknown", not guessed.
"""

import json
import os
import re
import sys
from datetime import datetime, timezone

import jiwer


sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from action_item_extraction import (  # noqa: E402
    ACTION_PATTERNS,
    DECISION_PATTERNS,
    contains_pattern,
)


REFERENCE_TRANSCRIPT_FILE = "outputs/evaluations/reference_transcript.txt"
BASE_TRANSCRIPT_FILE = "outputs/transcripts/meeting.json"

DIARIZATION_FILE = "outputs/transcripts/diarization.json"
REFERENCE_DIARIZATION_FILE = "outputs/evaluations/reference_diarization.json"
DER_RESULTS_FILE = "outputs/evaluations/der_results.json"
SPEAKER_TRANSCRIPT_FILE = "outputs/transcripts/speaker_transcript.json"

NLP_PROCESSED_FILE = "outputs/nlp/nlp_processed_transcript.json"
TFIDF_KEYWORDS_FILE = "outputs/nlp/tfidf_keywords.json"
TOPIC_MODELING_FILE = "outputs/nlp/topic_modeling.json"

ACTION_ITEM_RESULTS_FILE = "outputs/evaluations/action_item_results.json"
ACTION_ITEMS_DECISIONS_FILE = "outputs/nlp/action_items_decisions.json"

MEETING_SUMMARY_FILE = "outputs/summaries/meeting_summary.json"
REFERENCE_SUMMARY_FILE = "outputs/evaluations/reference_summary.txt"

OUTPUT_FILE = "outputs/evaluations/error_analysis.json"

# Heuristic, curated discourse/filler lexicon used only for the
# NER-false-positive and keyword/topic-noise checks below. This is a
# lexical approximation, not a human-verified annotation.
DISCOURSE_MARKERS = {
    "yeah", "okay", "ok", "so", "well", "um", "uh", "umm", "uhh",
    "hmm", "right", "actually", "basically", "literally", "like",
    "oh", "alright", "yes", "no", "sure", "gonna", "kinda", "sort",
    "kind",
}


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lowercase_only(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# ASR ERROR ANALYSIS
# ============================================================

def analyze_asr():

    with open(REFERENCE_TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        reference = f.read().strip()

    with open(BASE_TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        segments = json.load(f)

    sample_segments = segments[:17]
    hypothesis = " ".join(
        s["text"].strip() for s in sample_segments if s.get("text", "").strip()
    )

    reference_normalized = normalize_text(reference)
    hypothesis_normalized = normalize_text(hypothesis)

    word_output = jiwer.process_words(
        reference_normalized, hypothesis_normalized
    )

    ref_words = word_output.references[0]
    hyp_words = word_output.hypotheses[0]

    substitutions, insertions, deletions = [], [], []

    for chunk in word_output.alignments[0]:

        if chunk.type == "substitute":
            for r_idx, h_idx in zip(
                range(chunk.ref_start_idx, chunk.ref_end_idx),
                range(chunk.hyp_start_idx, chunk.hyp_end_idx),
            ):
                substitutions.append(
                    {"reference": ref_words[r_idx], "hypothesis": hyp_words[h_idx]}
                )

        elif chunk.type == "insert":
            for h_idx in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                insertions.append(hyp_words[h_idx])

        elif chunk.type == "delete":
            for r_idx in range(chunk.ref_start_idx, chunk.ref_end_idx):
                deletions.append(ref_words[r_idx])

    # Punctuation impact: same texts, but only lowercased (punctuation
    # kept as separate/attached tokens instead of stripped). The delta
    # against the normalized WER above isolates errors attributable to
    # punctuation/formatting rather than word content.
    reference_raw = lowercase_only(reference)
    hypothesis_raw = lowercase_only(hypothesis)

    wer_raw = jiwer.wer(reference_raw, hypothesis_raw)
    wer_normalized = word_output.wer

    # Filler/discourse word usage: simple word-boundary count, no
    # judgment about whether a given occurrence is "correct".
    filler_terms = [
        "um", "uh", "umm", "uhh", "hmm", "like", "you know", "sort of",
        "kind of", "i mean",
    ]

    filler_counts = {}

    for term in filler_terms:

        pattern = r"\b" + re.escape(term) + r"\b"

        ref_count = len(re.findall(pattern, reference_normalized))
        hyp_count = len(re.findall(pattern, hypothesis_normalized))

        if ref_count or hyp_count:
            filler_counts[term] = {"reference": ref_count, "hypothesis": hyp_count}

    return {
        "source": {
            "reference_file": REFERENCE_TRANSCRIPT_FILE,
            "hypothesis_file": BASE_TRANSCRIPT_FILE,
            "hypothesis_segments_used": len(sample_segments),
        },
        "word_error_rate": {
            "normalized_wer_percentage": round(wer_normalized * 100, 4),
            "punctuation_inclusive_wer_percentage": round(wer_raw * 100, 4),
            "punctuation_attributable_delta_points": round(
                (wer_raw - wer_normalized) * 100, 4
            ),
            "methodology": (
                "normalized_wer strips punctuation before scoring (matches "
                "src/wer_evaluation.py); punctuation_inclusive_wer only "
                "lowercases, leaving punctuation attached to tokens. The "
                "delta approximates how much of the raw error rate comes "
                "from punctuation/formatting choices rather than word "
                "content."
            ),
        },
        "word_level_errors": {
            "substitutions": len(substitutions),
            "insertions": len(insertions),
            "deletions": len(deletions),
            "hits": word_output.hits,
            "substitution_examples": substitutions[:15],
            "insertion_examples": insertions[:15],
            "deletion_examples": deletions[:15],
        },
        "filler_word_usage": {
            "counts": filler_counts,
            "methodology": (
                "Simple word-boundary occurrence counts of a fixed filler "
                "lexicon, not a judgment about correctness of any single "
                "occurrence."
            ),
        },
    }


# ============================================================
# DIARIZATION ERROR ANALYSIS
# ============================================================

def analyze_diarization():

    with open(DER_RESULTS_FILE, "r", encoding="utf-8") as f:
        der_results = json.load(f)

    with open(DIARIZATION_FILE, "r", encoding="utf-8") as f:
        turns = json.load(f)

    turns_sorted = sorted(turns, key=lambda t: t["start"])

    overlap_events = []

    for i in range(len(turns_sorted) - 1):

        current = turns_sorted[i]
        nxt = turns_sorted[i + 1]

        if (
            nxt["start"] < current["end"]
            and nxt["speaker"] != current["speaker"]
        ):
            overlap_events.append({
                "speaker_a": current["speaker"],
                "speaker_a_end": current["end"],
                "speaker_b": nxt["speaker"],
                "speaker_b_start": nxt["start"],
                "overlap_seconds": round(current["end"] - nxt["start"], 3),
            })

    with open(SPEAKER_TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        speaker_segments = json.load(f)

    ratios = []

    for seg in speaker_segments:

        duration = seg["end"] - seg["start"]
        overlap = seg.get("speaker_overlap")

        if duration > 0 and overlap is not None:
            ratios.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
                "text_preview": seg["text"][:80],
                "overlap_ratio": round(overlap / duration, 3),
            })

    ratios_sorted = sorted(ratios, key=lambda r: r["overlap_ratio"])

    low_confidence = [r for r in ratios_sorted if r["overlap_ratio"] < 0.5]

    return {
        "source": {
            "der_results_file": DER_RESULTS_FILE,
            "diarization_file": DIARIZATION_FILE,
            "speaker_transcript_file": SPEAKER_TRANSCRIPT_FILE,
        },
        "speaker_confusion": {
            "confusion_seconds": der_results["components"].get("confusion"),
            "missed_detection_seconds":
                der_results["components"].get("missed detection"),
            "false_alarm_seconds": der_results["components"].get("false alarm"),
            "correct_seconds": der_results["components"].get("correct"),
            "der_percentage": der_results["der_percentage"],
            "evaluation_window": (
                f"{der_results.get('evaluation_end_seconds')}s "
                "(manually-verified reference only)"
            ),
            "methodology": (
                f"Reused directly from {DER_RESULTS_FILE} "
                "(pyannote.metrics DiarizationErrorRate components)."
            ),
        },
        "overlapping_speech": {
            "overlapping_turn_pairs_found": len(overlap_events),
            "total_diarization_turns_checked": len(turns_sorted),
            "examples": overlap_events[:10],
            "methodology": (
                "Adjacent turns (sorted by start time) with different "
                "speakers whose time ranges intersect, checked across the "
                "full predicted diarization output "
                f"({DIARIZATION_FILE}), not just the manually-verified "
                "window."
            ),
        },
        "boundary_confidence": {
            "segments_checked": len(ratios),
            "segments_below_50pct_overlap": len(low_confidence),
            "segments_below_50pct_overlap_percentage": round(
                100 * len(low_confidence) / len(ratios), 2
            ) if ratios else None,
            "lowest_confidence_examples": ratios_sorted[:10],
            "methodology": (
                "speaker_overlap (written by src/merge_transcript.py) is "
                "the number of seconds of a Whisper segment covered by "
                "its assigned diarization turn. overlap_ratio = "
                "speaker_overlap / segment_duration. A low ratio means "
                "the diarization turn only covered part of the ASR "
                "segment -- likely a speaker change or diarization "
                "boundary landing mid-segment, i.e. a plausible boundary "
                "error, not a confirmed one."
            ),
        },
    }


# ============================================================
# NLP ERROR ANALYSIS
# ============================================================

def analyze_nlp():

    with open(NLP_PROCESSED_FILE, "r", encoding="utf-8") as f:
        nlp_data = json.load(f)

    entities = nlp_data.get("named_entities", [])

    flagged_entities = [
        e for e in entities
        if e.get("entity", "").strip().lower() in DISCOURSE_MARKERS
    ]

    with open(TFIDF_KEYWORDS_FILE, "r", encoding="utf-8") as f:
        tfidf_data = json.load(f)

    top_keywords = tfidf_data.get("top_keywords", [])

    flagged_keywords = [
        kw for kw in top_keywords
        if kw.get("term", "").strip().lower() in DISCOURSE_MARKERS
    ]

    with open(TOPIC_MODELING_FILE, "r", encoding="utf-8") as f:
        topic_data = json.load(f)

    topics = topic_data.get("topics", [])

    topic_word_sets = []
    topic_noise = []

    for topic in topics:

        words = [w["word"] for w in topic.get("top_words", [])]
        word_set = set(words)
        topic_word_sets.append(word_set)

        flagged = [w for w in words if w.lower() in DISCOURSE_MARKERS]

        topic_noise.append({
            "topic_id": topic.get("topic_id"),
            "top_words": words,
            "flagged_discourse_words": flagged,
            "flagged_fraction": round(len(flagged) / len(words), 3)
            if words else None,
        })

    pairwise_jaccard = []

    for i in range(len(topic_word_sets)):
        for j in range(i + 1, len(topic_word_sets)):

            a, b = topic_word_sets[i], topic_word_sets[j]
            union = a | b

            jaccard = len(a & b) / len(union) if union else 0.0

            pairwise_jaccard.append({
                "topic_a": topics[i].get("topic_id"),
                "topic_b": topics[j].get("topic_id"),
                "jaccard_similarity": round(jaccard, 3),
            })

    avg_jaccard = (
        sum(p["jaccard_similarity"] for p in pairwise_jaccard)
        / len(pairwise_jaccard)
    ) if pairwise_jaccard else None

    return {
        "source": {
            "nlp_processed_file": NLP_PROCESSED_FILE,
            "tfidf_keywords_file": TFIDF_KEYWORDS_FILE,
            "topic_modeling_file": TOPIC_MODELING_FILE,
        },
        "ner_false_positives": {
            "total_entities": len(entities),
            "flagged_as_likely_false_positive": len(flagged_entities),
            "flagged_fraction": round(
                len(flagged_entities) / len(entities), 3
            ) if entities else None,
            "examples": flagged_entities[:15],
            "methodology": (
                "NER uses NLTK's maxent ne_chunk (src/nlp_preprocessing.py), "
                "a classical statistical chunker, not a neural NER model. "
                "Entities whose text matches a curated discourse-marker "
                "lexicon (e.g. 'Okay', 'Yeah') are flagged as likely false "
                "positives; this is a lexical heuristic, not a human "
                "review of every entity."
            ),
        },
        "keyword_noise": {
            "top_keywords_checked": len(top_keywords),
            "flagged_as_discourse_noise": len(flagged_keywords),
            "flagged_fraction": round(
                len(flagged_keywords) / len(top_keywords), 3
            ) if top_keywords else None,
            "examples": flagged_keywords,
            "root_cause": (
                "src/tfidf_keywords.py builds TfidfVectorizer(lowercase=True, "
                "ngram_range=(1,2), min_df=2, max_df=0.95) with no "
                "stop_words argument set -- no stopword or filler removal "
                "is configured, so high-frequency discourse words compete "
                "directly with content words on raw TF-IDF weight."
            ),
        },
        "topic_quality": {
            "number_of_topics": len(topics),
            "per_topic_discourse_noise": topic_noise,
            "average_pairwise_jaccard_similarity": round(avg_jaccard, 3)
            if avg_jaccard is not None else None,
            "pairwise_jaccard": pairwise_jaccard,
            "root_cause": (
                "src/topic_modeling.py builds CountVectorizer(lowercase=True, "
                "min_df=2, max_df=0.95) with no stop_words argument set -- "
                "same gap as the TF-IDF keyword extractor."
            ),
            "methodology": (
                "average_pairwise_jaccard_similarity is the mean Jaccard "
                "overlap of each pair of topics' top-word sets; higher "
                "values indicate less distinct topics."
            ),
        },
    }


# ============================================================
# ACTION ITEM ERROR ANALYSIS
# ============================================================

def analyze_action_items():

    with open(ACTION_ITEM_RESULTS_FILE, "r", encoding="utf-8") as f:
        eval_results = json.load(f)

    with open(ACTION_ITEMS_DECISIONS_FILE, "r", encoding="utf-8") as f:
        extraction_output = json.load(f)

    all_action_items = extraction_output.get("action_items", [])

    evaluated_count = eval_results["prediction_count"]
    unverified_count = len(all_action_items) - evaluated_count

    false_negatives = eval_results.get("false_negative_items", [])

    coverage_gap_analysis = []

    for reference_text in false_negatives:

        action_match = contains_pattern(reference_text, ACTION_PATTERNS)
        decision_match = contains_pattern(reference_text, DECISION_PATTERNS)

        coverage_gap_analysis.append({
            "missed_reference_item": reference_text,
            "matches_action_pattern": action_match,
            "matches_decision_pattern": decision_match,
            "likely_cause": (
                "pattern_coverage_gap"
                if not action_match and not decision_match
                else "extracted_elsewhere_or_paraphrase_mismatch"
            ),
        })

    matches = eval_results.get("matches", [])

    avg_reference_words = None
    avg_prediction_words_matched = None

    if matches:
        avg_reference_words = sum(
            len(m["reference"].split()) for m in matches
        ) / len(matches)
        avg_prediction_words_matched = sum(
            len(m["prediction"].split()) for m in matches
        ) / len(matches)

    avg_words_all_extracted = (
        sum(
            len(item.get("action", "").split())
            for item in all_action_items
        ) / len(all_action_items)
    ) if all_action_items else None

    return {
        "source": {
            "action_item_results_file": ACTION_ITEM_RESULTS_FILE,
            "action_items_decisions_file": ACTION_ITEMS_DECISIONS_FILE,
        },
        "verified_window": {
            "reference_count": eval_results["reference_count"],
            "true_positives": eval_results["true_positives"],
            "false_positives": eval_results["false_positives"],
            "false_negatives": eval_results["false_negatives"],
            "precision": eval_results["precision"],
            "recall": eval_results["recall"],
            "f1": eval_results["f1"],
        },
        "unverified_extractions": {
            "total_extracted_full_meeting": len(all_action_items),
            "evaluated_against_reference": evaluated_count,
            "unverified_count": unverified_count,
            "note": (
                "These items were extracted by the same keyword-pattern "
                "rules across the full ~30-minute meeting, but no "
                "reference exists outside the first-75-second window. "
                "Their precision is genuinely unknown, not assumed "
                "correct or incorrect."
            ),
        },
        "missed_items_coverage_gap": {
            "false_negative_examples": coverage_gap_analysis,
            "methodology": (
                "Each missed reference item is checked against the actual "
                "ACTION_PATTERNS and DECISION_PATTERNS regex lists from "
                "src/action_item_extraction.py (imported directly, not "
                "re-typed). If neither list matches the reference item's "
                "own wording, that item could never have been extracted "
                "by this rule set regardless of context -- a genuine "
                "pattern-coverage gap rather than an extraction mistake."
            ),
        },
        "extraction_verbosity": {
            "avg_words_per_matched_reference_item": (
                round(avg_reference_words, 1)
                if avg_reference_words is not None else None
            ),
            "avg_words_per_matched_prediction": (
                round(avg_prediction_words_matched, 1)
                if avg_prediction_words_matched is not None else None
            ),
            "avg_words_per_all_extracted_items": (
                round(avg_words_all_extracted, 1)
                if avg_words_all_extracted is not None else None
            ),
            "note": (
                "The rule-based extractor keeps the full matching "
                "transcript segment as the 'action', rather than "
                "distilling a short imperative statement like the "
                "human-written reference items. This is visible directly "
                "in the one matched example, where prediction word count "
                "is much higher than the matched reference's."
            ),
        },
    }


# ============================================================
# SUMMARIZATION ERROR ANALYSIS
# ============================================================

def analyze_summarization():

    with open(MEETING_SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    with open(REFERENCE_SUMMARY_FILE, "r", encoding="utf-8") as f:
        reference_summary = f.read().strip()

    with open(SPEAKER_TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        speaker_segments = json.load(f)

    final_summary = summary_data.get("final_summary", "").strip()

    source_word_count = sum(
        len(seg.get("text", "").split()) for seg in speaker_segments
    )
    summary_word_count = len(final_summary.split())

    compression_ratio = (
        round(source_word_count / summary_word_count, 1)
        if summary_word_count else None
    )

    # Coverage of the manually-verified opening's key facts. This is a
    # fixed, small set of concrete facts drawn directly from
    # reference_summary.txt (not invented), checked for presence in the
    # generated summary.
    key_facts = {
        "moving_to_new_building": ["new building", "moving"],
        "three_week_timeline": ["three weeks"],
        "november_2nd_deadline": ["november"],
        "printer_fax_equipment": ["printer", "fax"],
        "final_administration_approval": ["approval", "administration"],
    }

    final_summary_lower = final_summary.lower()

    coverage = {}

    for fact_name, phrases in key_facts.items():

        found = any(phrase in final_summary_lower for phrase in phrases)
        coverage[fact_name] = {
            "phrases_checked": phrases,
            "present_in_summary": found,
        }

    missing_facts = [k for k, v in coverage.items() if not v["present_in_summary"]]

    # Grounding check: capitalized tokens / numbers in the summary that
    # don't appear anywhere in the source transcript are candidate
    # unsupported details. Sentence-initial capitals are excluded by
    # only checking tokens that are capitalized AND not the first word
    # of a sentence.
    source_text_lower = " ".join(
        seg.get("text", "") for seg in speaker_segments
    ).lower()

    sentences = re.split(r"(?<=[.!?])\s+", final_summary)

    candidate_tokens = []

    for sentence in sentences:

        words = sentence.split()

        for idx, word in enumerate(words):

            cleaned = re.sub(r"[^\w]", "", word)

            if not cleaned:
                continue

            is_number = cleaned.isdigit()
            is_capitalized = cleaned[0].isupper() and idx > 0

            if is_number or is_capitalized:
                candidate_tokens.append(cleaned)

    unsupported_candidates = sorted(set(
        token for token in candidate_tokens
        if token.lower() not in source_text_lower
    ))

    return {
        "source": {
            "meeting_summary_file": MEETING_SUMMARY_FILE,
            "reference_summary_file": REFERENCE_SUMMARY_FILE,
            "speaker_transcript_file": SPEAKER_TRANSCRIPT_FILE,
        },
        "length": {
            "source_word_count_full_meeting": source_word_count,
            "summary_word_count": summary_word_count,
            "compression_ratio": compression_ratio,
            "note": (
                "compression_ratio is source words per summary word, "
                "computed over the full ~30-minute meeting, since T5 "
                "summarizes the entire transcript."
            ),
        },
        "missing_info": {
            "key_facts_checked": coverage,
            "missing_facts": missing_facts,
            "methodology": (
                "A small, fixed set of concrete facts drawn from "
                "reference_summary.txt (the manually-verified excerpt of "
                "the meeting's opening) is checked for presence as a "
                "substring in the generated summary. Absence means the "
                "phrase was not found verbatim/near-verbatim, not "
                "necessarily that the concept is entirely unaddressed."
            ),
        },
        "possible_unsupported_details": {
            "candidate_tokens": unsupported_candidates,
            "methodology": (
                "Capitalized (non-sentence-initial) words and standalone "
                "numbers in the generated summary that do not appear "
                "anywhere in the full source transcript text. This is a "
                "surface lexical grounding check, not a semantic fact "
                "check -- a paraphrase would not be flagged."
            ),
        },
    }


def main():

    print("=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    print()
    print("Analyzing ASR errors...")
    asr = analyze_asr()

    print("Analyzing diarization errors...")
    diarization = analyze_diarization()

    print("Analyzing NLP errors...")
    nlp = analyze_nlp()

    print("Analyzing action-item extraction errors...")
    action_items = analyze_action_items()

    print("Analyzing summarization errors...")
    summarization = analyze_summarization()

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "All findings come from a single manually-verified "
            "development recording, not a benchmark dataset.",
            "NER-false-positive, keyword-noise, and topic-noise checks "
            "use a small curated discourse-marker lexicon as a "
            "heuristic -- they approximate likely errors, they are not "
            "a human-reviewed annotation of every item.",
            "The 'possible_unsupported_details' summarization check is a "
            "surface lexical grounding check (word/number presence), "
            "not a semantic fact-check -- a correct paraphrase can be "
            "flagged, and a genuine hallucination using only "
            "already-seen words would not be.",
            "Action items outside the first-75-second reference window "
            "have never been checked against any ground truth; their "
            "correctness is reported as unknown, not assumed.",
        ],
        "asr": asr,
        "diarization": diarization,
        "nlp": nlp,
        "action_items": action_items,
        "summarization": summarization,
    }

    os.makedirs("outputs/evaluations", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 60)
    print()
    print("Results saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
