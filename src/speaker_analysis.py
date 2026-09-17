"""
Speaker contribution analysis.

Aggregates the ALREADY-COMPUTED speaker-aware transcript, per-speaker
TF-IDF keywords, per-speaker topic distribution, and speaker-tagged
action items/decisions into an objective, per-speaker contribution
summary. No new model inference is performed here -- every number is a
direct aggregation of existing pipeline outputs:

  outputs/transcripts/speaker_transcript.json
  outputs/nlp/tfidf_keywords.json
  outputs/nlp/topic_modeling.json
  outputs/nlp/action_items_decisions.json

Explicitly NOT computed or claimed here: personality, competence,
leadership, importance, or performance. Only objective, countable
measures (speaking time, participation percentage, utterance count,
question count, topic/keyword association, and decisions/action items
associated with a speaker) -- see the "objective_measurement_only"
note carried in the output JSON.

PyAnnote's raw cluster labels (SPEAKER_00, SPEAKER_01, ...) are
anonymous and their numbering is not meaningful on its own, so this
script also assigns a stable, human-readable "Speaker N" display label
in first-appearance order. "UNKNOWN" (segments merge_transcript.py
could not confidently assign to any diarization turn) is kept as its
own bucket and is deliberately NOT relabeled as a numbered speaker,
since it is not a real identified participant.
"""

import json
import os
import re
from collections import defaultdict


SPEAKER_TRANSCRIPT_FILE = "outputs/transcripts/speaker_transcript.json"
TFIDF_FILE = "outputs/nlp/tfidf_keywords.json"
TOPIC_FILE = "outputs/nlp/topic_modeling.json"
ACTION_ITEMS_FILE = "outputs/nlp/action_items_decisions.json"

OUTPUT_FILE = "outputs/evaluations/speaker_analysis.json"

TOP_KEYWORDS_PER_SPEAKER = 8
TOP_TOPICS_PER_SPEAKER = 3
MAX_NOTABLE_CONTRIBUTIONS = 5
CONTRIBUTION_PREVIEW_CHARS = 160

# A segment counts as a question if it ends with "?" (Whisper's own
# punctuation, the most direct signal available) or opens with a
# common interrogative word. This is a heuristic, not a grammatical
# parse -- documented as such in the output's "methodology" field.
QUESTION_STARTERS = {
    "what", "why", "how", "who", "when", "where",
    "should", "could", "would", "is", "are", "do", "does", "did",
    "can", "will",
}


def is_question(text):

    stripped = text.strip()

    if not stripped:
        return False

    if stripped.endswith("?"):
        return True

    first_word = re.split(r"\s+", stripped.lower())[0].strip(".,!;:")

    return first_word in QUESTION_STARTERS


def build_speaker_display_names(segments):
    """Stable Speaker N labels in first-appearance order. UNKNOWN is
    kept as its own label, not numbered."""

    display_names = {}
    next_number = 1

    for segment in segments:

        speaker = segment["speaker"]

        if speaker in display_names:
            continue

        if speaker == "UNKNOWN":
            display_names[speaker] = "Unknown / Unmatched"
        else:
            display_names[speaker] = f"Speaker {next_number}"
            next_number += 1

    return display_names


def format_duration(total_seconds):

    total_seconds = int(round(total_seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"


def main():

    print("=" * 60)
    print("SPEAKER CONTRIBUTION ANALYSIS")
    print("=" * 60)

    with open(SPEAKER_TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        segments = json.load(f)

    tfidf_data = {}
    if os.path.exists(TFIDF_FILE):
        with open(TFIDF_FILE, "r", encoding="utf-8") as f:
            tfidf_data = json.load(f)

    topic_data = {}
    if os.path.exists(TOPIC_FILE):
        with open(TOPIC_FILE, "r", encoding="utf-8") as f:
            topic_data = json.load(f)

    action_data = {}
    if os.path.exists(ACTION_ITEMS_FILE):
        with open(ACTION_ITEMS_FILE, "r", encoding="utf-8") as f:
            action_data = json.load(f)

    display_names = build_speaker_display_names(segments)

    # ----------------------------------------------------------------
    # Per-speaker segment grouping
    # ----------------------------------------------------------------

    segments_by_speaker = defaultdict(list)

    for segment in segments:
        segments_by_speaker[segment["speaker"]].append(segment)

    total_speaking_time = sum(
        max(0.0, s["end"] - s["start"]) for s in segments
    )

    # ----------------------------------------------------------------
    # Topic word labels (topic_id -> top words), for readable topic
    # names instead of bare numeric IDs.
    # ----------------------------------------------------------------

    topic_word_labels = {}

    for topic in topic_data.get("topics", []):

        words = [w["word"] for w in topic.get("top_words", [])[:3]]
        topic_word_labels[str(topic.get("topic_id"))] = words

    # ----------------------------------------------------------------
    # Action items / decisions grouped by speaker
    # ----------------------------------------------------------------

    action_items_by_speaker = defaultdict(list)
    decisions_by_speaker = defaultdict(list)

    for item in action_data.get("action_items", []):
        action_items_by_speaker[item.get("speaker", "UNKNOWN")].append(item)

    for item in action_data.get("decisions", []):
        decisions_by_speaker[item.get("speaker", "UNKNOWN")].append(item)

    # ----------------------------------------------------------------
    # Per-speaker aggregation
    # ----------------------------------------------------------------

    speaker_results = []

    for speaker, speaker_segments in segments_by_speaker.items():

        speaking_time = sum(
            max(0.0, s["end"] - s["start"]) for s in speaker_segments
        )

        participation_percentage = (
            round(100 * speaking_time / total_speaking_time, 2)
            if total_speaking_time > 0
            else 0.0
        )

        question_segments = [
            s for s in speaker_segments if is_question(s["text"])
        ]

        speaker_keywords = tfidf_data.get("speaker_keywords", {}).get(
            speaker, []
        )[:TOP_KEYWORDS_PER_SPEAKER]

        speaker_topic_distribution = topic_data.get(
            "speaker_topic_distribution", {}
        ).get(speaker, {})

        top_topics = sorted(
            speaker_topic_distribution.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:TOP_TOPICS_PER_SPEAKER]

        topics_formatted = [
            {
                "topic_id": int(topic_id),
                "top_words": topic_word_labels.get(topic_id, []),
                "segment_count": count,
            }
            for topic_id, count in top_topics
        ]

        speaker_action_items = action_items_by_speaker.get(speaker, [])
        speaker_decisions = decisions_by_speaker.get(speaker, [])

        # Notable contributions: the speaker's own extracted
        # decisions and action items, verbatim (truncated for
        # display), not a generated paraphrase. Decisions first
        # (rarer / typically more significant), then action items,
        # capped at MAX_NOTABLE_CONTRIBUTIONS.
        notable_contributions = []

        for item in speaker_decisions:

            text = item.get("decision", "").strip()

            notable_contributions.append({
                "type": "decision",
                "text": text[:CONTRIBUTION_PREVIEW_CHARS],
                "segment_id": item.get("segment_id"),
            })

        for item in speaker_action_items:

            text = item.get("action", "").strip()

            notable_contributions.append({
                "type": "action_item",
                "text": text[:CONTRIBUTION_PREVIEW_CHARS],
                "segment_id": item.get("segment_id"),
            })

        notable_contributions = notable_contributions[
            :MAX_NOTABLE_CONTRIBUTIONS
        ]

        speaker_results.append({
            "speaker_label": speaker,
            "display_name": display_names[speaker],
            "speaking_time_seconds": round(speaking_time, 2),
            "speaking_time_formatted": format_duration(speaking_time),
            "participation_percentage": participation_percentage,
            "utterance_count": len(speaker_segments),
            "question_count": len(question_segments),
            "topics": topics_formatted,
            "keywords": speaker_keywords,
            "decision_count": len(speaker_decisions),
            "action_item_count": len(speaker_action_items),
            "notable_contributions": notable_contributions,
        })

    # Order by speaking time, descending -- a neutral, objective sort
    # key (not a claim about importance).
    speaker_results.sort(
        key=lambda r: r["speaking_time_seconds"], reverse=True
    )

    results = {
        "source": {
            "speaker_transcript_file": SPEAKER_TRANSCRIPT_FILE,
            "tfidf_keywords_file": TFIDF_FILE,
            "topic_modeling_file": TOPIC_FILE,
            "action_items_decisions_file": ACTION_ITEMS_FILE,
        },
        "total_speaking_time_seconds": round(total_speaking_time, 2),
        "total_segments": len(segments),
        "speaker_count": len([
            s for s in speaker_results if s["speaker_label"] != "UNKNOWN"
        ]),
        "objective_measurement_only": (
            "All figures below are objective counts/measurements "
            "(speaking time, participation percentage, utterance/"
            "question counts, topic and keyword association, and "
            "counts of decisions/action items associated with a "
            "speaker). None of this implies or should be read as a "
            "judgment of personality, competence, leadership, "
            "importance, or performance."
        ),
        "methodology": {
            "speaking_time": "sum(end - start) over that speaker's segments in speaker_transcript.json.",
            "participation_percentage": "speaker's speaking time / total speaking time across ALL segments (including UNKNOWN) * 100.",
            "question_detection": (
                "Heuristic: a segment counts as a question if it ends "
                "with '?' or opens with a common interrogative word "
                "(what/why/how/who/when/where/should/could/would/is/"
                "are/do/does/did/can/will). Not a grammatical parse."
            ),
            "topics": "Reused from outputs/nlp/topic_modeling.json's speaker_topic_distribution (segment counts per LDA topic for that speaker), not recomputed.",
            "keywords": "Reused from outputs/nlp/tfidf_keywords.json's speaker_keywords (already speaker-specific TF-IDF terms), not recomputed. Inherits the same discourse-word noise already documented in error_analysis.json.",
            "decisions_and_action_items": "Grouped from outputs/nlp/action_items_decisions.json by its existing 'speaker' field.",
            "notable_contributions": "The speaker's own extracted decisions/action item text, verbatim (truncated for display), not a generated paraphrase or summary.",
        },
        "speakers": speaker_results,
    }

    os.makedirs("outputs/evaluations", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print(f"Total speaking time: {format_duration(total_speaking_time)}")
    print(f"Speakers found: {results['speaker_count']} (+ UNKNOWN bucket if present)")
    print()

    for speaker in speaker_results:

        print(
            f"{speaker['display_name']:20} "
            f"{speaker['speaking_time_formatted']:>8}  "
            f"{speaker['participation_percentage']:>6.2f}%  "
            f"{speaker['utterance_count']:>4} utterances  "
            f"{speaker['question_count']:>3} questions  "
            f"{speaker['decision_count']} decisions  "
            f"{speaker['action_item_count']} action items"
        )

    print()
    print("=" * 60)
    print("SPEAKER CONTRIBUTION ANALYSIS COMPLETE")
    print("=" * 60)
    print()
    print("Results saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
