"""
Evidence-linked decisions and action items.

Builds an evidence trail from every extracted action item/decision back
to its source transcript segment, using the join key
src/action_item_extraction.py already writes (`segment_id`, a 1-based
index into outputs/transcripts/speaker_transcript.json). No changes to
action_item_extraction.py are made or needed for this -- this script
only reads its existing output and cross-references it.

No confidence score is produced: extraction here is purely rule-based
(regex keyword-pattern matching), not a probabilistic model, so a
"confidence" number would be invented rather than measured. Every
item's `extraction_method` and a human-readable
`detection_method_label` are shown instead, answering "why was this
extracted" without pretending to a certainty score that does not exist.

Honesty note, surfaced explicitly in the output rather than hidden:
today, action_item_extraction.py's clean_action_text() only strips
whitespace -- it does not shorten or paraphrase the matched segment.
So the extracted `text` and the `evidence_text` looked up here are
currently IDENTICAL for every item. Each item carries a
`text_matches_evidence_verbatim` flag documenting this plainly, rather
than presenting two fields that look distinct but aren't.
"""

import json
import os
import sys
from datetime import timedelta


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speaker_analysis import build_speaker_display_names  # noqa: E402


ACTION_ITEMS_FILE = "outputs/nlp/action_items_decisions.json"
SPEAKER_TRANSCRIPT_FILE = "outputs/transcripts/speaker_transcript.json"

OUTPUT_FILE = "outputs/evaluations/evidence_linked_items.json"

DETECTION_METHOD_LABELS = {
    "keyword_pattern": "Rule-based linguistic pattern (keyword match)",
}


def format_timestamp(seconds):

    if seconds is None:
        return "—"

    td = timedelta(seconds=round(seconds))

    total_seconds = int(td.total_seconds())
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def link_item(item, item_type, segments_by_id, display_names):

    segment_id = item.get("segment_id")
    segment = segments_by_id.get(segment_id)

    if segment is not None:

        start = segment.get("start")
        end = segment.get("end")
        evidence_text = segment.get("text", "")
        speaker = segment.get("speaker", item.get("speaker", "UNKNOWN"))

    else:

        # Defensive fallback: the item's own start_time, no end time
        # or evidence text available if the segment can't be found.
        start = item.get("start_time")
        end = None
        evidence_text = ""
        speaker = item.get("speaker", "UNKNOWN")

    extracted_text = (
        item.get("action") if item_type == "action_item"
        else item.get("decision")
    ) or ""

    extraction_method = item.get("extraction_method", "unknown")

    linked = {
        "type": item_type,
        "text": extracted_text,
        "speaker": speaker,
        "speaker_display_name": display_names.get(speaker, speaker),
        "start": start,
        "end": end,
        "timestamp_formatted": (
            f"{format_timestamp(start)} – {format_timestamp(end)}"
            if start is not None and end is not None
            else format_timestamp(start)
        ),
        "evidence_text": evidence_text,
        "text_matches_evidence_verbatim": (
            extracted_text.strip() == evidence_text.strip()
            and bool(evidence_text)
        ),
        "extraction_method": extraction_method,
        "detection_method_label": DETECTION_METHOD_LABELS.get(
            extraction_method, extraction_method
        ),
        "segment_id": segment_id,
        "confidence": None,
        "confidence_note": (
            "No confidence score is produced: this item was extracted "
            "by a rule-based keyword-pattern match, not a "
            "probabilistic model, so there is no defensible way to "
            "compute a numeric confidence."
        ),
    }

    if item_type == "action_item":
        linked["deadline"] = item.get("deadline")

    return linked


def main():

    print("=" * 60)
    print("EVIDENCE-LINKED DECISIONS AND ACTION ITEMS")
    print("=" * 60)

    with open(ACTION_ITEMS_FILE, "r", encoding="utf-8") as f:
        action_data = json.load(f)

    with open(SPEAKER_TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        segments = json.load(f)

    segments_by_id = {
        index + 1: segment for index, segment in enumerate(segments)
    }

    display_names = build_speaker_display_names(segments)

    linked_action_items = [
        link_item(item, "action_item", segments_by_id, display_names)
        for item in action_data.get("action_items", [])
    ]

    linked_decisions = [
        link_item(item, "decision", segments_by_id, display_names)
        for item in action_data.get("decisions", [])
    ]

    verbatim_count = sum(
        1 for item in linked_action_items + linked_decisions
        if item["text_matches_evidence_verbatim"]
    )

    total_count = len(linked_action_items) + len(linked_decisions)

    results = {
        "source": {
            "action_items_decisions_file": ACTION_ITEMS_FILE,
            "speaker_transcript_file": SPEAKER_TRANSCRIPT_FILE,
        },
        "methodology": (
            "Every item is joined back to its source transcript "
            "segment via segment_id (already written by "
            "src/action_item_extraction.py, a 1-based index into "
            "speaker_transcript.json). No new extraction or inference "
            "is performed; this only cross-references existing "
            "outputs. evidence_text is the full text of that source "
            "segment. No confidence score is invented -- extraction "
            "is rule-based, not probabilistic."
        ),
        "verbatim_match_note": (
            f"{verbatim_count} of {total_count} items currently have "
            "text identical to their evidence_text, because "
            "action_item_extraction.py's clean_action_text() only "
            "strips whitespace -- it does not shorten or paraphrase "
            "the matched segment. This is stated plainly rather than "
            "presenting 'text' and 'evidence' as if they were "
            "independently distinct."
        ),
        "action_items": linked_action_items,
        "decisions": linked_decisions,
    }

    os.makedirs("outputs/evaluations", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print(f"Action items linked: {len(linked_action_items)}")
    print(f"Decisions linked: {len(linked_decisions)}")
    print(
        f"Verbatim text==evidence: {verbatim_count}/{total_count}"
    )

    print()
    print("Sample decision:")

    if linked_decisions:

        sample = linked_decisions[0]

        print(f"  Speaker: {sample['speaker_display_name']}")
        print(f"  Timestamp: {sample['timestamp_formatted']}")
        print(f"  Text: {sample['text'][:100]}")
        print(f"  Method: {sample['detection_method_label']}")

    print()
    print("=" * 60)
    print("EVIDENCE LINKING COMPLETE")
    print("=" * 60)
    print()
    print("Results saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
