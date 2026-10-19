"""
Evaluates TakeawayExtractor output (a report.json) against a hand-
labelled ground truth JSON. A predicted item matches a labelled item
if EITHER its evidence timestamp is within +-30s of the labelled
time_sec, OR its text is a fuzzy match (rapidfuzz token_sort_ratio
>= 80), whichever is true first. Computes precision/recall/F1 per
category (decisions/action_items/deadlines) for each extractor's
report.json, and writes outputs/eval/takeaways_eval.md.

Usage:
    python scripts/eval_takeaways.py \\
        --labels path/to/labelled.json \\
        --report rule_based=outputs/reports/meeting_001/report.json \\
        --report llm=outputs/reports/meeting_001_llm/report.json

Build two separate report.json files to compare (e.g. two
meeting_ids, one built with each extractor_name) -- build_report()
saves to a single path per meeting_id today.

Labelled JSON shape (times in seconds):
{
  "decisions": [{"text": "...", "time_sec": 123.0}, ...],
  "action_items": [{"text": "...", "time_sec": 456.0}, ...],
  "deadlines": [{"text": "...", "time_sec": 789.0}, ...]
}
"""

import argparse
import json
import os

from rapidfuzz import fuzz


TIME_WINDOW_SECONDS = 30
FUZZY_MATCH_THRESHOLD = 80  # rapidfuzz score, 0-100

CATEGORIES = [
    "decisions",
    "action_items",
    "deadlines_strict",
    "deadlines_lenient",
]


def load_json(path):

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def item_timestamp(item):
    """Predicted report items nest their time under evidence.timestamp."""

    evidence = item.get("evidence", {})
    return evidence.get("timestamp")


def is_match(predicted_item, labelled_item):

    predicted_time = item_timestamp(predicted_item)
    labelled_time = labelled_item.get("time_sec")

    if predicted_time is not None and labelled_time is not None:

        if abs(predicted_time - labelled_time) <= TIME_WINDOW_SECONDS:
            return True

    predicted_text = predicted_item.get("text", "")
    labelled_text = labelled_item.get("text", "")

    similarity = fuzz.token_sort_ratio(predicted_text, labelled_text)

    return similarity >= FUZZY_MATCH_THRESHOLD


def score_category(predicted_items, labelled_items):
    """Greedy one-to-one matching: each labelled item is claimed by
    at most one predicted item, and vice versa."""

    matched_labelled_indices = set()
    matched_predicted_indices = set()

    for p_index, predicted_item in enumerate(predicted_items):

        for l_index, labelled_item in enumerate(labelled_items):

            if l_index in matched_labelled_indices:
                continue

            if is_match(predicted_item, labelled_item):

                matched_labelled_indices.add(l_index)
                matched_predicted_indices.add(p_index)
                break

    true_positives = len(matched_predicted_indices)
    false_positives = len(predicted_items) - true_positives
    false_negatives = len(labelled_items) - len(matched_labelled_indices)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positive_items": [
            item for index, item in enumerate(predicted_items)
            if index not in matched_predicted_indices
        ],
        "missed_items": [
            item for index, item in enumerate(labelled_items)
            if index not in matched_labelled_indices
        ],
    }


def evaluate_report(report_path, labels):

    report = load_json(report_path)

    results = {}

    results["decisions"] = score_category(
        report.get("decisions", []), labels.get("decisions", [])
    )
    results["action_items"] = score_category(
        report.get("action_items", []), labels.get("action_items", [])
    )

    strict_deadlines = report.get("deadlines", [])
    lenient_deadlines = list(strict_deadlines)
    for category in ("decisions", "action_items"):
        for item in report.get(category, []):
            if item.get("due_date") or item.get("due_date_text"):
                duplicate = any(
                    item_timestamp(item) is not None
                    and item_timestamp(existing) is not None
                    and abs(item_timestamp(item) - item_timestamp(existing)) < 0.01
                    and fuzz.token_sort_ratio(
                        item.get("text", ""), existing.get("text", "")
                    ) >= FUZZY_MATCH_THRESHOLD
                    for existing in lenient_deadlines
                )
                if not duplicate:
                    lenient_deadlines.append(item)

    labelled_deadlines = labels.get("deadlines", [])
    results["deadlines_strict"] = score_category(
        strict_deadlines, labelled_deadlines
    )
    results["deadlines_lenient"] = score_category(
        lenient_deadlines, labelled_deadlines
    )

    return results


def render_markdown(all_results, labels_path):

    lines = [
        "# Takeaway Extraction Evaluation",
        "",
        f"Labels: `{labels_path}`",
        f"Matching: timestamp window +/-{TIME_WINDOW_SECONDS}s OR "
        f"fuzzy text similarity >= {FUZZY_MATCH_THRESHOLD}",
        "",
    ]

    extractors = list(all_results)
    headers = ["Category"]
    for name in extractors:
        headers.extend([f"{name} Precision", f"{name} Recall", f"{name} F1"])

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    category_labels = {
        "decisions": "Decisions",
        "action_items": "Action items",
        "deadlines_strict": "Deadlines (strict)",
        "deadlines_lenient": "Deadlines (lenient)",
    }
    for category in CATEGORIES:
        row = [category_labels[category]]
        for name in extractors:
            result = all_results[name][category]
            row.extend([
                str(result["precision"]),
                str(result["recall"]),
                str(result["f1"]),
            ])
        lines.append("| " + " | ".join(row) + " |")

    for name, results in all_results.items():
        lines.extend(["", f"## {name}", "", "### False positives"])
        for category in CATEGORIES:
            items = results[category]["false_positive_items"]
            if items:
                lines.append(f"- **{category_labels[category]}**")
                lines.extend(f"  - {format_error_item(item)}" for item in items)
        lines.extend(["", "### Missed items"])
        for category in CATEGORIES:
            items = results[category]["missed_items"]
            if items:
                lines.append(f"- **{category_labels[category]}**")
                lines.extend(f"  - {format_error_item(item, labelled=True)}" for item in items)

    return "\n".join(lines)


def format_error_item(item, labelled=False):

    timestamp = item.get("time_sec") if labelled else item_timestamp(item)
    when = f"{timestamp:.1f}s" if isinstance(timestamp, (int, float)) else "time unknown"
    return f"{when} — {item.get('text', '')}"


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate takeaway extractors against labelled ground truth."
        )
    )

    parser.add_argument(
        "--labels", required=True, help="Path to labelled JSON"
    )

    parser.add_argument(
        "--report",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="extractor_name=path/to/report.json (repeatable)",
    )

    parser.add_argument(
        "--output",
        default="outputs/eval/takeaways_eval.md",
    )

    args = parser.parse_args()

    labels = load_json(args.labels)

    all_results = {}

    for entry in args.report:

        name, _, path = entry.partition("=")
        all_results[name] = evaluate_report(path, labels)

    markdown = render_markdown(all_results, args.labels)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as file:
        file.write(markdown)

    print(f"Wrote {args.output}")
    print()
    print(markdown)


if __name__ == "__main__":
    main()
