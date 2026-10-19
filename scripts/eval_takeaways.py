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

CATEGORIES = ["decisions", "action_items", "deadlines"]


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
    }


def evaluate_report(report_path, labels):

    report = load_json(report_path)

    results = {}

    for category in CATEGORIES:

        predicted_items = report.get(category, [])
        labelled_items = labels.get(category, [])

        results[category] = score_category(predicted_items, labelled_items)

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

    for category in CATEGORIES:
        row = [category]
        for name in extractors:
            result = all_results[name][category]
            row.extend([
                str(result["precision"]),
                str(result["recall"]),
                str(result["f1"]),
            ])
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


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
