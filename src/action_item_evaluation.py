import json
import os
import re

from rapidfuzz import fuzz


REFERENCE_FILE = "outputs/evaluations/reference_action_items.txt"
PREDICTION_FILE = "outputs/nlp/action_items_decisions.json"
OUTPUT_FILE = "outputs/evaluations/action_item_results.json"

# Your manually annotated reference covers the beginning
# of the meeting. We evaluate the same 75-second region.
EVALUATION_END = 75.0

# Similarity threshold for matching paraphrased actions.
SIMILARITY_THRESHOLD = 50


print("=" * 60)
print("ACTION ITEM EVALUATION")
print("=" * 60)


# ---------------------------------------------------------
# Load reference
# ---------------------------------------------------------

print()
print("Loading reference action items...")

with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
    reference_lines = [
        line.strip()
        for line in f
        if line.strip()
    ]

if not reference_lines:
    raise ValueError("Reference action-item file is empty.")

print("Reference action items:", len(reference_lines))


# ---------------------------------------------------------
# Load predictions
# ---------------------------------------------------------

print()
print("Loading extracted action items...")

with open(PREDICTION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

predicted_items = data.get("action_items", [])

print("Total predicted action items:", len(predicted_items))


# ---------------------------------------------------------
# Extract prediction information
# ---------------------------------------------------------

def extract_text(item):
    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):
        return (
            item.get("action")
            or item.get("action_item")
            or item.get("text")
            or item.get("sentence")
            or ""
        ).strip()

    return ""


def extract_start(item):
    if not isinstance(item, dict):
        return None

    value = (
        item.get("start")
        if item.get("start") is not None
        else item.get("start_time")
    )

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_end(item):
    if not isinstance(item, dict):
        return None

    value = (
        item.get("end")
        if item.get("end") is not None
        else item.get("end_time")
    )

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------
# Filter predictions to evaluation region
# ---------------------------------------------------------

predictions_in_region = []
predictions_without_timestamps = []

for item in predicted_items:

    text = extract_text(item)

    if not text:
        continue

    start = extract_start(item)
    end = extract_end(item)

    # If timestamps exist, keep actions overlapping
    # the first 75 seconds.
    if start is not None or end is not None:

        if start is None:
            start = 0.0

        if end is None:
            end = start

        if start < EVALUATION_END and end > 0:

            predictions_in_region.append(
                {
                    "text": text,
                    "start": start,
                    "end": end
                }
            )

    else:

        predictions_without_timestamps.append(
            {
                "text": text
            }
        )


# If the extractor doesn't store timestamps, we cannot
# honestly restrict it to 75 seconds.
if predictions_in_region:

    prediction_records = predictions_in_region

    print(
        "Predictions in evaluation region:",
        len(prediction_records)
    )

else:

    prediction_records = predictions_without_timestamps

    print()
    print(
        "WARNING: prediction file does not contain usable "
        "timestamps."
    )

    print(
        "Using all predicted action items."
    )


prediction_texts = [
    item["text"]
    for item in prediction_records
]


# ---------------------------------------------------------
# Text normalization
# ---------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "but",
    "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "be",
    "we", "you", "they", "he", "she",
    "it", "this", "that", "these", "those",
    "have", "has", "had", "do", "does",
    "did", "will", "would", "should",
    "can", "could", "just", "so",
    "very", "really", "also", "then"
}


def normalize(text):

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in STOPWORDS
    ]

    return " ".join(words)


reference_normalized = [
    normalize(item)
    for item in reference_lines
]

prediction_normalized = [
    normalize(item)
    for item in prediction_texts
]


# ---------------------------------------------------------
# Similarity matching
# ---------------------------------------------------------

print()
print("Matching predictions to reference actions...")
print(
    "Similarity threshold:",
    SIMILARITY_THRESHOLD
)

candidate_pairs = []

for p_index, prediction in enumerate(
    prediction_normalized
):

    for r_index, reference in enumerate(
        reference_normalized
    ):

        if not prediction or not reference:
            continue

        token_similarity = fuzz.token_set_ratio(
            prediction,
            reference
        )

        sequence_similarity = fuzz.ratio(
            prediction,
            reference
        )

        # Average the two similarity measures.
        similarity = (
            0.7 * token_similarity
            + 0.3 * sequence_similarity
        )

        candidate_pairs.append(
            (
                similarity,
                p_index,
                r_index
            )
        )


candidate_pairs.sort(
    reverse=True,
    key=lambda x: x[0]
)


matches = []

used_predictions = set()
used_references = set()


for similarity, p_index, r_index in candidate_pairs:

    if similarity < SIMILARITY_THRESHOLD:
        break

    if p_index in used_predictions:
        continue

    if r_index in used_references:
        continue

    used_predictions.add(p_index)
    used_references.add(r_index)

    matches.append(
        {
            "reference": reference_lines[r_index],
            "prediction": prediction_texts[p_index],
            "similarity": round(
                float(similarity),
                2
            )
        }
    )


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

true_positives = len(matches)

false_positives = (
    len(prediction_texts)
    - true_positives
)

false_negatives = (
    len(reference_lines)
    - true_positives
)


precision = (
    true_positives
    / (true_positives + false_positives)
    if true_positives + false_positives > 0
    else 0.0
)

recall = (
    true_positives
    / (true_positives + false_negatives)
    if true_positives + false_negatives > 0
    else 0.0
)

f1 = (
    2 * precision * recall
    / (precision + recall)
    if precision + recall > 0
    else 0.0
)


# ---------------------------------------------------------
# False positives / negatives
# ---------------------------------------------------------

false_positive_items = []

for index, text in enumerate(prediction_texts):

    if index not in used_predictions:

        false_positive_items.append(text)


false_negative_items = []

for index, text in enumerate(reference_lines):

    if index not in used_references:

        false_negative_items.append(text)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results = {
    "metric": "Action Item Extraction",
    "evaluation_type": "sample_level_fuzzy_matching",
    "evaluation_end_seconds": EVALUATION_END,
    "similarity_method": (
        "RapidFuzz token_set_ratio + ratio"
    ),
    "similarity_threshold": SIMILARITY_THRESHOLD,
    "reference_file": REFERENCE_FILE,
    "prediction_file": PREDICTION_FILE,
    "reference_count": len(reference_lines),
    "prediction_count": len(prediction_texts),
    "true_positives": true_positives,
    "false_positives": false_positives,
    "false_negatives": false_negatives,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "matches": matches,
    "false_positive_items": false_positive_items,
    "false_negative_items": false_negative_items
}


os.makedirs(
    "outputs/evaluations",
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print()
print("=" * 60)
print("ACTION ITEM RESULTS")
print("=" * 60)

print()
print("Reference items:", len(reference_lines))
print("Predicted items:", len(prediction_texts))

print()
print("True positives:", true_positives)
print("False positives:", false_positives)
print("False negatives:", false_negatives)

print()
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")


# ---------------------------------------------------------
# Matched items
# ---------------------------------------------------------

print()
print("=" * 60)
print("MATCHED ACTION ITEMS")
print("=" * 60)

if matches:

    for match in matches:

        print()
        print(
            f"Similarity: "
            f"{match['similarity']:.2f}%"
        )

        print(
            "Reference:"
        )

        print(
            f"  {match['reference']}"
        )

        print(
            "Prediction:"
        )

        print(
            f"  {match['prediction']}"
        )

else:

    print("No matches found.")


# ---------------------------------------------------------
# False positives
# ---------------------------------------------------------

print()
print("=" * 60)
print("FALSE POSITIVES")
print("=" * 60)

for item in false_positive_items[:20]:

    print(
        f"- {item}"
    )

if len(false_positive_items) > 20:

    print(
        f"... and "
        f"{len(false_positive_items) - 20} more."
    )


# ---------------------------------------------------------
# False negatives
# ---------------------------------------------------------

print()
print("=" * 60)
print("FALSE NEGATIVES")
print("=" * 60)

for item in false_negative_items:

    print(
        f"- {item}"
    )


print()
print("=" * 60)
print("ACTION ITEM EVALUATION COMPLETE")
print("=" * 60)

print()
print("Results saved to:")
print(OUTPUT_FILE)