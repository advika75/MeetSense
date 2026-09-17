import json
import os

from rouge_score import rouge_scorer


REFERENCE_FILE = "outputs/evaluations/reference_summary.txt"
GENERATED_FILE = "outputs/summaries/meeting_summary.json"
OUTPUT_FILE = "outputs/evaluations/rouge_results.json"


print("=" * 60)
print("ROUGE SUMMARY EVALUATION")
print("=" * 60)


# ---------------------------------------------------------
# Load reference summary
# ---------------------------------------------------------

print()
print("Loading reference summary...")

with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
    reference_summary = f.read().strip()

if not reference_summary:
    raise ValueError("Reference summary is empty.")

print("Reference summary loaded.")


# ---------------------------------------------------------
# Load generated summary
# ---------------------------------------------------------

print()
print("Loading generated T5 summary...")

with open(GENERATED_FILE, "r", encoding="utf-8") as f:
    summary_data = json.load(f)

generated_summary = summary_data.get(
    "final_summary",
    ""
).strip()

if not generated_summary:
    raise ValueError(
        "No final_summary found in meeting_summary.json."
    )

print("Generated summary loaded.")


# ---------------------------------------------------------
# Calculate ROUGE
# ---------------------------------------------------------

print()
print("Calculating ROUGE scores...")

scorer = rouge_scorer.RougeScorer(
    [
        "rouge1",
        "rouge2",
        "rougeL"
    ],
    use_stemmer=True
)

scores = scorer.score(
    reference_summary,
    generated_summary
)


# ---------------------------------------------------------
# Extract scores
# ---------------------------------------------------------

results = {
    "metric": "ROUGE",
    "evaluation_type": "sample_level",
    "reference_file": REFERENCE_FILE,
    "generated_summary_file": GENERATED_FILE,

    "rouge1": {
        "precision": float(scores["rouge1"].precision),
        "recall": float(scores["rouge1"].recall),
        "fmeasure": float(scores["rouge1"].fmeasure)
    },

    "rouge2": {
        "precision": float(scores["rouge2"].precision),
        "recall": float(scores["rouge2"].recall),
        "fmeasure": float(scores["rouge2"].fmeasure)
    },

    "rougeL": {
        "precision": float(scores["rougeL"].precision),
        "recall": float(scores["rougeL"].recall),
        "fmeasure": float(scores["rougeL"].fmeasure)
    }
}


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

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
        indent=2
    )


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print()
print("=" * 60)
print("ROUGE RESULTS")
print("=" * 60)

for metric_name in ["rouge1", "rouge2", "rougeL"]:

    score = scores[metric_name]

    print()
    print(metric_name.upper())

    print(
        f"  Precision: {score.precision:.4f}"
    )

    print(
        f"  Recall:    {score.recall:.4f}"
    )

    print(
        f"  F1:        {score.fmeasure:.4f}"
    )


print()
print("=" * 60)
print("ROUGE EVALUATION COMPLETE")
print("=" * 60)

print()
print("Results saved to:")
print(OUTPUT_FILE)