import json
import os

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate


REFERENCE_FILE = (
    "outputs/evaluations/reference_diarization.json"
)

PREDICTION_FILE = (
    "outputs/transcripts/diarization.json"
)

OUTPUT_FILE = (
    "outputs/evaluations/der_results.json"
)

EVALUATION_END = 75.0


print("=" * 60)
print("DIARIZATION ERROR RATE (DER) EVALUATION")
print("=" * 60)


# ---------------------------------------------------------
# Load reference
# ---------------------------------------------------------

print()
print("Loading manually verified reference...")

with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
    reference_data = json.load(f)

reference = Annotation()

for item in reference_data:

    if not item.get("verified", False):
        raise ValueError(
            "Reference contains unverified segments."
        )

    start = float(item["start"])
    end = float(item["end"])
    speaker = item["speaker"]

    reference[
        Segment(start, end)
    ] = speaker


# ---------------------------------------------------------
# Load PyAnnote prediction
# ---------------------------------------------------------

print("Loading PyAnnote prediction...")

with open(PREDICTION_FILE, "r", encoding="utf-8") as f:
    prediction_data = json.load(f)

prediction = Annotation()

for item in prediction_data:

    start = float(item["start"])
    end = float(item["end"])
    speaker = item["speaker"]

    # Keep only the evaluation period
    if start >= EVALUATION_END:
        continue

    end = min(end, EVALUATION_END)

    if end <= start:
        continue

    prediction[
        Segment(start, end)
    ] = speaker


# ---------------------------------------------------------
# Calculate DER
# ---------------------------------------------------------

print()
print("Calculating DER...")
print("Evaluation period: 6.393 - 75.0 seconds")

metric = DiarizationErrorRate()

der = metric(
    reference,
    prediction
)

components = metric.compute_components(
    reference,
    prediction
)


# ---------------------------------------------------------
# Extract components
# ---------------------------------------------------------

missed_speech = components.get(
    "missed detection",
    0.0
)

false_alarm = components.get(
    "false alarm",
    0.0
)

confusion = components.get(
    "confusion",
    0.0
)

total_reference = (
    missed_speech
    + confusion
    + components.get("total", 0.0)
)

# pyannote's components contain absolute durations.
# Use the metric's DER directly for the final score.


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results = {
    "metric": "Diarization Error Rate",
    "metric_name": "DER",
    "evaluation_type": "sample_level",
    "reference_file": REFERENCE_FILE,
    "prediction_file": PREDICTION_FILE,
    "evaluation_end_seconds": EVALUATION_END,
    "der": float(der),
    "der_percentage": float(der * 100),
    "components": {
        key: float(value)
        for key, value in components.items()
    }
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
        indent=2
    )


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print()
print("=" * 60)
print("DER RESULTS")
print("=" * 60)

print()
print(
    f"DER: {der:.4f}"
)

print(
    f"DER percentage: {der * 100:.2f}%"
)

print()
print("Error components:")

for key, value in components.items():

    print(
        f"  {key}: {value:.4f}"
    )

print()
print("Evaluation type:")
print("Sample-level evaluation")

print()
print("Results saved to:")
print(OUTPUT_FILE)