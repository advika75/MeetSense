"""
Diarization Error Rate (DER) evaluation for the EXPERIMENTAL VBx
diarization backend.

This is a deliberate near-duplicate of src/der_evaluation.py (kept
fully separate rather than parameterizing the original -- see the
implementation plan this was built from), evaluated under the SAME
conditions as the existing PyAnnote DER evaluation:

  - the same audio recording
  - the same manually-verified reference file
    (outputs/evaluations/reference_diarization.json)
  - the same evaluation window (6.393-75.0 seconds)
  - the same metric implementation (pyannote.metrics.diarization.
    DiarizationErrorRate), which performs its own optimal reference-
    to-hypothesis speaker-label matching internally -- so VBx's
    different label scheme (SPEAKER_00, SPEAKER_01, ... assigned by
    this project's own RTTM-to-JSON conversion, not by VBx itself)
    does not need to correspond to PyAnnote's or the reference's
    labels for this to be a valid comparison.

src/der_evaluation.py (the PyAnnote evaluation) and its output
(outputs/evaluations/der_results.json) are not read or modified here.
"""

import json
import os

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate


REFERENCE_FILE = (
    "outputs/evaluations/reference_diarization.json"
)

PREDICTION_FILE = (
    "outputs/transcripts/diarization_vbx.json"
)

OUTPUT_FILE = (
    "outputs/evaluations/der_vbx.json"
)

EVALUATION_END = 75.0


print("=" * 60)
print("DIARIZATION ERROR RATE (DER) EVALUATION -- VBx (experimental)")
print("=" * 60)


# ---------------------------------------------------------
# Load reference
# ---------------------------------------------------------

print()
print("Loading manually verified reference...")

if not os.path.exists(PREDICTION_FILE):
    raise FileNotFoundError(
        f"{PREDICTION_FILE} not found. Run src/diarization_vbx.py first."
    )

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
# Load VBx prediction
# ---------------------------------------------------------

print("Loading VBx prediction...")

with open(PREDICTION_FILE, "r", encoding="utf-8") as f:
    prediction_data = json.load(f)

prediction = Annotation()

for item in prediction_data:

    start = float(item["start"])
    end = float(item["end"])
    speaker = item["speaker"]

    # Keep only the evaluation period (identical windowing to the
    # PyAnnote evaluation, for a like-for-like comparison).
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
# Save results
# ---------------------------------------------------------

results = {
    "metric": "Diarization Error Rate",
    "metric_name": "DER",
    "method": "VBx (AHC+VB-HMM over x-vectors)",
    "evaluation_type": "sample_level",
    "reference_file": REFERENCE_FILE,
    "prediction_file": PREDICTION_FILE,
    "evaluation_end_seconds": EVALUATION_END,
    "der": float(der),
    "der_percentage": float(der * 100),
    "components": {
        key: float(value)
        for key, value in components.items()
    },
    "notes": [
        "Experimental diarization method, evaluated for research "
        "comparison against the production PyAnnote pipeline "
        "(see outputs/evaluations/der_results.json).",
        "VBx requires externally supplied VAD; this run used "
        "MeetSense's existing Silero VAD segments "
        "(outputs/vad/speech_segments.json). PyAnnote's pipeline "
        "performs its own internal segmentation instead, so the two "
        "systems being compared do not receive identical "
        "segmentation input -- a genuine architectural difference, "
        "not an evaluation bias.",
    ],
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
print("VBx DER RESULTS")
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
