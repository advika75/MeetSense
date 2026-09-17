import json
import re
import os

from jiwer import wer


REFERENCE_FILE = "outputs/evaluations/reference_transcript.txt"
WHISPER_FILE = "outputs/transcripts/meeting.json"
OUTPUT_FILE = "outputs/evaluations/wer_results.json"


print("=" * 60)
print("WORD ERROR RATE (WER) EVALUATION")
print("=" * 60)


# ---------------------------------------------------------
# Load reference transcript
# ---------------------------------------------------------

print()
print("Loading reference transcript...")

with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
    reference = f.read().strip()

if not reference:
    raise ValueError("Reference transcript is empty.")

print("Reference transcript loaded.")


# ---------------------------------------------------------
# Load Whisper transcript
# ---------------------------------------------------------

print()
print("Loading Whisper transcript...")

with open(WHISPER_FILE, "r", encoding="utf-8") as f:
    whisper_data = json.load(f)


if isinstance(whisper_data, list):

    whisper_segments = whisper_data

elif isinstance(whisper_data, dict):

    whisper_segments = whisper_data.get(
        "segments",
        []
    )

else:

    raise ValueError(
        "Unexpected format in meeting.json"
    )


if not whisper_segments:
    raise ValueError(
        "No Whisper segments found."
    )


# ---------------------------------------------------------
# Use the first 17 Whisper segments
# ---------------------------------------------------------

sample_segments = whisper_segments[:17]

hypothesis_parts = []

for segment in sample_segments:

    text = segment.get("text", "").strip()

    if text:
        hypothesis_parts.append(text)


hypothesis = " ".join(hypothesis_parts)


print(
    "Whisper segments used:",
    len(sample_segments)
)


# ---------------------------------------------------------
# Normalize text
# ---------------------------------------------------------

def normalize_text(text):

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


reference_normalized = normalize_text(
    reference
)

hypothesis_normalized = normalize_text(
    hypothesis
)


# ---------------------------------------------------------
# Calculate WER
# ---------------------------------------------------------

print()
print("Calculating WER...")

error_rate = wer(
    reference_normalized,
    hypothesis_normalized
)

accuracy = max(
    0.0,
    (1.0 - error_rate) * 100
)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results = {
    "metric": "Word Error Rate",
    "metric_name": "WER",
    "evaluation_type": "sample_level",
    "reference_file": REFERENCE_FILE,
    "hypothesis_file": WHISPER_FILE,
    "reference_segments": 17,
    "hypothesis_segments": len(sample_segments),
    "wer": float(error_rate),
    "wer_percentage": float(error_rate * 100),
    "approximate_word_accuracy_percentage": float(accuracy)
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
print("WER RESULTS")
print("=" * 60)

print()
print(
    f"WER: {error_rate:.4f}"
)

print(
    f"WER percentage: "
    f"{error_rate * 100:.2f}%"
)

print(
    f"Approximate word accuracy: "
    f"{accuracy:.2f}%"
)

print()
print("Evaluation type:")
print("Sample-level evaluation")

print()
print("Results saved to:")
print(OUTPUT_FILE)