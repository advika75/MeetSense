import json
import os


INPUT_FILE = "outputs/transcripts/diarization.json"
OUTPUT_FILE = "outputs/evaluations/reference_diarization.json"

MAX_TIME = 75.0


print("=" * 60)
print("CREATE DER REFERENCE")
print("=" * 60)

print()
print("Loading diarization output...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    diarization = json.load(f)

if not isinstance(diarization, list):
    raise ValueError("diarization.json must contain a list.")

print("Total diarization segments:", len(diarization))

# Keep only the first 75 seconds
sample = []

for segment in diarization:

    start = float(segment.get("start", 0))
    end = float(segment.get("end", 0))

    if start >= MAX_TIME:
        continue

    if end > MAX_TIME:
        end = MAX_TIME

    if end <= start:
        continue

    sample.append(
        {
            "start": start,
            "end": end,
            "speaker": segment.get("speaker", "UNKNOWN"),
            "verified": False
        }
    )


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
        sample,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print("Reference template created.")
print("Time range:", f"0 - {MAX_TIME} seconds")
print("Segments:", len(sample))

print()
print("File:")
print(OUTPUT_FILE)

print()
print("IMPORTANT:")
print("This is only a TEMPLATE.")
print("The speaker labels must be manually verified")
print("by listening to the corresponding audio.")