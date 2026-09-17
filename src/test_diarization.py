import os
import json
import torch

from dotenv import load_dotenv
from pyannote.audio import Pipeline


# Load .env
load_dotenv(
    dotenv_path="/Users/advikamishra/Desktop/NLP/.env"
)

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN was not found in .env"
    )


# Paths
AUDIO_FILE = "data/audio/meeting.wav"
OUTPUT_FILE = "outputs/transcripts/diarization_test.json"


print("=" * 60)
print("PYANNOTE SPEAKER DIARIZATION TEST")
print("=" * 60)

print("PyTorch:", torch.__version__)
print(
    "MPS available:",
    torch.backends.mps.is_available()
)

# Select device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)
print("Audio:", AUDIO_FILE)


# Load PyAnnote
print()
print("Loading PyAnnote Community-1...")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=HF_TOKEN
)

pipeline.to(device)

print("PyAnnote pipeline loaded successfully.")


# Run diarization
print()
print("Running speaker diarization...")
print("Please wait...")

output = pipeline(AUDIO_FILE)

diarization = output.speaker_diarization


# Extract segments
segments = []

for turn, speaker in diarization:

    segment = {
        "start": round(float(turn.start), 3),
        "end": round(float(turn.end), 3),
        "speaker": str(speaker)
    }

    segments.append(segment)


# Create output folder
os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# Save JSON
with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        segments,
        f,
        indent=2,
        ensure_ascii=False
    )


# Results
speakers = sorted(
    set(
        segment["speaker"]
        for segment in segments
    )
)


print()
print("=" * 60)
print("DIARIZATION COMPLETE")
print("=" * 60)

print(
    "Number of speaker segments:",
    len(segments)
)

print()
print("Detected speakers:")

for speaker in speakers:
    print("  -", speaker)


print()
print("First 10 segments:")

for segment in segments[:10]:

    print(
        f"{segment['start']:8.2f}s - "
        f"{segment['end']:8.2f}s  "
        f"{segment['speaker']}"
    )


print()
print("Saved to:")
print(OUTPUT_FILE)