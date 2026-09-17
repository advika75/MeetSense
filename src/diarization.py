import os

# --------------------------------------------------
# Fix FFmpeg shared-library path on macOS
# Must happen BEFORE importing torch / pyannote
# --------------------------------------------------

ffmpeg_lib = "/opt/homebrew/opt/ffmpeg/lib"

if os.path.exists(ffmpeg_lib):
    current_path = os.environ.get("DYLD_LIBRARY_PATH", "")

    if current_path:
        os.environ["DYLD_LIBRARY_PATH"] = (
            ffmpeg_lib + ":" + current_path
        )
    else:
        os.environ["DYLD_LIBRARY_PATH"] = ffmpeg_lib


# --------------------------------------------------
# Imports
# --------------------------------------------------

import json
import torch

from dotenv import load_dotenv
from pyannote.audio import Pipeline


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv(
    dotenv_path="/Users/advikamishra/Desktop/NLP/.env"
)

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN was not found in .env"
    )


# --------------------------------------------------
# Configuration
# --------------------------------------------------

AUDIO_FILE = "data/audio/meeting.wav"

OUTPUT_FILE = (
    "outputs/transcripts/diarization.json"
)


# --------------------------------------------------
# Device
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# --------------------------------------------------
# Start
# --------------------------------------------------

print("=" * 60)
print("SPEAKER DIARIZATION")
print("=" * 60)

print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("Device:", device)
print("Audio file:", AUDIO_FILE)


# --------------------------------------------------
# Load PyAnnote Community-1
# --------------------------------------------------

print()
print("Loading PyAnnote Community-1...")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=HF_TOKEN
)

pipeline.to(device)

print("PyAnnote loaded successfully.")


# --------------------------------------------------
# Run diarization
# --------------------------------------------------

print()
print("Running speaker diarization...")
print("Please wait...")

output = pipeline(AUDIO_FILE)

diarization = output.speaker_diarization


# --------------------------------------------------
# Extract speaker segments
# --------------------------------------------------

segments = []

for turn, speaker in diarization:

    segments.append(
        {
            "start": round(float(turn.start), 3),
            "end": round(float(turn.end), 3),
            "speaker": str(speaker)
        }
    )


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

os.makedirs(
    "outputs/transcripts",
    exist_ok=True
)


# --------------------------------------------------
# Save diarization
# --------------------------------------------------

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


# --------------------------------------------------
# Statistics
# --------------------------------------------------

speakers = sorted(
    set(
        segment["speaker"]
        for segment in segments
    )
)


speaker_durations = {}

for segment in segments:

    speaker = segment["speaker"]

    duration = (
        segment["end"] -
        segment["start"]
    )

    speaker_durations[speaker] = (
        speaker_durations.get(speaker, 0)
        + duration
    )


# --------------------------------------------------
# Display results
# --------------------------------------------------

print()
print("=" * 60)
print("DIARIZATION COMPLETE")
print("=" * 60)

print()
print("Number of speaker segments:", len(segments))

print()
print("Number of speakers:", len(speakers))

print()
print("Detected speakers:")

for speaker in speakers:

    duration = speaker_durations[speaker]

    print(
        f"  {speaker}: "
        f"{duration:.2f} seconds"
    )


print()
print("First 10 speaker segments:")

for segment in segments[:10]:

    print(
        f"{segment['start']:8.2f}s - "
        f"{segment['end']:8.2f}s   "
        f"{segment['speaker']}"
    )


print()
print("Output saved to:")
print(OUTPUT_FILE)