import os
import json
import tempfile
from datetime import datetime

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import whisper


# =========================
# SETTINGS
# =========================

SAMPLE_RATE = 16000
CHUNK_SECONDS = 10
MODEL_NAME = "small"

# Always save relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "transcripts",
    "live_meeting.json"
)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# =========================
# LOAD WHISPER
# =========================

print("\nLoading Whisper model...")
print(f"Model: {MODEL_NAME}")

model = whisper.load_model(MODEL_NAME)

print("Whisper model loaded.\n")


# =========================
# TRANSCRIPT STORAGE
# =========================

transcript = []

meeting_start = datetime.now()
audio_time = 0.0


# =========================
# SAVE TRANSCRIPT
# =========================

def save_transcript():

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        json.dump(
            transcript,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("MEETING TRANSCRIPT SAVED")
    print("=" * 60)
    print(f"File: {OUTPUT_FILE}")
    print(f"Segments saved: {len(transcript)}")
    print("=" * 60 + "\n")


# =========================
# LIVE TRANSCRIPTION
# =========================

print("=" * 60)
print("LIVE MEETING TRANSCRIPTION")
print("=" * 60)
print("Speak into your microphone.")
print("Whisper will automatically detect the language.")
print("Press Ctrl+C to stop.")
print("=" * 60)


try:

    while True:

        print("\nListening...")

        # Record audio
        audio = sd.rec(
            int(CHUNK_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        audio = np.squeeze(audio)

        # Convert to WAV-compatible format
        audio_int16 = np.int16(audio * 32767)

        # Temporary WAV file
        temp_path = None

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:

            temp_path = temp_file.name

        write(
            temp_path,
            SAMPLE_RATE,
            audio_int16
        )

        # =========================
        # WHISPER
        # =========================

        print("Transcribing...")

        try:

            result = model.transcribe(
                temp_path,
                fp16=False,
                task="transcribe",
                temperature=0,
                condition_on_previous_text=True
            )

            text = result["text"].strip()

            # Whisper automatically detects language
            detected_language = result.get(
                "language",
                "unknown"
            )

        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        # =========================
        # SAVE RESULT
        # =========================

        if text:

            current_time = datetime.now().strftime(
                "%H:%M:%S"
            )

            segment = {

                "time": current_time,

                "start": audio_time,

                "end": audio_time + CHUNK_SECONDS,

                "language": detected_language,

                "speaker": "UNKNOWN",

                "text": text
            }

            transcript.append(segment)

            print("\n" + "-" * 60)
            print(
                f"[{current_time}] "
                f"Language: {detected_language}"
            )
            print(f"LIVE: {text}")
            print("-" * 60)

            # Save after every segment
            save_transcript()

        else:

            print("No speech detected.")

        audio_time += CHUNK_SECONDS


# =========================
# STOP MEETING
# =========================

except KeyboardInterrupt:

    print("\nStopping live meeting...")

    save_transcript()

    print("Live transcription stopped.")