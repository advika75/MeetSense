import os
import json
import tempfile
from datetime import datetime

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from dotenv import load_dotenv
from pyannote.audio import Pipeline


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000

# Record this much audio at a time
CHUNK_SECONDS = 10

# Run speaker diarization on this much accumulated audio
DIARIZATION_WINDOW_SECONDS = 30

# Whisper model
MODEL_NAME = "small"

# PyAnnote model
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "transcripts"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "live_speaker_transcript.json"
)

TEMP_AUDIO_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "temp_audio"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(
    os.path.join(BASE_DIR, ".env")
)

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN was not found in .env"
    )


# ============================================================
# LOAD WHISPER
# ============================================================

print()
print("=" * 70)
print("Loading Whisper...")
print("=" * 70)

whisper_model = whisper.load_model(
    MODEL_NAME
)

print("Whisper loaded.")


# ============================================================
# LOAD PYANNOTE
# ============================================================

print()
print("=" * 70)
print("Loading PyAnnote...")
print("=" * 70)

diarization_pipeline = Pipeline.from_pretrained(
    DIARIZATION_MODEL,
    token=HF_TOKEN
)

print("PyAnnote loaded.")


# ============================================================
# STORAGE
# ============================================================

transcript = []

audio_buffer = []

total_audio_time = 0.0


# ============================================================
# SAVE JSON
# ============================================================

def save_transcript():

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            transcript,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print("LIVE SPEAKER TRANSCRIPT SAVED")
    print("=" * 70)
    print(f"File: {OUTPUT_FILE}")
    print(f"Segments: {len(transcript)}")
    print("=" * 70)


# ============================================================
# RECORD AUDIO CHUNK
# ============================================================

def record_chunk():

    print()
    print(
        f"🎙️ Listening for {CHUNK_SECONDS} seconds..."
    )

    audio = sd.rec(
        int(CHUNK_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = np.squeeze(audio)

    return audio


# ============================================================
# SAVE TEMP WAV
# ============================================================

def save_temp_wav(audio):

    audio_int16 = np.int16(
        np.clip(audio, -1, 1) * 32767
    )

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        dir=TEMP_AUDIO_DIR,
        delete=False
    )

    temp_path = temp_file.name

    temp_file.close()

    write(
        temp_path,
        SAMPLE_RATE,
        audio_int16
    )

    return temp_path


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcribe_audio(
    audio_path,
    offset_time
):

    print("📝 Transcribing...")

    result = whisper_model.transcribe(
        audio_path,
        fp16=False,
        task="transcribe",
        temperature=0,
        condition_on_previous_text=True,
        verbose=False
    )

    detected_language = result.get(
        "language",
        "unknown"
    )

    segments = []

    for segment in result.get(
        "segments",
        []
    ):

        text = segment.get(
            "text",
            ""
        ).strip()

        if not text:
            continue

        start = (
            offset_time
            + segment["start"]
        )

        end = (
            offset_time
            + segment["end"]
        )

        segments.append(
            {
                "start": start,
                "end": end,
                "language": detected_language,
                "text": text
            }
        )

    return segments


# ============================================================
# DIARIZATION
# ============================================================

def diarize_audio(audio_path):

    print("👥 Detecting speakers...")

    output = diarization_pipeline(
        audio_path
    )

    # Community-1 returns a DiarizeOutput object.
    # The actual speaker diarization is inside
    # output.speaker_diarization.

    diarization = output.speaker_diarization

    speaker_segments = []

    for turn, _, speaker in diarization.itertracks(
        yield_label=True
    ):

        speaker_segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker
            }
        )

    print(
        f"👥 Speakers detected: "
        f"{len(set(s['speaker'] for s in speaker_segments))}"
    )

    return speaker_segments

# ============================================================
# FIND SPEAKER
# ============================================================

def find_speaker(
    transcript_start,
    transcript_end,
    speaker_segments
):

    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for speaker_segment in speaker_segments:

        overlap_start = max(
            transcript_start,
            speaker_segment["start"]
        )

        overlap_end = min(
            transcript_end,
            speaker_segment["end"]
        )

        overlap = max(
            0.0,
            overlap_end - overlap_start
        )

        if overlap > best_overlap:

            best_overlap = overlap

            best_speaker = (
                speaker_segment["speaker"]
            )

    return best_speaker


# ============================================================
# PROCESS WINDOW
# ============================================================

def process_window(
    audio,
    window_start_time
):

    if len(audio) == 0:
        return

    wav_path = save_temp_wav(
        audio
    )

    try:

        # ----------------------------------------------------
        # Whisper
        # ----------------------------------------------------

        whisper_segments = transcribe_audio(
            wav_path,
            window_start_time
        )

        if not whisper_segments:

            print("No speech detected.")

            return

        # ----------------------------------------------------
        # PyAnnote
        # ----------------------------------------------------

        speaker_segments = diarize_audio(
            wav_path
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        for segment in whisper_segments:

            # PyAnnote timestamps are relative
            # to this audio window.
            relative_start = (
                segment["start"]
                - window_start_time
            )

            relative_end = (
                segment["end"]
                - window_start_time
            )

            speaker = find_speaker(
                relative_start,
                relative_end,
                speaker_segments
            )

            final_segment = {
                "time": datetime.now().strftime(
                    "%H:%M:%S"
                ),
                "start": round(
                    segment["start"],
                    2
                ),
                "end": round(
                    segment["end"],
                    2
                ),
                "language": segment[
                    "language"
                ],
                "speaker": speaker,
                "text": segment["text"]
            }

            transcript.append(
                final_segment
            )

            print()
            print("-" * 70)
            print(
                f"[{final_segment['start']:.2f}s - "
                f"{final_segment['end']:.2f}s]"
            )
            print(
                f"🗣️ {speaker}"
            )
            print(
                f"🌐 Language: "
                f"{final_segment['language']}"
            )
            print(
                f"💬 {final_segment['text']}"
            )
            print("-" * 70)

        save_transcript()

    finally:

        if os.path.exists(wav_path):

            os.remove(
                wav_path
            )


# ============================================================
# MAIN LIVE LOOP
# ============================================================

print()
print("=" * 70)
print("🎙️ SPEAKER-AWARE LIVE MEETING TRANSCRIPTION")
print("=" * 70)
print()
print("Whisper: automatic language detection")
print("Speaker diarization: PyAnnote")
print(f"Whisper model: {MODEL_NAME}")
print(
    f"Diarization window: "
    f"{DIARIZATION_WINDOW_SECONDS} seconds"
)
print()
print("Speak into the microphone.")
print("Press Ctrl+C to stop.")
print("=" * 70)


try:

    while True:

        # Record 10 seconds
        chunk = record_chunk()

        audio_buffer.append(
            chunk
        )

        total_audio_time += CHUNK_SECONDS

        # ----------------------------------------------------
        # Every 30 seconds, process accumulated audio
        # ----------------------------------------------------

        if total_audio_time % DIARIZATION_WINDOW_SECONDS == 0:

            print()
            print()
            print("#" * 70)
            print(
                "🔄 PROCESSING SPEAKER WINDOW"
            )
            print("#" * 70)

            combined_audio = np.concatenate(
                audio_buffer
            )

            window_start = (
                total_audio_time
                - DIARIZATION_WINDOW_SECONDS
            )

            process_window(
                combined_audio,
                window_start
            )

            # Clear processed audio
            audio_buffer = []


except KeyboardInterrupt:

    print()
    print("=" * 70)
    print("Stopping meeting...")
    print("=" * 70)

    # Process remaining audio
    if audio_buffer:

        print(
            "Processing remaining audio..."
        )

        remaining_audio = np.concatenate(
            audio_buffer
        )

        remaining_duration = (
            len(remaining_audio)
            / SAMPLE_RATE
        )

        window_start = (
            total_audio_time
            - remaining_duration
        )

        process_window(
            remaining_audio,
            window_start
        )

    save_transcript()

    print()
    print("✅ Live meeting stopped.")
    print(
        f"Transcript saved to:\n{OUTPUT_FILE}"
    )