import os
import json
import time
import queue
import threading
from datetime import datetime

import numpy as np
import sounddevice as sd
import whisper


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000

CHANNELS = 1

MICROPHONE_DEVICE = 1

CHUNK_SECONDS = 10

MODEL_NAME = "base"

OUTPUT_DIR = "outputs/transcripts"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "live_meeting.json"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# AUDIO QUEUE
# ============================================================

audio_queue = queue.Queue()


# ============================================================
# MICROPHONE CALLBACK
# ============================================================

def audio_callback(
    indata,
    frames,
    time_info,
    status
):

    if status:

        print(
            "\nAudio status:",
            status
        )

    audio = indata[:, 0].copy()

    audio_queue.put(audio)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LIVE MEETING TRANSCRIPTION")
    print("=" * 70)

    print("\nLoading Whisper model...")

    model = whisper.load_model(
        MODEL_NAME
    )

    print(
        f"Whisper model loaded: {MODEL_NAME}"
    )

    print(
        f"Microphone device: {MICROPHONE_DEVICE}"
    )

    print(
        f"Sample rate: {SAMPLE_RATE} Hz"
    )

    print(
        f"Chunk size: {CHUNK_SECONDS} seconds"
    )

    print("\nStarting microphone...")

    print(
        "\nSpeak normally into your MacBook microphone."
    )

    print(
        "Press Ctrl+C to stop.\n"
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    live_segments = []

    chunk_number = 0

    meeting_start_time = time.time()

    # --------------------------------------------------------
    # Start microphone
    # --------------------------------------------------------

    with sd.InputStream(

        device=MICROPHONE_DEVICE,

        samplerate=SAMPLE_RATE,

        channels=CHANNELS,

        dtype="float32",

        callback=audio_callback
    ):

        while True:

            chunk_number += 1

            # ------------------------------------------------
            # Collect approximately 10 seconds
            # ------------------------------------------------

            required_samples = (
                SAMPLE_RATE * CHUNK_SECONDS
            )

            audio_parts = []

            collected_samples = 0

            while collected_samples < required_samples:

                audio = audio_queue.get()

                audio_parts.append(
                    audio
                )

                collected_samples += len(
                    audio
                )

            audio_chunk = np.concatenate(
                audio_parts
            )

            # Trim to exact duration
            audio_chunk = audio_chunk[
                :required_samples
            ]

            # ------------------------------------------------
            # Check whether audio is mostly silent
            # ------------------------------------------------

            volume = np.sqrt(
                np.mean(
                    audio_chunk ** 2
                )
            )

            print(
                f"\n[Chunk {chunk_number}] "
                f"Audio level: {volume:.5f}"
            )

            if volume < 0.005:

                print(
                    "No significant speech detected."
                )

                continue

            # ------------------------------------------------
            # Whisper transcription
            # ------------------------------------------------

            print(
                "Transcribing..."
            )

            result = model.transcribe(

                audio_chunk,

                # IMPORTANT:
                # Do NOT force English.
                # Whisper detects the language.
                language=None,

                task="transcribe",

                fp16=False,

                verbose=False,

                condition_on_previous_text=False
            )

            detected_language = result.get(
                "language",
                "unknown"
            )

            print(
                f"Detected language: "
                f"{detected_language}"
            )

            # ------------------------------------------------
            # Extract segments
            # ------------------------------------------------

            text = result.get(
                "text",
                ""
            ).strip()

            if not text:

                print(
                    "No speech recognized."
                )

                continue

            elapsed_time = (
                time.time()
                - meeting_start_time
            )

            segment = {

                "chunk_id":
                    chunk_number,

                "timestamp":
                    round(
                        elapsed_time,
                        2
                    ),

                "language":
                    detected_language,

                "text":
                    text,

                "created_at":
                    datetime.now().isoformat()
            }

            live_segments.append(
                segment
            )

            # ------------------------------------------------
            # Display live transcript
            # ------------------------------------------------

            print(
                "\n" + "-" * 70
            )

            print(
                f"LIVE TRANSCRIPT "
                f"[{elapsed_time:.1f}s]"
            )

            print(
                text
            )

            print(
                "-" * 70
            )

            # ------------------------------------------------
            # Save continuously
            # ------------------------------------------------

            output = {

                "mode":
                    "live",

                "model":
                    MODEL_NAME,

                "sample_rate":
                    SAMPLE_RATE,

                "microphone_device":
                    MICROPHONE_DEVICE,

                "segments":
                    live_segments
            }

            with open(
                OUTPUT_FILE,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    output,
                    file,
                    indent=4,
                    ensure_ascii=False
                )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n\nLive meeting stopped."
        )

        print(
            f"Transcript saved to:"
        )

        print(
            OUTPUT_FILE
        )

    except Exception as error:

        print(
            "\nERROR:"
        )

        print(
            error
        )