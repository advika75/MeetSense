import os
import json
import torch
import soundfile as sf
import numpy as np

from silero_vad import load_silero_vad, get_speech_timestamps


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "outputs/preprocessed/meeting_16khz_mono.wav"

OUTPUT_DIR = "outputs/vad"

SEGMENTS_FILE = os.path.join(
    OUTPUT_DIR,
    "speech_segments.json"
)

SPEECH_AUDIO_FILE = os.path.join(
    OUTPUT_DIR,
    "speech_only.wav"
)

SAMPLE_RATE = 16000


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(file_path):

    print("\nLoading preprocessed audio...")

    audio, sample_rate = sf.read(file_path)

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Audio samples: {len(audio)}")
    print(
        f"Duration: {len(audio) / sample_rate:.2f} seconds"
    )

    return audio, sample_rate


# ============================================================
# RUN SILERO VAD
# ============================================================

def detect_speech(audio, sample_rate, model=None):
    """Run Silero VAD over an audio array. If model is not given,
    loads a fresh model each call (the original per-call behavior);
    the live pipeline instead loads one model once (via
    load_silero_vad()) and passes it in here on every chunk, to avoid
    reloading the VAD model per chunk."""

    if model is None:

        print("\nLoading Silero VAD model...")

        model = load_silero_vad()

        print("Silero VAD loaded.")

    # Convert NumPy → PyTorch tensor
    audio_tensor = torch.tensor(
        audio,
        dtype=torch.float32
    )

    print("\nDetecting speech regions...")

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sample_rate,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300,
        speech_pad_ms=100
    )

    return speech_timestamps


# ============================================================
# CONVERT TIMESTAMPS TO SECONDS
# ============================================================

def convert_timestamps(speech_timestamps, sample_rate):

    segments = []

    for index, segment in enumerate(speech_timestamps):

        start_sample = segment["start"]
        end_sample = segment["end"]

        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate

        duration = end_time - start_time

        segments.append(
            {
                "segment_id": index,
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "duration": round(duration, 3)
            }
        )

    return segments


# ============================================================
# EXTRACT SPEECH AUDIO
# ============================================================

def create_speech_only_audio(
    audio,
    speech_timestamps,
    sample_rate
):

    print("\nCreating speech-only audio...")

    speech_parts = []

    for segment in speech_timestamps:

        start = segment["start"]
        end = segment["end"]

        speech_parts.append(
            audio[start:end]
        )

    if not speech_parts:

        print("WARNING: No speech detected.")

        return np.array([], dtype=np.float32)

    speech_audio = np.concatenate(
        speech_parts
    )

    return speech_audio


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    segments,
    speech_audio,
    sample_rate
):

    print("\nSaving VAD results...")

    # Save timestamp information
    with open(
        SEGMENTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            segments,
            file,
            indent=4
        )

    # Save speech-only audio
    if len(speech_audio) > 0:

        sf.write(
            SPEECH_AUDIO_FILE,
            speech_audio,
            sample_rate
        )

    print(f"Speech segments saved to:")
    print(SEGMENTS_FILE)

    print(f"\nSpeech-only audio saved to:")
    print(SPEECH_AUDIO_FILE)


# ============================================================
# CALCULATE VAD STATISTICS
# ============================================================

def calculate_statistics(
    segments,
    original_duration
):

    speech_duration = sum(
        segment["duration"]
        for segment in segments
    )

    silence_duration = (
        original_duration - speech_duration
    )

    if original_duration > 0:

        speech_percentage = (
            speech_duration /
            original_duration
        ) * 100

        silence_percentage = (
            silence_duration /
            original_duration
        ) * 100

    else:

        speech_percentage = 0
        silence_percentage = 0

    statistics = {

        "total_audio_duration_seconds":
            round(original_duration, 2),

        "speech_duration_seconds":
            round(speech_duration, 2),

        "silence_duration_seconds":
            round(silence_duration, 2),

        "speech_percentage":
            round(speech_percentage, 2),

        "silence_percentage":
            round(silence_percentage, 2),

        "number_of_speech_segments":
            len(segments)
    }

    return statistics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("DEEP LEARNING VOICE ACTIVITY DETECTION")
    print("=" * 65)

    # Check input
    if not os.path.exists(INPUT_FILE):

        print("\nERROR:")
        print("Preprocessed audio was not found:")
        print(INPUT_FILE)

        print(
            "\nRun Step 2 first:"
        )

        print(
            "python src/audio_preprocessing.py"
        )

        return

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio, sample_rate = load_audio(
        INPUT_FILE
    )

    if sample_rate != SAMPLE_RATE:

        print(
            f"\nWARNING: Expected {SAMPLE_RATE} Hz"
        )

        print(
            f"Found {sample_rate} Hz"
        )

    original_duration = (
        len(audio) / sample_rate
    )

    # --------------------------------------------------------
    # Detect speech
    # --------------------------------------------------------

    speech_timestamps = detect_speech(
        audio,
        sample_rate
    )

    print(
        f"\nDetected "
        f"{len(speech_timestamps)} speech segments."
    )

    # --------------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------------

    segments = convert_timestamps(
        speech_timestamps,
        sample_rate
    )

    # --------------------------------------------------------
    # Create speech-only audio
    # --------------------------------------------------------

    speech_audio = create_speech_only_audio(
        audio,
        speech_timestamps,
        sample_rate
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        segments,
        speech_audio,
        sample_rate
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    statistics = calculate_statistics(
        segments,
        original_duration
    )

    print("\n" + "=" * 65)
    print("VAD STATISTICS")
    print("=" * 65)

    for key, value in statistics.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    statistics_file = os.path.join(
        OUTPUT_DIR,
        "vad_statistics.json"
    )

    with open(
        statistics_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            statistics,
            file,
            indent=4
        )

    print(
        f"\nStatistics saved to:"
    )

    print(statistics_file)

    print("\n" + "=" * 65)
    print("VAD COMPLETE")
    print("=" * 65)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()