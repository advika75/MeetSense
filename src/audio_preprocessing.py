import os
import json
import numpy as np
import soundfile as sf
from scipy import signal


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/audio/meeting.wav"

OUTPUT_DIR = "outputs/preprocessed"
OUTPUT_AUDIO = os.path.join(
    OUTPUT_DIR,
    "meeting_16khz_mono.wav"
)

OUTPUT_METADATA = os.path.join(
    OUTPUT_DIR,
    "audio_metadata.json"
)

TARGET_SAMPLE_RATE = 16000


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(file_path):
    print("\nLoading audio...")

    audio, sample_rate = sf.read(file_path)

    print(f"Original sample rate: {sample_rate} Hz")
    print(f"Original shape: {audio.shape}")

    return audio, sample_rate


# ============================================================
# CONVERT TO MONO
# ============================================================

def convert_to_mono(audio):
    print("\nConverting audio to mono...")

    if audio.ndim == 1:
        mono_audio = audio
    else:
        mono_audio = np.mean(audio, axis=1)

    print(f"Mono shape: {mono_audio.shape}")

    return mono_audio


# ============================================================
# RESAMPLE
# ============================================================

def resample_audio(audio, original_rate, target_rate):
    print(
        f"\nResampling audio: "
        f"{original_rate} Hz → {target_rate} Hz"
    )

    if original_rate == target_rate:
        return audio

    number_of_samples = int(
        round(len(audio) * target_rate / original_rate)
    )

    resampled_audio = signal.resample(
        audio,
        number_of_samples
    )

    return resampled_audio


# ============================================================
# NORMALIZE AUDIO
# ============================================================

def normalize_audio(audio):
    print("\nNormalizing audio...")

    max_value = np.max(np.abs(audio))

    if max_value == 0:
        return audio

    normalized_audio = audio / max_value

    return normalized_audio


# ============================================================
# CALCULATE AUDIO STATISTICS
# ============================================================

def calculate_statistics(audio, sample_rate):

    duration = len(audio) / sample_rate

    rms = np.sqrt(np.mean(audio ** 2))

    peak = np.max(np.abs(audio))

    silence_threshold = 0.01

    non_silent_samples = np.sum(
        np.abs(audio) > silence_threshold
    )

    speech_percentage = (
        non_silent_samples / len(audio)
    ) * 100

    silence_percentage = 100 - speech_percentage

    statistics = {
        "sample_rate": sample_rate,
        "duration_seconds": round(duration, 2),
        "duration_minutes": round(duration / 60, 2),
        "rms_energy": round(float(rms), 6),
        "peak_amplitude": round(float(peak), 6),
        "speech_percentage_estimate": round(
            float(speech_percentage),
            2
        ),
        "silence_percentage_estimate": round(
            float(silence_percentage),
            2
        )
    }

    return statistics


# ============================================================
# SAVE AUDIO
# ============================================================

def save_audio(audio, sample_rate, output_file):

    print(f"\nSaving processed audio...")
    print(f"Output: {output_file}")

    sf.write(
        output_file,
        audio,
        sample_rate
    )

    print("Processed audio saved successfully.")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("AUDIO PREPROCESSING PIPELINE")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"\nERROR: Audio file not found:")
        print(INPUT_FILE)
        return

    # Step 1: Load
    audio, original_rate = load_audio(INPUT_FILE)

    original_duration = len(audio) / original_rate

    # Step 2: Mono
    audio = convert_to_mono(audio)

    # Step 3: Resampling
    audio = resample_audio(
        audio,
        original_rate,
        TARGET_SAMPLE_RATE
    )

    # Step 4: Normalization
    audio = normalize_audio(audio)

    # Step 5: Statistics
    statistics = calculate_statistics(
        audio,
        TARGET_SAMPLE_RATE
    )

    # Add original information
    statistics["original_sample_rate"] = original_rate
    statistics["original_duration_seconds"] = round(
        original_duration,
        2
    )

    statistics["target_sample_rate"] = TARGET_SAMPLE_RATE
    statistics["channels"] = 1

    # Step 6: Save audio
    save_audio(
        audio,
        TARGET_SAMPLE_RATE,
        OUTPUT_AUDIO
    )

    # Step 7: Save metadata
    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            statistics,
            file,
            indent=4
        )

    print("\nMetadata saved:")
    print(OUTPUT_METADATA)

    # Display statistics
    print("\n" + "=" * 60)
    print("AUDIO STATISTICS")
    print("=" * 60)

    for key, value in statistics.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()