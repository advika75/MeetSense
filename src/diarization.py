import os
import sys

# --------------------------------------------------
# Fix FFmpeg shared-library path on macOS
# Must happen BEFORE importing torch / pyannote
# --------------------------------------------------
#
# macOS's dynamic linker only honors DYLD_LIBRARY_PATH if it is
# present in the process environment at exec() time -- setting it via
# os.environ from inside an already-running interpreter has no effect
# on later dlopen() calls (confirmed during final validation: this
# silently broke PyAnnote's torchcodec-based audio loader, which could
# not resolve Homebrew FFmpeg's @rpath-referenced libavutil/libavcodec
# libraries even though this block "set" DYLD_LIBRARY_PATH). Re-exec
# this same script once with the corrected environment instead.

ffmpeg_lib = "/opt/homebrew/opt/ffmpeg/lib"

if (
    os.path.exists(ffmpeg_lib)
    and os.environ.get("_MEETSENSE_DYLD_FIXED") != "1"
):
    current_path = os.environ.get("DYLD_LIBRARY_PATH", "")

    if current_path:
        os.environ["DYLD_LIBRARY_PATH"] = (
            ffmpeg_lib + ":" + current_path
        )
    else:
        os.environ["DYLD_LIBRARY_PATH"] = ffmpeg_lib

    os.environ["_MEETSENSE_DYLD_FIXED"] = "1"

    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)


# --------------------------------------------------
# Imports
# --------------------------------------------------

import json
import torch
import numpy as np

from dotenv import load_dotenv
from pyannote.audio import Pipeline


# --------------------------------------------------
# Configuration
# --------------------------------------------------

AUDIO_FILE = "data/audio/meeting.wav"

OUTPUT_FILE = (
    "outputs/transcripts/diarization.json"
)

DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


# --------------------------------------------------
# Device selection
# --------------------------------------------------

def select_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# --------------------------------------------------
# Load pipeline (reused by both batch and live callers)
# --------------------------------------------------

def load_diarization_pipeline(device=None):

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    load_dotenv(
        dotenv_path=os.path.join(project_root, ".env")
    )

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN was not found in .env"
        )

    if device is None:
        device = select_device()

    pipeline = Pipeline.from_pretrained(
        DIARIZATION_MODEL,
        token=hf_token
    )

    pipeline.to(device)

    return pipeline


# --------------------------------------------------
# Run diarization on an audio file / array, return plain segments
# --------------------------------------------------

def diarize_audio(pipeline, audio_source):
    """Run diarization and return a list of
    {"start", "end", "speaker"} dicts with float, second-based
    timestamps relative to the start of audio_source.

    audio_source may be a file path (str) accepted by the pyannote
    pipeline, or any other input the pipeline itself supports.
    """

    output = pipeline(audio_source)

    diarization = output.speaker_diarization

    segments = []

    for turn, speaker in diarization:

        segments.append(
            {
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
                "speaker": str(speaker)
            }
        )

    return segments


def embed_speech_segments(pipeline, audio, segments, sample_rate):
    """Embed a batch of local ``(start, end)`` speech intervals using
    the embedding model already loaded inside the pyannote pipeline.

    ``audio`` is mono float32 audio. Segments are padded to a common
    length for a single model call and accompanied by frame masks so
    padding does not contribute to the embedding.
    """
    import torch

    embedding = getattr(pipeline, "_embedding", None)
    if embedding is None:
        raise RuntimeError("Cached pyannote pipeline has no loaded speaker embedding model")
    target_rate = int(getattr(embedding, "sample_rate", sample_rate))
    waveforms = []
    lengths = []
    for start, end in segments:
        crop = np.asarray(audio[int(start * sample_rate):int(end * sample_rate)], dtype=np.float32)
        if target_rate != sample_rate:
            crop_tensor = torch.from_numpy(crop)[None, None]
            crop = torch.nn.functional.interpolate(
                crop_tensor, size=round(len(crop) * target_rate / sample_rate),
                mode="linear", align_corners=False,
            )[0, 0].numpy()
        waveforms.append(torch.from_numpy(crop))
        lengths.append(len(crop))

    if not waveforms:
        return np.zeros((0, 0), dtype=np.float32)
    max_length = max(lengths)
    batch = torch.zeros((len(waveforms), 1, max_length), dtype=torch.float32)
    for index, waveform in enumerate(waveforms):
        batch[index, 0, :len(waveform)] = waveform
    model = getattr(embedding, "model_", None)
    if model is not None and hasattr(model, "num_frames"):
        frame_count = model.num_frames(max_length)
        masks = torch.zeros((len(waveforms), frame_count), dtype=torch.float32)
        for index, length in enumerate(lengths):
            masks[index, :model.num_frames(length)] = 1.0
        vectors = np.asarray(embedding(batch, masks=masks), dtype=np.float32)
    else:
        vectors = np.asarray(embedding(batch), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


# --------------------------------------------------
# Batch CLI entry point (unchanged behavior)
# --------------------------------------------------

def main():

    device = select_device()

    print("=" * 60)
    print("SPEAKER DIARIZATION")
    print("=" * 60)

    print("PyTorch version:", torch.__version__)
    print("MPS available:", torch.backends.mps.is_available())
    print("Device:", device)
    print("Audio file:", AUDIO_FILE)

    print()
    print("Loading PyAnnote Community-1...")

    pipeline = load_diarization_pipeline(device)

    print("PyAnnote loaded successfully.")

    print()
    print("Running speaker diarization...")
    print("Please wait...")

    segments = diarize_audio(pipeline, AUDIO_FILE)

    os.makedirs(
        "outputs/transcripts",
        exist_ok=True
    )

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


if __name__ == "__main__":
    main()
