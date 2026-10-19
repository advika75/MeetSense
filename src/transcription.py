import whisper
import torch
import json
import os


def load_whisper_model(model_name="base", device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return whisper.load_model(model_name, device=device)


def transcribe_audio(audio_path, model=None):
    """Transcribe a file path or a raw audio array/tensor. If model is
    not given, loads a fresh "base" model each call (the original
    per-call behavior); the live pipeline instead loads one model
    once via load_whisper_model() and passes it in here on every
    chunk, to avoid reloading Whisper per chunk."""

    if model is None:

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {device}")
        print("Loading Whisper model...")

        model = load_whisper_model("base", device=device)

    print("Transcribing audio...")

    result = model.transcribe(
        audio_path,
        fp16=torch.cuda.is_available()
    )

    return result


def save_transcript(result, output_path):

    segments = []

    for segment in result["segments"]:

        segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip()
        })

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            segments,
            file,
            indent=4,
            ensure_ascii=False
        )


if __name__ == "__main__":

    audio_path = "data/audio/meeting.wav"

    output_path = "outputs/transcripts/meeting.json"

    # Check file
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        exit()

    result = transcribe_audio(audio_path)

    save_transcript(result, output_path)

    print("\n========== TRANSCRIPT ==========\n")

    for segment in result["segments"]:

        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()

        print(
            f"[{start:.2f} - {end:.2f}] {text}"
        )

    print("\nTranscript saved to:")
    print(output_path)