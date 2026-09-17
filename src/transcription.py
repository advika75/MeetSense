import whisper
import torch
import json
import os


def transcribe_audio(audio_path):

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Load Whisper model
    print("Loading Whisper model...")

    model = whisper.load_model("base", device=device)

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