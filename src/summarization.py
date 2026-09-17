import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


INPUT_FILE = "outputs/transcripts/speaker_transcript.json"
OUTPUT_FILE = "outputs/summaries/meeting_summary.json"

MODEL_NAME = "google-t5/t5-small"

MAX_INPUT_TOKENS = 450
MAX_OUTPUT_TOKENS = 120


print("=" * 60)
print("TRANSFORMER MEETING SUMMARIZATION")
print("=" * 60)

print()
print("Loading speaker-aware transcript...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    transcript = json.load(f)

if not isinstance(transcript, list):
    raise ValueError(
        "speaker_transcript.json must contain a list."
    )

print("Transcript segments:", len(transcript))


# ---------------------------------------------------------
# Build meeting text
# ---------------------------------------------------------

meeting_parts = []

for segment in transcript:

    speaker = segment.get("speaker", "UNKNOWN")
    text = segment.get("text", "").strip()

    if text:

        meeting_parts.append(
            f"{speaker}: {text}"
        )


meeting_text = " ".join(meeting_parts)

print("Meeting text characters:", len(meeting_text))


# ---------------------------------------------------------
# Device selection
# ---------------------------------------------------------

if torch.backends.mps.is_available():

    device = torch.device("mps")
    print("Using Apple Silicon GPU: MPS")

else:

    device = torch.device("cpu")
    print("MPS unavailable. Using CPU.")


# ---------------------------------------------------------
# Load tokenizer and model
# ---------------------------------------------------------

print()
print("Loading transformer model:")
print(MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

print("Transformer model loaded.")


# ---------------------------------------------------------
# Split meeting into manageable chunks
# ---------------------------------------------------------

def create_chunks(text):

    words = text.split()

    chunks = []

    current_chunk = []

    current_tokens = 0

    for word in words:

        test_chunk = current_chunk + [word]

        encoded = tokenizer(
            "summarize: " + " ".join(test_chunk),
            truncation=False,
            return_tensors=None
        )

        token_count = len(
            encoded["input_ids"]
        )

        if (
            token_count > MAX_INPUT_TOKENS
            and current_chunk
        ):

            chunks.append(
                " ".join(current_chunk)
            )

            current_chunk = [word]

        else:

            current_chunk = test_chunk

        current_tokens = token_count

    if current_chunk:

        chunks.append(
            " ".join(current_chunk)
        )

    return chunks


print()
print("Splitting meeting into chunks...")

chunks = create_chunks(meeting_text)

print("Number of chunks:", len(chunks))


# ---------------------------------------------------------
# Summarization function
# ---------------------------------------------------------

def summarize_text(text):

    prompt = "summarize: " + text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_OUTPUT_TOKENS,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    summary = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return summary.strip()


# ---------------------------------------------------------
# Summarize each chunk
# ---------------------------------------------------------

chunk_summaries = []

print()
print("Generating chunk summaries...")

for index, chunk in enumerate(chunks, start=1):

    print(
        f"Summarizing chunk "
        f"{index}/{len(chunks)}..."
    )

    summary = summarize_text(chunk)

    chunk_summaries.append(summary)


# ---------------------------------------------------------
# Generate final summary
# ---------------------------------------------------------

print()
print("Generating final meeting summary...")

combined_summary = " ".join(
    chunk_summaries
)

final_summary = summarize_text(
    combined_summary
)


# ---------------------------------------------------------
# Save output
# ---------------------------------------------------------

output = {
    "model": MODEL_NAME,
    "source_file": INPUT_FILE,
    "transcript_segments": len(transcript),
    "number_of_chunks": len(chunks),
    "chunk_summaries": chunk_summaries,
    "final_summary": final_summary
}


os.makedirs(
    "outputs/summaries",
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False
    )


# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------

print()
print("=" * 60)
print("FINAL MEETING SUMMARY")
print("=" * 60)

print()
print(final_summary)

print()
print("=" * 60)
print("SUMMARIZATION COMPLETE")
print("=" * 60)

print()
print("Output saved to:")
print(OUTPUT_FILE)