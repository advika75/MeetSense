"""
TF-IDF extractive summarization baseline.

Ranks the sentences already produced by nlp_preprocessing.py
(outputs/nlp/nlp_processed_transcript.json) by summed TF-IDF weight,
greedily selects the highest-scoring sentences up to a target word
budget, then re-orders the selection chronologically.

Used as a non-neural baseline against the T5-small abstractive
summary in model_comparison.py. Reuses scikit-learn (already a
project dependency) instead of adding a new summarization library.
"""

import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


NLP_PROCESSED_FILE = "outputs/nlp/nlp_processed_transcript.json"
SUMMARY_FILE = "outputs/summaries/meeting_summary.json"
OUTPUT_FILE = "outputs/summaries/extractive_summary_baseline.json"

DEFAULT_TARGET_WORD_COUNT = 60
MIN_SENTENCES = 3
MIN_SENTENCE_WORDS = 3


def load_sentences(nlp_processed_path=NLP_PROCESSED_FILE):
    """Flatten all sentences from the NLP-processed transcript,
    in original chronological order, with speaker/time attribution."""

    with open(nlp_processed_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = []

    for segment in data.get("processed_segments", []):

        speaker = segment.get("speaker", "UNKNOWN")
        start = segment.get("start")
        end = segment.get("end")

        for sentence in segment.get("sentences", []):

            text = sentence.strip()

            if len(text.split()) < MIN_SENTENCE_WORDS:
                continue

            sentences.append({
                "text": text,
                "speaker": speaker,
                "start": start,
                "end": end,
            })

    return sentences


def generate_extractive_summary(
    nlp_processed_path=NLP_PROCESSED_FILE,
    target_word_count=DEFAULT_TARGET_WORD_COUNT,
    min_sentences=MIN_SENTENCES,
):
    """Return a dict describing a TF-IDF-ranked extractive summary."""

    sentences = load_sentences(nlp_processed_path)

    if len(sentences) < min_sentences:
        raise ValueError(
            f"Not enough sentences ({len(sentences)}) to build an "
            f"extractive summary from {nlp_processed_path}."
        )

    texts = [s["text"] for s in sentences]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Sentence importance = sum of TF-IDF weight of its terms.
    scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()

    ranked_indices = list(np.argsort(scores)[::-1])

    selected_indices = []
    word_total = 0

    for idx in ranked_indices:

        if (
            word_total >= target_word_count
            and len(selected_indices) >= min_sentences
        ):
            break

        selected_indices.append(int(idx))
        word_total += len(texts[idx].split())

    # Re-order chronologically for readability.
    selected_indices.sort()

    selected_sentences = [sentences[i] for i in selected_indices]
    summary_text = " ".join(s["text"] for s in selected_sentences)

    return {
        "method": "TF-IDF extractive baseline (sentence-level, "
                  "chronologically re-ordered after ranking)",
        "source_file": nlp_processed_path,
        "total_sentences_considered": len(sentences),
        "sentences_selected": len(selected_indices),
        "target_word_count": target_word_count,
        "actual_word_count": word_total,
        "selected_sentences": selected_sentences,
        "summary": summary_text,
    }


if __name__ == "__main__":

    print("=" * 60)
    print("TF-IDF EXTRACTIVE SUMMARIZATION BASELINE")
    print("=" * 60)

    target_word_count = DEFAULT_TARGET_WORD_COUNT
    target_source = "default"

    if os.path.exists(SUMMARY_FILE):

        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            t5_summary = json.load(f)

        t5_text = t5_summary.get("final_summary", "").strip()

        if t5_text:
            target_word_count = len(t5_text.split())
            target_source = f"matched to T5 summary length ({SUMMARY_FILE})"

    print()
    print(f"Target word count: {target_word_count} ({target_source})")

    result = generate_extractive_summary(
        target_word_count=target_word_count
    )

    os.makedirs("outputs/summaries", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print()
    print("Sentences considered:", result["total_sentences_considered"])
    print("Sentences selected:", result["sentences_selected"])
    print("Actual word count:", result["actual_word_count"])

    print()
    print("Extractive summary:")
    print(result["summary"])

    print()
    print("Results saved to:")
    print(OUTPUT_FILE)
