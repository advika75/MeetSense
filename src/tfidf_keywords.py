import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer


INPUT_FILE = "outputs/nlp/nlp_processed_transcript.json"
OUTPUT_FILE = "outputs/nlp/tfidf_keywords.json"


# ---------------------------------------------------------
# Build (document, metadata) pairs from processed segments
# ---------------------------------------------------------

def build_documents(segments):

    documents = []
    metadata = []

    for segment in segments:

        tokens = segment.get("lemmatized_tokens", [])

        text = " ".join(tokens)

        if text.strip():

            documents.append(text)

            metadata.append(
                {
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "speaker": segment.get("speaker"),
                    "original_text": segment.get("original_text")
                }
            )

    return documents, metadata


# ---------------------------------------------------------
# Compute global + per-speaker TF-IDF keywords
# ---------------------------------------------------------
#
# This refits TfidfVectorizer over the full document set every time
# it's called. There is no incremental/streaming variant used here:
# IDF weights are only meaningful relative to the current corpus, so
# the live pipeline calls this periodically on the growing corpus
# (a full refit each time) rather than updating scores incrementally.

def compute_tfidf_keywords(
    documents,
    metadata,
    min_df=2,
    top_k=30,
    speaker_top_k=15
):

    if not documents:

        return {
            "document_count": 0,
            "vocabulary_size": 0,
            "top_keywords": [],
            "speaker_keywords": {}
        }

    # min_df can't exceed the number of documents, or sklearn raises.
    effective_min_df = min(min_df, len(documents))

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=effective_min_df,
        max_df=0.95
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)

    except ValueError:

        # Empty vocabulary after min_df/max_df filtering -- happens
        # with very few/short documents early in a live session.
        return {
            "document_count": len(documents),
            "vocabulary_size": 0,
            "top_keywords": [],
            "speaker_keywords": {}
        }

    feature_names = vectorizer.get_feature_names_out()

    scores = tfidf_matrix.sum(axis=0).A1

    keyword_scores = [
        {"term": word, "score": float(score)}
        for word, score in zip(feature_names, scores)
    ]

    keyword_scores.sort(key=lambda x: x["score"], reverse=True)

    top_keywords = keyword_scores[:top_k]

    speaker_documents = {}

    for document, meta in zip(documents, metadata):

        speaker = meta["speaker"]

        speaker_documents.setdefault(speaker, []).append(document)

    speaker_keywords = {}

    for speaker, docs in speaker_documents.items():

        if not docs:
            continue

        try:

            speaker_vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1
            )

            matrix = speaker_vectorizer.fit_transform(docs)

            features = speaker_vectorizer.get_feature_names_out()

        except ValueError:

            speaker_keywords[speaker] = []
            continue

        speaker_scores = matrix.sum(axis=0).A1

        terms = [
            {"term": term, "score": float(score)}
            for term, score in zip(features, speaker_scores)
        ]

        terms.sort(key=lambda x: x["score"], reverse=True)

        speaker_keywords[speaker] = terms[:speaker_top_k]

    return {
        "document_count": len(documents),
        "vocabulary_size": len(feature_names),
        "top_keywords": top_keywords,
        "speaker_keywords": speaker_keywords
    }


# ---------------------------------------------------------
# Batch CLI entry point (unchanged behavior)
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("TF-IDF KEYWORD EXTRACTION")
    print("=" * 60)

    print()
    print("Loading NLP-processed transcript...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data["processed_segments"]

    print("Segments loaded:", len(segments))

    documents, metadata = build_documents(segments)

    print("Documents for TF-IDF:", len(documents))

    print()
    print("Running TF-IDF...")

    result = compute_tfidf_keywords(documents, metadata)

    print("Vocabulary size:", result["vocabulary_size"])

    output = {
        "source_file": INPUT_FILE,
        "document_count": result["document_count"],
        "vocabulary_size": result["vocabulary_size"],
        "top_keywords": result["top_keywords"],
        "speaker_keywords": result["speaker_keywords"]
    }

    os.makedirs("outputs/nlp", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("TOP TF-IDF KEYWORDS")
    print("=" * 60)

    for i, item in enumerate(result["top_keywords"], start=1):

        print(
            f"{i:2}. "
            f"{item['term']:<30} "
            f"{item['score']:.4f}"
        )

    print()
    print("Output saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
