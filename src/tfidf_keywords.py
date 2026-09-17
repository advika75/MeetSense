import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer


INPUT_FILE = "outputs/nlp/nlp_processed_transcript.json"
OUTPUT_FILE = "outputs/nlp/tfidf_keywords.json"


print("=" * 60)
print("TF-IDF KEYWORD EXTRACTION")
print("=" * 60)

print()
print("Loading NLP-processed transcript...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

segments = data["processed_segments"]

print("Segments loaded:", len(segments))


# ---------------------------------------------------------
# Build documents using lemmatized tokens
# ---------------------------------------------------------

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


print("Documents for TF-IDF:", len(documents))


# ---------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------

print()
print("Running TF-IDF...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

tfidf_matrix = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

print("Vocabulary size:", len(feature_names))


# ---------------------------------------------------------
# Global keyword scores
# ---------------------------------------------------------

scores = tfidf_matrix.sum(axis=0).A1

keyword_scores = []

for word, score in zip(feature_names, scores):

    keyword_scores.append(
        {
            "term": word,
            "score": float(score)
        }
    )


keyword_scores.sort(
    key=lambda x: x["score"],
    reverse=True
)


top_keywords = keyword_scores[:30]


# ---------------------------------------------------------
# Speaker-wise keywords
# ---------------------------------------------------------

speaker_documents = {}

for document, meta in zip(documents, metadata):

    speaker = meta["speaker"]

    if speaker not in speaker_documents:
        speaker_documents[speaker] = []

    speaker_documents[speaker].append(document)


speaker_keywords = {}

for speaker, docs in speaker_documents.items():

    if not docs:
        continue

    speaker_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1
    )

    matrix = speaker_vectorizer.fit_transform(docs)

    features = speaker_vectorizer.get_feature_names_out()

    speaker_scores = matrix.sum(axis=0).A1

    terms = []

    for term, score in zip(features, speaker_scores):

        terms.append(
            {
                "term": term,
                "score": float(score)
            }
        )

    terms.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    speaker_keywords[speaker] = terms[:15]


# ---------------------------------------------------------
# Save output
# ---------------------------------------------------------

output = {
    "source_file": INPUT_FILE,
    "document_count": len(documents),
    "vocabulary_size": len(feature_names),
    "top_keywords": top_keywords,
    "speaker_keywords": speaker_keywords
}


os.makedirs("outputs/nlp", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False
    )


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print()
print("=" * 60)
print("TOP TF-IDF KEYWORDS")
print("=" * 60)

for i, item in enumerate(top_keywords, start=1):

    print(
        f"{i:2}. "
        f"{item['term']:<30} "
        f"{item['score']:.4f}"
    )


print()
print("Output saved to:")
print(OUTPUT_FILE)