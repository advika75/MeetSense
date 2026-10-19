import json
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


INPUT_FILE = "outputs/nlp/nlp_processed_transcript.json"
OUTPUT_FILE = "outputs/nlp/topic_modeling.json"

NUMBER_OF_TOPICS = 5
TOP_WORDS_PER_TOPIC = 10


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
# Compute LDA topics + per-segment topic assignments
# ---------------------------------------------------------
#
# LDA is refit from scratch over the full document set every call
# (scikit-learn's partial_fit exists but would require the vocabulary
# to be fixed up front, which it isn't here). The live pipeline calls
# this periodically over the growing corpus rather than updating
# topics incrementally per chunk -- this is a periodic batch refit,
# not streaming topic modeling.

def compute_topics(
    documents,
    metadata,
    n_topics=NUMBER_OF_TOPICS,
    top_words_per_topic=TOP_WORDS_PER_TOPIC,
    min_df=2
):

    empty_result = {
        "document_count": len(documents),
        "vocabulary_size": 0,
        "number_of_topics": 0,
        "topics": [],
        "topic_counts": {},
        "speaker_topic_distribution": {},
        "segment_topic_assignments": []
    }

    if not documents:
        return empty_result

    effective_min_df = min(min_df, len(documents))

    vectorizer = CountVectorizer(
        lowercase=True,
        min_df=effective_min_df,
        max_df=0.95
    )

    try:
        document_term_matrix = vectorizer.fit_transform(documents)

    except ValueError:
        return empty_result

    feature_names = vectorizer.get_feature_names_out()

    if len(feature_names) == 0:
        return empty_result

    # LDA needs at least as many documents/features as topics.
    effective_n_topics = max(
        1,
        min(n_topics, len(documents), len(feature_names))
    )

    lda = LatentDirichletAllocation(
        n_components=effective_n_topics,
        random_state=42,
        learning_method="batch",
        max_iter=20
    )

    lda.fit(document_term_matrix)

    topics = []

    for topic_index, topic in enumerate(lda.components_):

        top_indices = topic.argsort()[
            -top_words_per_topic:
        ][::-1]

        words = [
            {
                "word": feature_names[index],
                "weight": float(topic[index])
            }
            for index in top_indices
        ]

        topics.append(
            {
                "topic_id": topic_index + 1,
                "top_words": words
            }
        )

    topic_distribution = lda.transform(document_term_matrix)

    topic_assignments = []

    for index, distribution in enumerate(topic_distribution):

        dominant_topic = int(distribution.argmax())

        topic_assignments.append(
            {
                "start": metadata[index]["start"],
                "end": metadata[index]["end"],
                "speaker": metadata[index]["speaker"],
                "text": metadata[index]["original_text"],
                "dominant_topic": dominant_topic + 1,
                "topic_probabilities": [
                    float(value) for value in distribution
                ]
            }
        )

    topic_counts = {}

    for item in topic_assignments:

        topic_counts[item["dominant_topic"]] = (
            topic_counts.get(item["dominant_topic"], 0) + 1
        )

    speaker_topics = {}

    for item in topic_assignments:

        speaker = item["speaker"]
        topic = item["dominant_topic"]

        speaker_topics.setdefault(speaker, {})

        speaker_topics[speaker][topic] = (
            speaker_topics[speaker].get(topic, 0) + 1
        )

    return {
        "document_count": len(documents),
        "vocabulary_size": len(feature_names),
        "number_of_topics": effective_n_topics,
        "topics": topics,
        "topic_counts": topic_counts,
        "speaker_topic_distribution": speaker_topics,
        "segment_topic_assignments": topic_assignments
    }


# ---------------------------------------------------------
# Batch CLI entry point (unchanged behavior)
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("TOPIC MODELING")
    print("=" * 60)

    print()
    print("Loading NLP-processed transcript...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data["processed_segments"]

    print("Segments loaded:", len(segments))

    documents, metadata = build_documents(segments)

    print("Documents for topic modeling:", len(documents))

    print()
    print("Creating document-term matrix...")

    print()
    print("Training LDA topic model...")
    print("Number of topics:", NUMBER_OF_TOPICS)

    result = compute_topics(documents, metadata)

    print("LDA training complete.")
    print("Vocabulary size:", result["vocabulary_size"])

    output = {
        "source_file": INPUT_FILE,
        "document_count": result["document_count"],
        "vocabulary_size": result["vocabulary_size"],
        "number_of_topics": result["number_of_topics"],
        "topics": result["topics"],
        "topic_counts": result["topic_counts"],
        "speaker_topic_distribution": result["speaker_topic_distribution"],
        "segment_topic_assignments": result["segment_topic_assignments"]
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
    print("DISCOVERED TOPICS")
    print("=" * 60)

    for topic in result["topics"]:

        words = [item["word"] for item in topic["top_words"]]

        print()
        print(
            f"Topic {topic['topic_id']}: "
            + ", ".join(words)
        )

    print()
    print("=" * 60)
    print("TOPIC DISTRIBUTION")
    print("=" * 60)

    for topic_id in sorted(result["topic_counts"]):

        print(
            f"Topic {topic_id}: "
            f"{result['topic_counts'][topic_id]} segments"
        )

    print()
    print("Output saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
