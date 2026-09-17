import json
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


INPUT_FILE = "outputs/nlp/nlp_processed_transcript.json"
OUTPUT_FILE = "outputs/nlp/topic_modeling.json"


print("=" * 60)
print("TOPIC MODELING")
print("=" * 60)

print()
print("Loading NLP-processed transcript...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

segments = data["processed_segments"]

print("Segments loaded:", len(segments))


# ---------------------------------------------------------
# Build documents
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


print("Documents for topic modeling:", len(documents))


# ---------------------------------------------------------
# Convert text into word-count matrix
# ---------------------------------------------------------

print()
print("Creating document-term matrix...")

vectorizer = CountVectorizer(
    lowercase=True,
    min_df=2,
    max_df=0.95
)

document_term_matrix = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

print("Vocabulary size:", len(feature_names))


# ---------------------------------------------------------
# LDA Topic Model
# ---------------------------------------------------------

NUMBER_OF_TOPICS = 5
TOP_WORDS_PER_TOPIC = 10

print()
print("Training LDA topic model...")
print("Number of topics:", NUMBER_OF_TOPICS)

lda = LatentDirichletAllocation(
    n_components=NUMBER_OF_TOPICS,
    random_state=42,
    learning_method="batch",
    max_iter=20
)

lda.fit(document_term_matrix)

print("LDA training complete.")


# ---------------------------------------------------------
# Extract topic words
# ---------------------------------------------------------

topics = []

for topic_index, topic in enumerate(lda.components_):

    top_indices = topic.argsort()[
        -TOP_WORDS_PER_TOPIC:
    ][::-1]

    words = []

    for index in top_indices:

        words.append(
            {
                "word": feature_names[index],
                "weight": float(topic[index])
            }
        )

    topics.append(
        {
            "topic_id": topic_index + 1,
            "top_words": words
        }
    )


# ---------------------------------------------------------
# Assign dominant topic to each segment
# ---------------------------------------------------------

print()
print("Assigning topics to transcript segments...")

topic_distribution = lda.transform(document_term_matrix)

topic_assignments = []

for index, distribution in enumerate(topic_distribution):

    dominant_topic = int(
        distribution.argmax()
    )

    topic_assignments.append(
        {
            "start": metadata[index]["start"],
            "end": metadata[index]["end"],
            "speaker": metadata[index]["speaker"],
            "text": metadata[index]["original_text"],
            "dominant_topic": dominant_topic + 1,
            "topic_probabilities": [
                float(value)
                for value in distribution
            ]
        }
    )


# ---------------------------------------------------------
# Topic distribution statistics
# ---------------------------------------------------------

topic_counts = {}

for item in topic_assignments:

    topic = item["dominant_topic"]

    if topic not in topic_counts:
        topic_counts[topic] = 0

    topic_counts[topic] += 1


# ---------------------------------------------------------
# Speaker-topic analysis
# ---------------------------------------------------------

speaker_topics = {}

for item in topic_assignments:

    speaker = item["speaker"]
    topic = item["dominant_topic"]

    if speaker not in speaker_topics:
        speaker_topics[speaker] = {}

    if topic not in speaker_topics[speaker]:
        speaker_topics[speaker][topic] = 0

    speaker_topics[speaker][topic] += 1


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

output = {
    "source_file": INPUT_FILE,
    "document_count": len(documents),
    "vocabulary_size": len(feature_names),
    "number_of_topics": NUMBER_OF_TOPICS,
    "topics": topics,
    "topic_counts": topic_counts,
    "speaker_topic_distribution": speaker_topics,
    "segment_topic_assignments": topic_assignments
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
# Display topics
# ---------------------------------------------------------

print()
print("=" * 60)
print("DISCOVERED TOPICS")
print("=" * 60)

for topic in topics:

    words = [
        item["word"]
        for item in topic["top_words"]
    ]

    print()
    print(
        f"Topic {topic['topic_id']}: "
        + ", ".join(words)
    )


print()
print("=" * 60)
print("TOPIC DISTRIBUTION")
print("=" * 60)

for topic_id in sorted(topic_counts):

    print(
        f"Topic {topic_id}: "
        f"{topic_counts[topic_id]} segments"
    )


print()
print("Output saved to:")
print(OUTPUT_FILE)