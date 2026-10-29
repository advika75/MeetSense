import json
import os
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = (
    "outputs/transcripts/speaker_transcript.json"
)

OUTPUT_FILE = (
    "outputs/nlp/nlp_processed_transcript.json"
)


# ============================================================
# NLTK RESOURCES
# ============================================================

NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("taggers/averaged_perceptron_tagger_eng",
     "averaged_perceptron_tagger_eng"),
    ("chunkers/maxent_ne_chunker_tab",
     "maxent_ne_chunker_tab"),
    ("corpora/words", "words"),
]

_resources_ready = False


def ensure_nltk_resources():
    """Download any missing NLTK resources. Cheap to call repeatedly
    (checked once per process) -- both the batch script and the live
    pipeline call this before any tokenization."""

    global _resources_ready

    if _resources_ready:
        return

    for resource_path, resource_name in NLTK_RESOURCES:

        try:
            nltk.data.find(resource_path)

        except LookupError:

            print(
                f"Downloading NLTK resource: {resource_name}"
            )

            nltk.download(
                resource_name,
                quiet=True
            )

    _resources_ready = True


# ============================================================
# LAZY NLP OBJECTS
# ============================================================

_stop_words = None
_lemmatizer = None


def _get_stop_words():

    global _stop_words

    if _stop_words is None:
        ensure_nltk_resources()
        _stop_words = set(stopwords.words("english"))

    return _stop_words


def _get_lemmatizer():

    global _lemmatizer

    if _lemmatizer is None:
        ensure_nltk_resources()
        _lemmatizer = WordNetLemmatizer()

    return _lemmatizer


# ============================================================
# PROCESS A SINGLE SEGMENT'S TEXT
# ============================================================

def process_segment_text(text):
    """Tokenize/tag/lemmatize one piece of text. Reused per-segment
    by both the batch script (over the whole transcript) and the live
    pipeline (over only newly-arrived segments each chunk)."""

    stop_words = _get_stop_words()
    lemmatizer = _get_lemmatizer()

    sentences = sent_tokenize(text)

    tokens = word_tokenize(text)

    filtered_tokens = [
        token.lower()
        for token in tokens
        if token.isalpha()
        and token.lower() not in stop_words
    ]

    tagged_tokens = pos_tag(
        filtered_tokens
    )

    lemmatized_tokens = []

    for token, tag in tagged_tokens:

        if tag.startswith("V"):
            pos = "v"

        elif tag.startswith("J"):
            pos = "a"

        elif tag.startswith("R"):
            pos = "r"

        elif tag.startswith("N"):
            pos = "n"

        else:
            pos = "n"

        lemma = lemmatizer.lemmatize(
            token,
            pos=pos
        )

        lemmatized_tokens.append(
            lemma
        )

    return {
        "sentences": sentences,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "pos_tags": tagged_tokens,
        "lemmatized_tokens": lemmatized_tokens
    }


# ============================================================
# NAMED ENTITY RECOGNITION
# ============================================================

def extract_entities(text):
    """Run NER over a piece of text and return unique
    {"entity", "type"} dicts. The batch script calls this once over
    the whole concatenated transcript; the live pipeline calls it
    per newly-arrived batch of segments only (a smaller context
    window per call, since it never re-scans already-processed text
    -- entities that only become recognizable with cross-segment
    context spanning a chunk boundary may be missed live even though
    the batch pass would catch them)."""

    if not text.strip():
        return []

    tokens = word_tokenize(text)

    tagged = pos_tag(tokens)

    tree = ne_chunk(tagged)

    entities = []

    seen = set()

    for subtree in tree:

        if hasattr(subtree, "label"):

            entity = " ".join(
                word
                for word, tag
                in subtree.leaves()
            )

            key = (entity, subtree.label())

            if key not in seen:

                seen.add(key)

                entities.append(
                    {
                        "entity": entity,
                        "type": subtree.label()
                    }
                )

    return entities


# ============================================================
# REUSABLE: RAW SEGMENTS -> PROCESSED SEGMENTS (NO FILE I/O)
# ============================================================

def build_processed_segments(transcript_segments):
    """Same per-segment transform as main()'s loop below, but takes an
    in-memory segment list and returns the processed list directly --
    no fixed INPUT_FILE/OUTPUT_FILE involved. Lets any caller (e.g. a
    per-report Keywords & Topics view) feed build_documents() in
    tfidf_keywords.py / topic_modeling.py without a batch pipeline run."""

    ensure_nltk_resources()

    processed_segments = []

    for segment in transcript_segments:

        speaker = segment.get("speaker", "UNKNOWN")
        text = (segment.get("text") or "").strip()

        if not text:
            continue

        result = process_segment_text(text)

        processed_segments.append(
            {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "speaker": speaker,
                "original_text": text,
                "sentences": result["sentences"],
                "tokens": result["tokens"],
                "filtered_tokens": result["filtered_tokens"],
                "pos_tags": result["pos_tags"],
                "lemmatized_tokens": result["lemmatized_tokens"],
            }
        )

    return processed_segments


# ============================================================
# BATCH CLI ENTRY POINT (unchanged behavior)
# ============================================================

def main():

    print("=" * 60)
    print("NLP PREPROCESSING")
    print("=" * 60)

    print()
    print("Checking NLTK resources...")

    ensure_nltk_resources()

    print("NLTK resources ready.")

    print()
    print("Loading speaker-aware transcript...")

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        transcript = json.load(f)

    if not isinstance(transcript, list):

        raise ValueError(
            "speaker_transcript.json must contain a list."
        )

    print(
        "Transcript segments:",
        len(transcript)
    )

    processed_segments = []

    all_text = []

    for index, segment in enumerate(
        transcript,
        start=1
    ):

        speaker = segment.get(
            "speaker",
            "UNKNOWN"
        )

        text = segment.get(
            "text",
            ""
        ).strip()

        if not text:
            continue

        result = process_segment_text(text)

        processed_segments.append(
            {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "speaker": speaker,
                "original_text": text,
                "sentences": result["sentences"],
                "tokens": result["tokens"],
                "filtered_tokens": result["filtered_tokens"],
                "pos_tags": result["pos_tags"],
                "lemmatized_tokens": result["lemmatized_tokens"]
            }
        )

        all_text.append(text)

        if index % 50 == 0:

            print(
                f"Processed {index} segments..."
            )

    print()
    print("Running Named Entity Recognition...")

    full_text = " ".join(all_text)

    unique_entities = extract_entities(full_text)

    output = {
        "source_file": INPUT_FILE,
        "segment_count": len(processed_segments),
        "processed_segments": processed_segments,
        "named_entities": unique_entities
    }

    os.makedirs(
        "outputs/nlp",
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

    print()
    print("=" * 60)
    print("NLP PREPROCESSING COMPLETE")
    print("=" * 60)

    print()
    print(
        "Processed segments:",
        len(processed_segments)
    )

    print(
        "Named entities:",
        len(unique_entities)
    )

    print()
    print("Top named entities:")

    for entity in unique_entities[:15]:

        print(
            f"  {entity['entity']} "
            f"({entity['type']})"
        )

    print()
    print("Output saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
