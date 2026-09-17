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
# NLTK DATA
# ============================================================

print("=" * 60)
print("NLP PREPROCESSING")
print("=" * 60)

print()
print("Checking NLTK resources...")


nltk_resources = [
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


for resource_path, resource_name in nltk_resources:

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


print("NLTK resources ready.")


# ============================================================
# LOAD SPEAKER TRANSCRIPT
# ============================================================

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


# ============================================================
# NLP OBJECTS
# ============================================================

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


# ============================================================
# PROCESS SEGMENTS
# ============================================================

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


    # --------------------------------------------------------
    # Sentence segmentation
    # --------------------------------------------------------

    sentences = sent_tokenize(text)


    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    tokens = word_tokenize(text)


    # --------------------------------------------------------
    # Stopword removal
    # --------------------------------------------------------

    filtered_tokens = [
        token.lower()
        for token in tokens
        if token.isalpha()
        and token.lower() not in stop_words
    ]


    # --------------------------------------------------------
    # POS tagging
    # --------------------------------------------------------

    tagged_tokens = pos_tag(
        filtered_tokens
    )


    # --------------------------------------------------------
    # Lemmatization
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Save processed segment
    # --------------------------------------------------------

    processed_segments.append(
        {
            "start": segment.get("start"),
            "end": segment.get("end"),
            "speaker": speaker,
            "original_text": text,
            "sentences": sentences,
            "tokens": tokens,
            "filtered_tokens": filtered_tokens,
            "pos_tags": tagged_tokens,
            "lemmatized_tokens": lemmatized_tokens
        }
    )


    all_text.append(text)


    # Progress
    if index % 50 == 0:

        print(
            f"Processed {index} segments..."
        )


# ============================================================
# GLOBAL NER
# ============================================================

print()
print("Running Named Entity Recognition...")

full_text = " ".join(all_text)

global_tokens = word_tokenize(
    full_text
)

global_tagged = pos_tag(
    global_tokens
)

tree = ne_chunk(
    global_tagged
)


entities = []

for subtree in tree:

    if hasattr(
        subtree,
        "label"
    ):

        entity = " ".join(
            word
            for word, tag
            in subtree.leaves()
        )

        entities.append(
            {
                "entity": entity,
                "type": subtree.label()
            }
        )


# Remove duplicate entities
unique_entities = []

seen = set()

for entity in entities:

    key = (
        entity["entity"],
        entity["type"]
    )

    if key not in seen:

        seen.add(key)
        unique_entities.append(entity)


# ============================================================
# BUILD OUTPUT
# ============================================================

output = {
    "source_file": INPUT_FILE,
    "segment_count": len(processed_segments),
    "processed_segments": processed_segments,
    "named_entities": unique_entities
}


# ============================================================
# SAVE
# ============================================================

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


# ============================================================
# RESULTS
# ============================================================

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