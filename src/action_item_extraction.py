import os
import json
import re


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "outputs/transcripts/speaker_transcript.json"

OUTPUT_DIR = "outputs/nlp"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "action_items_decisions.json"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# ACTION KEYWORDS
# ============================================================

ACTION_PATTERNS = [

    r"\bneed to\b",
    r"\bneeds to\b",
    r"\bshould\b",
    r"\bmust\b",
    r"\bhave to\b",
    r"\bhas to\b",
    r"\bwill\b",
    r"\bwe need\b",
    r"\bi need\b",
    r"\bi will\b",
    r"\bi'll\b",
    r"\bwe will\b",
    r"\bwe'll\b",
    r"\bplease\b",
    r"\bmake sure\b",
    r"\bfollow up\b",
    r"\bprepare\b",
    r"\bcomplete\b",
    r"\bfinish\b",
    r"\breview\b",
    r"\bsend\b",
    r"\bupdate\b",
    r"\bcreate\b",
    r"\bschedule\b",
    r"\bsubmit\b",
    r"\bcheck\b",
    r"\bdiscuss\b"
]


# ============================================================
# DECISION KEYWORDS
# ============================================================

DECISION_PATTERNS = [

    r"\bwe decided\b",
    r"\bdecided to\b",
    r"\bwe agreed\b",
    r"\bagreed to\b",
    r"\bwe will use\b",
    r"\bwe are going with\b",
    r"\bwe chose\b",
    r"\bchosen\b",
    r"\bapproved\b",
    r"\bconfirmed\b",
    r"\bfinal decision\b",
    r"\bdecision is\b",
    r"\bthe plan is\b",
    r"\bwe'll use\b"
]


# ============================================================
# DEADLINE PATTERNS
# ============================================================

DEADLINE_PATTERNS = [

    r"\bby tomorrow\b",
    r"\bby today\b",
    r"\bby friday\b",
    r"\bby monday\b",
    r"\bby tuesday\b",
    r"\bby wednesday\b",
    r"\bby thursday\b",
    r"\bby saturday\b",
    r"\bby sunday\b",

    r"\bbefore tomorrow\b",
    r"\bbefore friday\b",
    r"\bbefore monday\b",

    r"\btomorrow\b",
    r"\bnext week\b",
    r"\bthis week\b",
    r"\bend of the week\b",
    r"\bnext month\b",

    r"\bby \d{1,2}(?:st|nd|rd|th)?\b"
]


# ============================================================
# EXTRACT SEGMENTS
# ============================================================

def extract_segments(data):

    if isinstance(data, list):

        return data

    if isinstance(data, dict):

        possible_keys = [
            "segments",
            "transcript",
            "speaker_segments",
            "results"
        ]

        for key in possible_keys:

            if key in data:

                value = data[key]

                if isinstance(value, list):

                    return value

    return []


# ============================================================
# GET TEXT
# ============================================================

def get_text(segment):

    if not isinstance(segment, dict):

        return ""

    possible_keys = [
        "text",
        "transcript",
        "sentence",
        "content"
    ]

    for key in possible_keys:

        value = segment.get(key)

        if isinstance(value, str):

            return value.strip()

    return ""


# ============================================================
# GET SPEAKER
# ============================================================

def get_speaker(segment):

    if not isinstance(segment, dict):

        return "Unknown"

    possible_keys = [
        "speaker",
        "speaker_id",
        "speaker_label"
    ]

    for key in possible_keys:

        value = segment.get(key)

        if value is not None:

            return str(value)

    return "Unknown"


# ============================================================
# GET TIMESTAMP
# ============================================================

def get_start_time(segment):

    if not isinstance(segment, dict):

        return None

    possible_keys = [
        "start",
        "start_time",
        "begin"
    ]

    for key in possible_keys:

        value = segment.get(key)

        if value is not None:

            return value

    return None


# ============================================================
# FIND PATTERN
# ============================================================

def contains_pattern(
    text,
    patterns
):

    text_lower = text.lower()

    for pattern in patterns:

        if re.search(
            pattern,
            text_lower
        ):

            return True

    return False


# ============================================================
# EXTRACT DEADLINE
# ============================================================

def extract_deadline(text):

    text_lower = text.lower()

    for pattern in DEADLINE_PATTERNS:

        match = re.search(
            pattern,
            text_lower
        )

        if match:

            return match.group(0)

    return None


# ============================================================
# CLEAN ACTION TEXT
# ============================================================

def clean_action_text(text):

    text = text.strip()

    # Remove excessive spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ACTION ITEM AND DECISION EXTRACTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        print("\nERROR:")
        print("Speaker transcript not found:")

        print(INPUT_FILE)

        return

    # --------------------------------------------------------
    # Load transcript
    # --------------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    segments = extract_segments(data)

    print(
        f"\nTranscript segments: {len(segments)}"
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    action_items = []

    decisions = []

    # --------------------------------------------------------
    # Process segments
    # --------------------------------------------------------

    for index, segment in enumerate(
        segments
    ):

        text = get_text(segment)

        if not text:

            continue

        speaker = get_speaker(segment)

        start_time = get_start_time(segment)

        deadline = extract_deadline(text)

        # ----------------------------------------------------
        # ACTION ITEM
        # ----------------------------------------------------

        if contains_pattern(
            text,
            ACTION_PATTERNS
        ):

            action_items.append({

                "id":
                    len(action_items) + 1,

                "segment_id":
                    index + 1,

                "speaker":
                    speaker,

                "start_time":
                    start_time,

                "action":
                    clean_action_text(text),

                "deadline":
                    deadline,

                "extraction_method":
                    "keyword_pattern"
            })

        # ----------------------------------------------------
        # DECISION
        # ----------------------------------------------------

        if contains_pattern(
            text,
            DECISION_PATTERNS
        ):

            decisions.append({

                "id":
                    len(decisions) + 1,

                "segment_id":
                    index + 1,

                "speaker":
                    speaker,

                "start_time":
                    start_time,

                "decision":
                    clean_action_text(text),

                "extraction_method":
                    "keyword_pattern"
            })

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output = {

        "method":
            "Rule-based Information Extraction",

        "total_segments":
            len(segments),

        "number_of_action_items":
            len(action_items),

        "number_of_decisions":
            len(decisions),

        "action_items":
            action_items,

        "decisions":
            decisions
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # DISPLAY ACTION ITEMS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ACTION ITEMS")
    print("=" * 70)

    if action_items:

        for item in action_items[:20]:

            print(
                f"\n[{item['id']}] "
                f"{item['speaker']}"
            )

            print(
                f"Action: {item['action']}"
            )

            if item["deadline"]:

                print(
                    f"Deadline: "
                    f"{item['deadline']}"
                )

    else:

        print(
            "No action items detected."
        )

    # --------------------------------------------------------
    # DISPLAY DECISIONS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DECISIONS")
    print("=" * 70)

    if decisions:

        for item in decisions[:20]:

            print(
                f"\n[{item['id']}] "
                f"{item['speaker']}"
            )

            print(
                f"Decision: "
                f"{item['decision']}"
            )

    else:

        print(
            "No decisions detected."
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "INFORMATION EXTRACTION COMPLETE"
    )

    print("\nSaved to:")

    print(
        OUTPUT_FILE
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()