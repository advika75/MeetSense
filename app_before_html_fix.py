import json
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# ============================================================
# THEME
# ============================================================

if "theme" not in st.session_state:
    st.session_state.theme = "light"

THEMES = {
    "light": {
        "bg": "#F4F6FA",
        "sidebar": "#FFFFFF",
        "card": "#FFFFFF",
        "text": "#172033",
        "text_secondary": "#526078",
        "muted": "#7B879B",
        "border": "#E1E6EF",
        "accent": "#6254E8",
        "accent_hover": "#5145D5",
        "accent_soft": "#EEEAFE",
        "success": "#138A5B",
        "success_bg": "#E8F7F0",
        "warning": "#B7791F",
        "warning_bg": "#FFF7E6",
        "danger": "#C53030",
        "danger_bg": "#FFF0F0",
        "input": "#FFFFFF",
        "sidebar_hover": "#F4F1FF",
        "code_bg": "#F3F4F7",
    },
    "dark": {
        "bg": "#0B1220",
        "sidebar": "#111A2B",
        "card": "#151F32",
        "text": "#F4F7FB",
        "text_secondary": "#AAB5C7",
        "muted": "#718096",
        "border": "#27344A",
        "accent": "#7C6CF2",
        "accent_hover": "#9183FF",
        "accent_soft": "#272343",
        "success": "#4FD1A1",
        "success_bg": "#12352B",
        "warning": "#F6C453",
        "warning_bg": "#3A2E12",
        "danger": "#FC8181",
        "danger_bg": "#3A1919",
        "input": "#111A2B",
        "sidebar_hover": "#1D2740",
        "code_bg": "#0F1726",
    },
}

theme = THEMES[st.session_state.theme]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MeetSense",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

AUDIO_DIR = BASE_DIR / "data" / "audio"
OUTPUT_DIR = BASE_DIR / "outputs"

TRANSCRIPT_DIR = OUTPUT_DIR / "transcripts"
NLP_DIR = OUTPUT_DIR / "nlp"
SUMMARY_DIR = OUTPUT_DIR / "summaries"
EVAL_DIR = OUTPUT_DIR / "evaluations"


AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OUTPUT FILES
# ============================================================

SPEAKER_TRANSCRIPT = TRANSCRIPT_DIR / "speaker_transcript.json"
MEETING_TRANSCRIPT = TRANSCRIPT_DIR / "meeting.json"
DIARIZATION = TRANSCRIPT_DIR / "diarization.json"

NLP_PROCESSED = NLP_DIR / "nlp_processed_transcript.json"
TFIDF = NLP_DIR / "tfidf_keywords.json"
TOPICS = NLP_DIR / "topic_modeling.json"
ACTION_ITEMS = NLP_DIR / "action_items_decisions.json"

SUMMARY = SUMMARY_DIR / "meeting_summary.json"

WER = EVAL_DIR / "wer_results.json"
DER = EVAL_DIR / "der_results.json"
ROUGE = EVAL_DIR / "rouge_results.json"
ACTION_EVAL = EVAL_DIR / "action_item_results.json"


# ============================================================
# CUSTOM CSS
# ============================================================



st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f6f7fb;
}

/* Hide Streamlit default UI */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: visible !important;
    background: transparent !important;
}

/* Main container */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e8eaf0;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

/* Brand */
.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 5px 8px 25px 8px;
}

.brand-icon {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: linear-gradient(135deg, #5b4bff, #7c3aed);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 21px;
    box-shadow: 0 5px 15px rgba(91,75,255,0.25);
}

.brand-name {
    font-size: 20px;
    font-weight: 800;
    color: #171827;
}

/* Top bar */
.topbar {
    background: white;
    border: 1px solid #e7e8ef;
    border-radius: 16px;
    padding: 13px 18px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 10px rgba(20,20,50,0.03);
}

.top-title {
    font-size: 14px;
    font-weight: 600;
    color: #676b7c;
}

/* Page title */
.page-title {
    font-size: 31px;
    font-weight: 800;
    color: #171827;
    margin-bottom: 4px;
}

.page-subtitle {
    color: #747889;
    font-size: 14px;
    margin-bottom: 25px;
}

/* Cards */
.card {
    background: white;
    border: 1px solid #e7e8ef;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: 0 3px 15px rgba(25,25,60,0.035);
}

.card-title {
    font-size: 16px;
    font-weight: 700;
    color: #202131;
    margin-bottom: 12px;
}

.card-subtitle {
    font-size: 12px;
    color: #85899a;
}

/* Stat cards */
.stat-card {
    background: white;
    border: 1px solid #e7e8ef;
    border-radius: 16px;
    padding: 19px;
    min-height: 120px;
    box-shadow: 0 3px 15px rgba(25,25,60,0.035);
}

.stat-icon {
    font-size: 20px;
    margin-bottom: 10px;
}

.stat-number {
    font-size: 27px;
    font-weight: 800;
    color: #171827;
}

.stat-label {
    color: #777b8d;
    font-size: 12px;
    margin-top: 3px;
}

/* Speaker */
.speaker-row {
    background: #fafaff;
    border: 1px solid #ececf5;
    border-radius: 13px;
    padding: 13px 15px;
    margin-bottom: 10px;
}

.speaker-name {
    font-size: 14px;
    font-weight: 700;
    color: #222332;
}

.speaker-time {
    font-size: 12px;
    color: #7b7f91;
}

/* Transcript */
.transcript-item {
    background: #ffffff;
    border: 1px solid #e9eaf0;
    border-radius: 14px;
    padding: 15px 17px;
    margin-bottom: 10px;
    transition: 0.2s;
}

.transcript-item:hover {
    border-color: #cfcaff;
    box-shadow: 0 5px 16px rgba(70,50,180,0.06);
}

.transcript-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
}

.transcript-speaker {
    font-size: 13px;
    font-weight: 700;
    color: #5142d8;
}

.transcript-time {
    font-size: 11px;
    color: #8b8e9c;
}

.transcript-text {
    font-size: 14px;
    line-height: 1.65;
    color: #343746;
}

/* Tags */
.tag {
    display: inline-block;
    padding: 7px 11px;
    margin: 4px;
    background: #f1efff;
    color: #5949dc;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

/* Action item */
.action-item {
    padding: 14px;
    border: 1px solid #e7e8ef;
    border-radius: 12px;
    margin-bottom: 9px;
    background: #fff;
}

.action-title {
    font-weight: 600;
    font-size: 13px;
    color: #252735;
}

/* Decision */
.decision {
    background: #f7f5ff;
    border-left: 4px solid #6b5cff;
    padding: 14px 16px;
    border-radius: 9px;
    margin-bottom: 10px;
    font-size: 13px;
    line-height: 1.5;
}

/* Summary */
.summary-box {
    background: linear-gradient(135deg, #f7f5ff, #fbfaff);
    border: 1px solid #e4e0ff;
    border-radius: 14px;
    padding: 18px;
    font-size: 14px;
    line-height: 1.7;
    color: #343444;
}

/* Evaluation */
.metric {
    background: #fafaff;
    border: 1px solid #e8e8f2;
    border-radius: 13px;
    padding: 15px;
    text-align: center;
}

.metric-name {
    color: #777b8d;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    font-size: 25px;
    font-weight: 800;
    color: #5142d8;
    margin-top: 5px;
}

/* Meeting header */
.meeting-header {
    background: white;
    border: 1px solid #e7e8ef;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}

.meeting-name {
    font-size: 25px;
    font-weight: 800;
    color: #171827;
}

.meeting-meta {
    color: #777b8d;
    font-size: 12px;
    margin-top: 7px;
}

/* Status */
.status-pill {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 20px;
    background: #e9faf0;
    color: #18864b;
    font-size: 11px;
    font-weight: 700;
}

/* Section headings */
.section-heading {
    font-size: 18px;
    font-weight: 750;
    color: #202131;
    margin-top: 12px;
    margin-bottom: 12px;
}

/* Divider */
.soft-divider {
    height: 1px;
    background: #ececf1;
    margin: 18px 0;
}


/* MEETSENSE SIDEBAR VISIBILITY FIX */

/* Sidebar base */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    color: #202131 !important;
}

/* Force ALL sidebar text to a dark readable color */
section[data-testid="stSidebar"] * {
    color: #202131 !important;
}

/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] p {
    color: #202131 !important;
    opacity: 1 !important;
}

/* Navigation radio groups */
section[data-testid="stSidebar"] [role="radiogroup"] {
    color: #202131 !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #202131 !important;
    opacity: 1 !important;
    background: transparent !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] label p {
    color: #202131 !important;
    opacity: 1 !important;
    font-weight: 500 !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] label div {
    color: #202131 !important;
    opacity: 1 !important;
}

/* Streamlit markdown containers inside sidebar */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #202131 !important;
    opacity: 1 !important;
}

/* Radio option text */
section[data-testid="stSidebar"] [data-testid="stRadio"] label,
section[data-testid="stSidebar"] [data-testid="stRadio"] label *,
section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: #202131 !important;
    opacity: 1 !important;
}

/* Selected option */
section[data-testid="stSidebar"] [role="radio"][aria-checked="true"],
section[data-testid="stSidebar"] [role="radio"][aria-checked="true"] * {
    color: #5142d8 !important;
    opacity: 1 !important;
}

/* Sidebar caption */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: #777b8d !important;
    opacity: 1 !important;
}

/* Keep the MeetSense brand readable */
section[data-testid="stSidebar"] .brand,
section[data-testid="stSidebar"] .brand * {
    color: #171827 !important;
}

/* Keep the purple AI information card readable */
section[data-testid="stSidebar"] .brand-name {
    color: #171827 !important;
}



/* ============================================================
   STREAMLIT SIDEBAR TOGGLE FIX
   ============================================================ */

[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}

[data-testid="stSidebarCollapsedControl"] button {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: #202131 !important;
    background: #ffffff !important;
    border: 1px solid #e2e4ec !important;
    border-radius: 10px !important;
    box-shadow: 0 3px 12px rgba(20, 20, 50, 0.12) !important;
}

[data-testid="stSidebarCollapsedControl"] svg {
    color: #202131 !important;
    fill: #202131 !important;
}



/* ============================================================
   MEETSENSE THEME SYSTEM — LIGHT / DARK
   ============================================================ */

:root {
    --ms-bg: #f4f6fa;
    --ms-surface: #ffffff;
    --ms-surface-2: #f8f9fc;
    --ms-border: #e2e6ef;
    --ms-text: #171b2d;
    --ms-text-2: #555d70;
    --ms-muted: #7b8395;
    --ms-primary: #5b4bdb;
    --ms-primary-2: #7567e8;
    --ms-accent: #3b82f6;
    --ms-success: #159570;
    --ms-danger: #dc5a5a;
    --ms-shadow: 0 8px 28px rgba(25, 32, 55, 0.07);
}

/* Main application */
.stApp {
    background: var(--ms-bg) !important;
    color: var(--ms-text) !important;
}

/* Main content */
.main .block-container {
    color: var(--ms-text) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--ms-surface) !important;
    border-right: 1px solid var(--ms-border) !important;
}

section[data-testid="stSidebar"] * {
    color: var(--ms-text) !important;
}

section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: var(--ms-text) !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: var(--ms-text) !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: var(--ms-text) !important;
    opacity: 1 !important;
}

/* General typography */
h1, h2, h3, h4, h5, h6 {
    color: var(--ms-text) !important;
    opacity: 1 !important;
}

p, li, label {
    color: var(--ms-text-2) !important;
    opacity: 1 !important;
}

/* Cards */
.card,
.metric-card,
.meeting-card,
.panel,
.section-card,
.summary-card {
    background: var(--ms-surface) !important;
    border-color: var(--ms-border) !important;
    box-shadow: var(--ms-shadow) !important;
    color: var(--ms-text) !important;
}

/* Card text */
.card *,
.metric-card *,
.meeting-card *,
.panel *,
.section-card *,
.summary-card * {
    color: inherit;
}

/* Inputs */
input,
textarea,
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: var(--ms-surface) !important;
    color: var(--ms-text) !important;
    border-color: var(--ms-border) !important;
}

input::placeholder,
textarea::placeholder {
    color: var(--ms-muted) !important;
}

/* Buttons */
.stButton > button {
    background: var(--ms-surface) !important;
    color: var(--ms-text) !important;
    border: 1px solid var(--ms-border) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.stButton > button:hover {
    border-color: var(--ms-primary) !important;
    color: var(--ms-primary) !important;
}

/* Primary buttons */
.stButton > button[kind="primary"] {
    background: var(--ms-primary) !important;
    color: #ffffff !important;
    border-color: var(--ms-primary) !important;
}

.stButton > button[kind="primary"]:hover {
    background: var(--ms-primary-2) !important;
    color: #ffffff !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: var(--ms-text-2) !important;
    font-weight: 600 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--ms-primary) !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: var(--ms-surface) !important;
    border-color: var(--ms-border) !important;
}

[data-testid="stExpander"] * {
    color: var(--ms-text) !important;
}

/* Alerts / info boxes */
[data-testid="stAlert"] {
    background: var(--ms-surface-2) !important;
    color: var(--ms-text) !important;
}

/* Code */
pre,
code {
    background: #20232d !important;
    color: #e8ecf5 !important;
}

/* ============================================================
   DARK MODE
   ============================================================ */

@media (prefers-color-scheme: dark) {

    :root {
        --ms-bg: #11141b;
        --ms-surface: #191d26;
        --ms-surface-2: #202530;
        --ms-border: #303746;
        --ms-text: #f1f4fa;
        --ms-text-2: #c0c7d5;
        --ms-muted: #929bad;
        --ms-primary: #8b7cf6;
        --ms-primary-2: #a094ff;
        --ms-accent: #60a5fa;
        --ms-success: #45c89c;
        --ms-danger: #f47777;
        --ms-shadow: 0 10px 30px rgba(0,0,0,0.28);
    }

    .stApp {
        background: var(--ms-bg) !important;
        color: var(--ms-text) !important;
    }

    section[data-testid="stSidebar"] {
        background: #151820 !important;
        border-right: 1px solid #2c3240 !important;
    }

    section[data-testid="stSidebar"] * {
        color: var(--ms-text) !important;
        opacity: 1 !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #d9deea !important;
        opacity: 1 !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #f5f7fb !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #f1f4fa !important;
    }

    p, li, label {
        color: #c0c7d5 !important;
    }

    .card,
    .metric-card,
    .meeting-card,
    .panel,
    .section-card,
    .summary-card {
        background: #191d26 !important;
        border-color: #303746 !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25) !important;
        color: #f1f4fa !important;
    }

    input,
    textarea,
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {
        background: #1c212b !important;
        color: #f1f4fa !important;
        border-color: #343b4b !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #8992a5 !important;
    }

    .stButton > button {
        background: #202530 !important;
        color: #edf1f8 !important;
        border-color: #384052 !important;
    }

    .stButton > button:hover {
        background: #292f3d !important;
        border-color: #8b7cf6 !important;
        color: #a094ff !important;
    }

    .stButton > button[kind="primary"] {
        background: #7567e8 !important;
        color: #ffffff !important;
        border-color: #7567e8 !important;
    }

    button[data-baseweb="tab"] {
        color: #aeb6c7 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #a094ff !important;
    }

    [data-testid="stExpander"] {
        background: #191d26 !important;
        border-color: #303746 !important;
    }

    [data-testid="stAlert"] {
        background: #202530 !important;
        color: #e8ecf5 !important;
    }
}

/* Streamlit collapse/expand control */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarNavCollapse"] {
    color: var(--ms-text) !important;
}

button[data-testid="stSidebarCollapseButton"] svg,
button[data-testid="stSidebarNavCollapse"] svg {
    fill: currentColor !important;
}

/* Prevent invisible text caused by previous CSS */
.stMarkdown,
.stCaption,
.stText,
.stMetric,
.stRadio,
.stSelectbox,
.stCheckbox,
.stFileUploader {
    opacity: 1 !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def format_time(seconds):
    try:
        seconds = float(seconds)
    except Exception:
        return "00:00"

    minutes = int(seconds // 60)
    secs = int(seconds % 60)

    return f"{minutes:02d}:{secs:02d}"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def get_transcript():
    data = load_json(SPEAKER_TRANSCRIPT)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if isinstance(data.get("segments"), list):
            return data["segments"]

        if isinstance(data.get("transcript"), list):
            return data["transcript"]

    return []


def get_summary_text():
    data = load_json(SUMMARY)

    if isinstance(data, str):
        return data

    if isinstance(data, dict):
        for key in [
            "summary",
            "final_summary",
            "text",
            "generated_summary",
        ]:
            if isinstance(data.get(key), str):
                return data[key]

    return "No summary available yet."


def get_speaker_name(item):
    return (
        item.get("speaker")
        or item.get("speaker_label")
        or item.get("speaker_id")
        or "Unknown Speaker"
    )


def get_segment_text(item):
    return (
        item.get("text")
        or item.get("transcript")
        or ""
    ).strip()


def get_speaker_statistics(transcript):
    stats = {}

    for item in transcript:
        speaker = get_speaker_name(item)

        start = safe_float(item.get("start"))
        end = safe_float(item.get("end"))

        duration = max(0, end - start)

        if speaker not in stats:
            stats[speaker] = 0

        stats[speaker] += duration

    return stats


def get_duration(transcript):
    if not transcript:
        return 0

    ends = [
        safe_float(item.get("end"))
        for item in transcript
        if item.get("end") is not None
    ]

    return max(ends) if ends else 0


# ============================================================
# PIPELINE
# ============================================================

PIPELINE_STEPS = [
    ("Audio preprocessing", "src/audio_preprocessing.py"),
    ("Voice activity detection", "src/vad.py"),
    ("Whisper transcription", "src/transcription.py"),
    ("Speaker diarization", "src/diarization.py"),
    ("Transcript-speaker alignment", "src/merge_transcript.py"),
    ("NLP preprocessing", "src/nlp_preprocessing.py"),
    ("TF-IDF keywords", "src/tfidf_keywords.py"),
    ("Topic modeling", "src/topic_modeling.py"),
    ("Action-item extraction", "src/action_item_extraction.py"),
    ("Transformer summarization", "src/summarization.py"),
]


def run_pipeline(progress_placeholder, log_placeholder):

    logs = []

    # Ensure Homebrew FFmpeg libraries are visible.
    ffmpeg_lib = "/opt/homebrew/opt/ffmpeg/lib"

    env = os.environ.copy()

    current_dyld = env.get("DYLD_LIBRARY_PATH", "")

    if current_dyld:
        env["DYLD_LIBRARY_PATH"] = ffmpeg_lib + ":" + current_dyld
    else:
        env["DYLD_LIBRARY_PATH"] = ffmpeg_lib

    total = len(PIPELINE_STEPS)

    for index, (name, script) in enumerate(PIPELINE_STEPS):

        progress = index / total

        progress_placeholder.progress(
            progress,
            text=f"Processing: {name}"
        )

        logs.append(f"▶ {name}")
        log_placeholder.code("\n".join(logs))

        try:

            result = subprocess.run(
                [sys.executable, script],
                cwd=str(BASE_DIR),
                env=env,
                capture_output=True,
                text=True,
            )

            if result.stdout:
                logs.extend(
                    result.stdout.strip().splitlines()[-8:]
                )

            if result.returncode != 0:

                logs.append(
                    f"❌ {name} failed with exit code "
                    f"{result.returncode}"
                )

                if result.stderr:
                    logs.extend(
                        result.stderr.strip().splitlines()[-15:]
                    )

                log_placeholder.code("\n".join(logs))

                return False, "\n".join(logs)

            logs.append(f"✓ {name} completed")
            log_placeholder.code("\n".join(logs))

        except Exception as e:

            logs.append(f"❌ Error: {e}")

            log_placeholder.code("\n".join(logs))

            return False, "\n".join(logs)

    progress_placeholder.progress(
        1.0,
        text="Analysis completed successfully"
    )

    logs.append("✓ Complete meeting analysis finished.")

    log_placeholder.code("\n".join(logs))

    return True, "\n".join(logs)



# ============================================================
# MEETSENSE WELCOME SCREEN
# ============================================================

if "ms_started" not in st.session_state:
    st.session_state.ms_started = False

if "ms_theme" not in st.session_state:
    st.session_state.ms_theme = "light"


# ------------------------------------------------------------
# WELCOME SCREEN THEME
# ------------------------------------------------------------

if st.session_state.ms_theme == "dark":

    st.markdown("""
    <style>

    .stApp {
        background: #111318 !important;
        color: #F4F5F7 !important;
    }

    .welcome-shell {
        background: #111318;
        color: #F4F5F7;
    }

    .welcome-brand-name {
        color: #F4F5F7 !important;
    }

    .welcome-eyebrow {
        color: #9A90FF !important;
    }

    .welcome-title {
        color: #F7F8FA !important;
    }

    .welcome-subtitle {
        color: #B9C0CC !important;
    }

    .welcome-chip {
        background: #1B1F27;
        border-color: #303641;
        color: #CBD1DB !important;
    }

    .welcome-visual {
        background: #191D25;
        border-color: #303641;
        box-shadow: 0 25px 70px rgba(0,0,0,.28);
    }

    .visual-window {
        background: #20242D;
        border-color: #353B47;
    }

    .visual-title {
        color: #F0F2F5 !important;
    }

    .speaker-name {
        color: #E7EAF0 !important;
    }

    .speaker-line {
        background: #363C48 !important;
    }

    .summary-box {
        background: #2B2948 !important;
        border-color: #46416F !important;
    }

    .summary-label {
        color: #A79FFF !important;
    }

    .summary-text {
        color: #C8CCD7 !important;
    }

    .welcome-footer {
        color: #858D9C !important;
    }

    </style>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <style>

    .stApp {
        background: #F5F6F8 !important;
        color: #171927 !important;
    }

    .welcome-shell {
        background: #F5F6F8;
        color: #171927;
    }

    .welcome-brand-name {
        color: #171927 !important;
    }

    .welcome-eyebrow {
        color: #6258D9 !important;
    }

    .welcome-title {
        color: #171927 !important;
    }

    .welcome-subtitle {
        color: #606879 !important;
    }

    .welcome-chip {
        background: #FFFFFF;
        border-color: #E1E4EA;
        color: #4F5666 !important;
    }

    .welcome-visual {
        background: #FFFFFF;
        border-color: #E1E4EA;
        box-shadow: 0 25px 70px rgba(30,35,55,.08);
    }

    .visual-window {
        background: #FAFBFC;
        border-color: #E4E6EB;
    }

    .visual-title {
        color: #303647 !important;
    }

    .speaker-name {
        color: #363C4C !important;
    }

    .speaker-line {
        background: #E3E6EC !important;
    }

    .summary-box {
        background: #F0EFFF !important;
        border-color: #DDD9FF !important;
    }

    .summary-label {
        color: #6258D9 !important;
    }

    .summary-text {
        color: #61697A !important;
    }

    .welcome-footer {
        color: #8B92A0 !important;
    }

    </style>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# WELCOME PAGE
# ------------------------------------------------------------

if not st.session_state.ms_started:

    st.markdown("""
    <style>

    .welcome-shell {
        min-height: 78vh;
        padding: 20px 4vw 40px 4vw;
    }

    .welcome-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 55px;
    }

    .welcome-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .welcome-logo {
        width: 44px;
        height: 44px;
        border-radius: 13px;
        background: linear-gradient(135deg,#6258D9,#8177EA);
        color: white !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        font-weight: 800;
        box-shadow: 0 8px 25px rgba(98,88,217,.22);
    }

    .welcome-brand-name {
        font-size: 21px;
        font-weight: 800;
    }

    .welcome-badge {
        border: 1px solid;
        border-radius: 30px;
        padding: 8px 13px;
        font-size: 12px;
        font-weight: 600;
        color: #687082 !important;
    }

    .welcome-grid {
        max-width: 1120px;
        margin: auto;
        display: grid;
        grid-template-columns: 1fr 0.9fr;
        gap: 75px;
        align-items: center;
    }

    .welcome-eyebrow {
        font-size: 12px;
        letter-spacing: .11em;
        font-weight: 800;
        margin-bottom: 18px;
    }

    .welcome-title {
        font-size: clamp(45px,5.2vw,70px);
        line-height: 1.02;
        letter-spacing: -.045em;
        font-weight: 800;
        margin: 0 0 22px 0;
    }

    .welcome-subtitle {
        max-width: 590px;
        font-size: 17px;
        line-height: 1.7;
        margin-bottom: 27px;
    }

    .welcome-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 25px;
    }

    .welcome-chip {
        border: 1px solid;
        border-radius: 8px;
        padding: 8px 11px;
        font-size: 11px;
        font-weight: 650;
    }

    .welcome-visual {
        border: 1px solid;
        border-radius: 24px;
        padding: 24px;
        min-height: 370px;
    }

    .visual-toolbar {
        display: flex;
        gap: 6px;
        margin-bottom: 20px;
    }

    .toolbar-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #A0A6B2;
    }

    .toolbar-dot.active {
        background: #6258D9;
    }

    .visual-window {
        border: 1px solid;
        border-radius: 16px;
        padding: 18px;
    }

    .visual-title {
        font-size: 13px;
        font-weight: 750;
        margin-bottom: 14px;
    }

    .speaker-row {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 11px 0;
        border-bottom: 1px solid;
    }

    .speaker-avatar {
        width: 33px;
        height: 33px;
        border-radius: 50%;
        background: #E8E6FF;
        color: #6258D9 !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        font-weight: 800;
        flex-shrink: 0;
    }

    .speaker-name {
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .speaker-line {
        height: 6px;
        border-radius: 10px;
        width: 82%;
    }

    .speaker-line.medium {
        width: 68%;
    }

    .speaker-line.short {
        width: 48%;
    }

    .summary-box {
        margin-top: 17px;
        padding: 13px;
        border: 1px solid;
        border-radius: 11px;
    }

    .summary-label {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .08em;
    }

    .summary-text {
        font-size: 11px;
        line-height: 1.55;
        margin-top: 5px;
    }

    .welcome-footer {
        text-align: center;
        margin-top: 55px;
        font-size: 11px;
    }

    @media (max-width: 850px) {

        .welcome-grid {
            grid-template-columns: 1fr;
            gap: 35px;
        }

        .welcome-title {
            font-size: 47px;
        }

    }

    </style>
    """, unsafe_allow_html=True)


    # Theme control
    theme_left, theme_right = st.columns([8, 2])

    with theme_right:

        dark_mode = st.toggle(
            "Dark mode",
            value=(st.session_state.ms_theme == "dark"),
            key="welcome_theme_toggle"
        )

        selected_theme = "dark" if dark_mode else "light"

        if selected_theme != st.session_state.ms_theme:
            st.session_state.ms_theme = selected_theme
            st.rerun()


    st.markdown("""
    <div class="welcome-shell">

        <div class="welcome-top">

            <div class="welcome-brand">

                <div class="welcome-logo">
                    ∞
                </div>

                <div class="welcome-brand-name">
                    MeetSense
                </div>

            </div>

            <div class="welcome-badge">
                Speaker-aware meeting intelligence
            </div>

        </div>


        <div class="welcome-grid">

            <div>

                <div class="welcome-eyebrow">
                    AI MEETING INTELLIGENCE
                </div>

                <div class="welcome-title">
                    Meeting intelligence,
                    <br>
                    simplified.
                </div>

                <div class="welcome-subtitle">
                    Turn meeting recordings into speaker-aware
                    transcripts, concise summaries, decisions,
                    topics and actionable tasks.
                </div>

                <div class="welcome-chips">

                    <div class="welcome-chip">
                        Whisper transcription
                    </div>

                    <div class="welcome-chip">
                        Speaker diarization
                    </div>

                    <div class="welcome-chip">
                        NLP analysis
                    </div>

                    <div class="welcome-chip">
                        Transformer summary
                    </div>

                </div>

            </div>


            <div class="welcome-visual">

                <div class="visual-toolbar">
                    <div class="toolbar-dot active"></div>
                    <div class="toolbar-dot"></div>
                    <div class="toolbar-dot"></div>
                </div>

                <div class="visual-window">

                    <div class="visual-title">
                        Meeting overview
                    </div>

                    <div class="speaker-row">

                        <div class="speaker-avatar">
                            S1
                        </div>

                        <div style="flex:1;">

                            <div class="speaker-name">
                                Speaker 01
                            </div>

                            <div class="speaker-line"></div>

                        </div>

                    </div>


                    <div class="speaker-row">

                        <div class="speaker-avatar">
                            S2
                        </div>

                        <div style="flex:1;">

                            <div class="speaker-name">
                                Speaker 02
                            </div>

                            <div class="speaker-line medium"></div>

                        </div>

                    </div>


                    <div class="speaker-row">

                        <div class="speaker-avatar">
                            S3
                        </div>

                        <div style="flex:1;">

                            <div class="speaker-name">
                                Speaker 03
                            </div>

                            <div class="speaker-line short"></div>

                        </div>

                    </div>


                    <div class="summary-box">

                        <div class="summary-label">
                            AI SUMMARY
                        </div>

                        <div class="summary-text">
                            Key discussion points, decisions
                            and action items are automatically
                            organized from the meeting.
                        </div>

                    </div>

                </div>

            </div>

        </div>


        <div class="welcome-footer">
            MeetSense • Speaker-Aware Meeting Transcription & Summarization
        </div>

    </div>
    """, unsafe_allow_html=True)


    # Start button
    start_left, start_center, start_right = st.columns([4, 2, 4])

    with start_center:

        if st.button(
            "Start  →",
            type="primary",
            use_container_width=True,
            key="welcome_start_button"
        ):

            st.session_state.ms_started = True
            st.rerun()


    st.stop()


# ------------------------------------------------------------
# WORKSPACE THEME CONTROL
# ------------------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:11px;
            font-weight:800;
            letter-spacing:.08em;
            margin-top:10px;
            margin-bottom:8px;
        ">
            APPEARANCE
        </div>
        """,
        unsafe_allow_html=True
    )

    workspace_dark = st.toggle(
        "Dark mode",
        value=(st.session_state.ms_theme == "dark"),
        key="workspace_theme_toggle"
    )

    selected_workspace_theme = (
        "dark" if workspace_dark else "light"
    )

    if selected_workspace_theme != st.session_state.ms_theme:

        st.session_state.ms_theme = selected_workspace_theme
        st.rerun()

    if st.button(
        "← Back to Welcome",
        use_container_width=True,
        key="workspace_back_button"
    ):

        st.session_state.ms_started = False
        st.rerun()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">∞</div>
            <div class="brand-name">MeetSense</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### GENERAL")

    page = st.radio(
        "",
        [
            "Dashboard",
            "Meeting",
            "My Meetings",
            "Live Meeting",
        ],
        label_visibility="collapsed",
    )

    st.markdown("### INTELLIGENCE")

    intelligence_page = st.radio(
        "",
        [
            "Transcript",
            "AI Summary",
            "Action Items",
            "Decisions",
        ],
        label_visibility="collapsed",
    )

    st.markdown("### ANALYTICS")

    analytics_page = st.radio(
        "",
        [
            "Keywords & Topics",
            "Evaluation",
            "Speakers",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.caption("Speaker-Aware Meeting Intelligence")

    st.markdown(
        """
        <div style="
            background:#f6f4ff;
            padding:13px;
            border-radius:12px;
            font-size:11px;
            color:#6258a8;
        ">
        <b>MeetSense AI</b><br>
        Whisper + PyAnnote + NLP + Transformers
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================

# ============================================================
# MEETSENSE THEME SWITCH
# ============================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark_mode = st.session_state.dark_mode

# Theme-specific colors
if dark_mode:
    THEME_BG = "#0f1117"
    THEME_SURFACE = "#171a23"
    THEME_SURFACE_2 = "#1e222d"
    THEME_BORDER = "#303543"
    THEME_TEXT = "#f3f4f8"
    THEME_TEXT_SECONDARY = "#b8bdca"
    THEME_TEXT_MUTED = "#8d94a5"
    THEME_INPUT = "#1b1f29"
else:
    THEME_BG = "#f6f7fb"
    THEME_SURFACE = "#ffffff"
    THEME_SURFACE_2 = "#f8f9fc"
    THEME_BORDER = "#e7e8ef"
    THEME_TEXT = "#171827"
    THEME_TEXT_SECONDARY = "#5e6273"
    THEME_TEXT_MUTED = "#747889"
    THEME_INPUT = "#ffffff"

# Theme switch in the main application
theme_col_1, theme_col_2 = st.columns([8, 2])

with theme_col_2:
    st.toggle(
        "🌙 Dark mode",
        key="dark_mode",
        help="Switch between light and dark appearance.",
    )

# Re-read state after the toggle
dark_mode = st.session_state.dark_mode

if dark_mode:
    THEME_BG = "#0f1117"
    THEME_SURFACE = "#171a23"
    THEME_SURFACE_2 = "#1e222d"
    THEME_BORDER = "#303543"
    THEME_TEXT = "#f3f4f8"
    THEME_TEXT_SECONDARY = "#b8bdca"
    THEME_TEXT_MUTED = "#8d94a5"
    THEME_INPUT = "#1b1f29"
else:
    THEME_BG = "#f6f7fb"
    THEME_SURFACE = "#ffffff"
    THEME_SURFACE_2 = "#f8f9fc"
    THEME_BORDER = "#e7e8ef"
    THEME_TEXT = "#171827"
    THEME_TEXT_SECONDARY = "#5e6273"
    THEME_TEXT_MUTED = "#747889"
    THEME_INPUT = "#ffffff"

# Dynamic theme overrides
st.markdown(
    f"""
    <style>

    /* ========================================================
       MEETSENSE DYNAMIC THEME
       ======================================================== */

    /* Main application */
    .stApp {{
        background: {THEME_BG} !important;
        color: {THEME_TEXT} !important;
    }}

    .main {{
        background: {THEME_BG} !important;
    }}

    .block-container {{
        color: {THEME_TEXT} !important;
    }}

    /* --------------------------------------------------------
       Sidebar
       -------------------------------------------------------- */

    section[data-testid="stSidebar"] {{
        background: {THEME_SURFACE} !important;
        border-right: 1px solid {THEME_BORDER} !important;
        color: {THEME_TEXT} !important;
    }}

    section[data-testid="stSidebar"] * {{
        color: {THEME_TEXT} !important;
    }}

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {{
        color: {THEME_TEXT} !important;
        opacity: 1 !important;
    }}

    section[data-testid="stSidebar"] [role="radiogroup"] {{
        color: {THEME_TEXT} !important;
    }}

    section[data-testid="stSidebar"] [role="radiogroup"] label {{
        color: {THEME_TEXT} !important;
        opacity: 1 !important;
    }}

    section[data-testid="stSidebar"] [role="radiogroup"] label * {{
        color: {THEME_TEXT} !important;
        opacity: 1 !important;
    }}

    section[data-testid="stSidebar"] [role="radio"][aria-checked="true"] {{
        color: #5b4bff !important;
    }}

    section[data-testid="stSidebar"] [role="radio"][aria-checked="true"] * {{
        color: #5b4bff !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {{
        color: {THEME_TEXT} !important;
        opacity: 1 !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{
        color: {THEME_TEXT_MUTED} !important;
        opacity: 1 !important;
    }}

    /* --------------------------------------------------------
       Existing MeetSense cards
       -------------------------------------------------------- */

    .card {{
        background: {THEME_SURFACE} !important;
        border-color: {THEME_BORDER} !important;
        color: {THEME_TEXT} !important;
    }}

    .card * {{
        color: {THEME_TEXT} !important;
    }}

    .card-title {{
        color: {THEME_TEXT} !important;
    }}

    .page-title {{
        color: {THEME_TEXT} !important;
    }}

    .page-subtitle {{
        color: {THEME_TEXT_SECONDARY} !important;
    }}

    .section-heading {{
        color: {THEME_TEXT} !important;
    }}

    .topbar {{
        background: {THEME_SURFACE} !important;
        border-color: {THEME_BORDER} !important;
    }}

    .top-title {{
        color: {THEME_TEXT_SECONDARY} !important;
    }}

    /* --------------------------------------------------------
       Transcript
       -------------------------------------------------------- */

    .transcript-item {{
        background: {THEME_SURFACE} !important;
        border-color: {THEME_BORDER} !important;
    }}

    .transcript-item * {{
        color: {THEME_TEXT} !important;
    }}

    .transcript-text {{
        color: {THEME_TEXT} !important;
    }}

    .transcript-time {{
        color: {THEME_TEXT_MUTED} !important;
    }}

    .transcript-speaker {{
        color: #5b4bff !important;
    }}

    /* --------------------------------------------------------
       Streamlit widgets
       -------------------------------------------------------- */

    [data-testid="stExpander"] {{
        background: {THEME_SURFACE} !important;
        border-color: {THEME_BORDER} !important;
    }}

    [data-testid="stExpander"] * {{
        color: {THEME_TEXT} !important;
    }}

    [data-testid="stAlert"] {{
        background: {THEME_SURFACE_2} !important;
        color: {THEME_TEXT} !important;
        border-color: {THEME_BORDER} !important;
    }}

    [data-testid="stAlert"] * {{
        color: {THEME_TEXT} !important;
    }}

    /* Text inputs */
    .stTextInput input,
    .stTextArea textarea {{
        background: {THEME_INPUT} !important;
        color: {THEME_TEXT} !important;
        border-color: {THEME_BORDER} !important;
    }}

    .stTextInput label,
    .stTextArea label {{
        color: {THEME_TEXT} !important;
    }}

    /* Select boxes */
    [data-baseweb="select"] > div {{
        background: {THEME_INPUT} !important;
        color: {THEME_TEXT} !important;
        border-color: {THEME_BORDER} !important;
    }}

    [data-baseweb="select"] * {{
        color: {THEME_TEXT} !important;
    }}

    /* File uploader */
    [data-testid="stFileUploader"] {{
        background: {THEME_SURFACE} !important;
        border-color: {THEME_BORDER} !important;
    }}

    [data-testid="stFileUploader"] * {{
        color: {THEME_TEXT} !important;
    }}

    /* Metrics */
    [data-testid="stMetric"] {{
        background: {THEME_SURFACE} !important;
        border-color: {THEME_BORDER} !important;
    }}

    [data-testid="stMetric"] label {{
        color: {THEME_TEXT_SECONDARY} !important;
    }}

    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {THEME_TEXT} !important;
    }}

    /* Tabs */
    button[data-baseweb="tab"] {{
        color: {THEME_TEXT_SECONDARY} !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: #5b4bff !important;
    }}

    /* Buttons */
    .stButton button {{
        color: {THEME_TEXT} !important;
        border-color: {THEME_BORDER} !important;
    }}

    /* Tables */
    [data-testid="stDataFrame"] {{
        border-color: {THEME_BORDER} !important;
    }}

    /* Markdown */
    [data-testid="stMarkdownContainer"] {{
        color: {THEME_TEXT} !important;
    }}

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {{
        color: {THEME_TEXT} !important;
    }}

    /* --------------------------------------------------------
       Header / collapsed sidebar control
       -------------------------------------------------------- */

    header {{
        visibility: visible !important;
        background: transparent !important;
    }}

    [data-testid="stSidebarCollapsedControl"] {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
    }}

    [data-testid="stSidebarCollapsedControl"] button {{
        background: {THEME_SURFACE} !important;
        color: {THEME_TEXT} !important;
        border: 1px solid {THEME_BORDER} !important;
        opacity: 1 !important;
    }}

    [data-testid="stSidebarCollapsedControl"] svg {{
        color: {THEME_TEXT} !important;
        fill: {THEME_TEXT} !important;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# THEME SWITCH
# ============================================================

with st.sidebar:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    theme_choice = st.radio(
        "Appearance",
        ["☀ Light", "◐ Dark"],
        index=0 if st.session_state.theme == "light" else 1,
        horizontal=True,
        key="theme_selector",
    )

    new_theme = "dark" if theme_choice == "◐ Dark" else "light"

    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()


# TOP BAR
# ============================================================

st.markdown(
    """
    <div class="topbar">
        <div class="top-title">
            Meeting Intelligence Workspace
        </div>
        <div>
            <span class="status-pill">● System Ready</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD / NEW MEETING
# ============================================================

with st.expander("＋  New Meeting / Analyze Audio", expanded=False):

    st.markdown(
        "### Upload a meeting recording"
    )

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Upload a meeting recording for complete speaker-aware analysis.",
    )

    if uploaded_file:

        st.audio(
            uploaded_file,
            format=uploaded_file.type,
        )

        st.markdown(
            f"**Selected:** `{uploaded_file.name}`"
        )

        if st.button(
            "🚀 Analyze Meeting",
            type="primary",
            use_container_width=True,
        ):

            target = AUDIO_DIR / "meeting.wav"

            try:

                with open(target, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.success(
                    "Audio uploaded successfully."
                )

                progress_placeholder = st.empty()
                log_placeholder = st.empty()

                with st.spinner(
                    "Running complete meeting analysis..."
                ):

                    success, logs = run_pipeline(
                        progress_placeholder,
                        log_placeholder,
                    )

                if success:

                    st.success(
                        "Meeting analysis completed successfully."
                    )

                    st.rerun()

                else:

                    st.error(
                        "The pipeline stopped because one of the processing stages failed."
                    )

                    with st.expander(
                        "View processing logs"
                    ):
                        st.code(logs)

            except Exception as e:

                st.error(
                    f"Could not process uploaded audio: {e}"
                )


# ============================================================
# LOAD DATA
# ============================================================

transcript = get_transcript()
summary_text = get_summary_text()

tfidf_data = load_json(TFIDF) or {}
topic_data = load_json(TOPICS) or {}
action_data = load_json(ACTION_ITEMS) or {}

wer_data = load_json(WER) or {}
der_data = load_json(DER) or {}
rouge_data = load_json(ROUGE) or {}
action_eval_data = load_json(ACTION_EVAL) or {}

speaker_stats = get_speaker_statistics(transcript)

duration = get_duration(transcript)

speaker_count = len(speaker_stats)

segment_count = len(transcript)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="page-title">Meeting Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Your intelligent meeting assistant — transcription, speakers, summaries and insights.</div>',
        unsafe_allow_html=True,
    )

    # Meeting header
    st.markdown(
        """
        <div class="meeting-header">
            <div class="meeting-name">
                Meeting Analysis
            </div>
            <div class="meeting-meta">
                🎙️ Speaker-Aware Meeting • AI Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stats
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-icon">⏱️</div>
                <div class="stat-number">
                    {format_time(duration)}
                </div>
                <div class="stat-label">
                    Meeting Duration
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-icon">👥</div>
                <div class="stat-number">
                    {speaker_count}
                </div>
                <div class="stat-label">
                    Speakers
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-icon">📝</div>
                <div class="stat-number">
                    {segment_count}
                </div>
                <div class="stat-label">
                    Transcript Segments
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:

        action_items = []

        if isinstance(action_data, dict):

            action_items = (
                action_data.get("action_items")
                or action_data.get("actions")
                or []
            )

        elif isinstance(action_data, list):

            action_items = action_data

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-icon">✅</div>
                <div class="stat-number">
                    {len(action_items)}
                </div>
                <div class="stat-label">
                    Action Items
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Main dashboard columns
    left, right = st.columns([1.45, 1])

    with left:

        # Audio
        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    🎧 Meeting Recording
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        audio_path = AUDIO_DIR / "meeting.wav"

        if audio_path.exists():
            st.audio(
                str(audio_path),
                format="audio/wav",
            )
        else:
            st.info(
                "Upload a meeting recording to begin."
            )

        # Transcript preview
        st.markdown(
            '<div class="section-heading">📝 Meeting Transcription</div>',
            unsafe_allow_html=True,
        )

        for item in transcript[:8]:

            speaker = get_speaker_name(item)
            text = get_segment_text(item)

            start = format_time(
                item.get("start", 0)
            )

            st.markdown(
                f"""
                <div class="transcript-item">
                    <div class="transcript-header">
                        <span class="transcript-speaker">
                            {speaker}
                        </span>
                        <span class="transcript-time">
                            {start}
                        </span>
                    </div>
                    <div class="transcript-text">
                        {text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if len(transcript) > 8:
            st.caption(
                f"Showing 8 of {len(transcript)} segments. Open Transcript for the complete meeting."
            )

    with right:

        st.markdown(
            """
            <div class="section-heading">
                ✨ AI Meeting Summary
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="summary-box">
                {summary_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-heading">
                👥 Speakers
            </div>
            """,
            unsafe_allow_html=True,
        )

        if speaker_stats:

            for speaker, speaker_duration in speaker_stats.items():

                percentage = (
                    speaker_duration / duration * 100
                    if duration
                    else 0
                )

                st.markdown(
                    f"""
                    <div class="speaker-row">
                        <div style="display:flex;justify-content:space-between;">
                            <span class="speaker-name">
                                {speaker}
                            </span>
                            <span class="speaker-time">
                                {format_time(speaker_duration)}
                            </span>
                        </div>
                        <div style="
                            margin-top:8px;
                            height:5px;
                            background:#ececf5;
                            border-radius:10px;
                        ">
                            <div style="
                                width:{min(percentage,100):.1f}%;
                                height:5px;
                                background:#6658df;
                                border-radius:10px;
                            "></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Top keywords
        st.markdown(
            """
            <div class="section-heading">
                🔑 Key Topics & Keywords
            </div>
            """,
            unsafe_allow_html=True,
        )

        keywords = []

        if isinstance(tfidf_data, dict):

            keywords = (
                tfidf_data.get("keywords")
                or tfidf_data.get("top_keywords")
                or tfidf_data.get("terms")
                or []
            )

        if isinstance(keywords, list):

            for item in keywords[:15]:

                if isinstance(item, dict):

                    word = (
                        item.get("term")
                        or item.get("keyword")
                        or item.get("word")
                        or ""
                    )

                else:

                    word = str(item)

                if word:

                    st.markdown(
                        f'<span class="tag">{word}</span>',
                        unsafe_allow_html=True,
                    )


# ============================================================
# MEETING PAGE
# ============================================================

elif page == "Meeting":

    st.markdown(
        '<div class="page-title">Meeting Details</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Review your recording, speakers, transcript and AI-generated insights.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="meeting-header">
            <div style="display:flex;justify-content:space-between;">
                <div>
                    <div class="meeting-name">
                        Meeting Recording
                    </div>
                    <div class="meeting-meta">
                        🎙️ {speaker_count} speakers
                        &nbsp; • &nbsp;
                        ⏱️ {format_time(duration)}
                        &nbsp; • &nbsp;
                        📝 {segment_count} segments
                    </div>
                </div>
                <div>
                    <span class="status-pill">
                        ANALYZED
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if (AUDIO_DIR / "meeting.wav").exists():

        st.markdown(
            '<div class="card-title">▶ Meeting Recording</div>',
            unsafe_allow_html=True,
        )

        st.audio(
            str(AUDIO_DIR / "meeting.wav"),
            format="audio/wav",
        )

    st.markdown(
        '<div class="section-heading">Meeting Overview</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">✨ Overview</div>
                <p style="font-size:14px;line-height:1.7;color:#555969;">
                    {summary_text}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">🎯 AI Insights</div>
                <p style="font-size:14px;line-height:1.7;color:#555969;">
                    The meeting has been processed using neural speech
                    recognition, speaker diarization and NLP analysis.
                    Explore the sections below for detailed insights.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# MY MEETINGS
# ============================================================

elif page == "My Meetings":

    st.markdown(
        '<div class="page-title">My Meetings</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Your analyzed meeting recordings.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="card">
            <div style="display:flex;justify-content:space-between;">
                <div>
                    <div class="card-title">
                        🎙️ Meeting Recording
                    </div>
                    <div class="card-subtitle">
                        {format_time(duration)}
                        &nbsp; • &nbsp;
                        {speaker_count} speakers
                        &nbsp; • &nbsp;
                        {segment_count} transcript segments
                    </div>
                </div>
                <span class="status-pill">
                    ANALYZED
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "The current version stores one analyzed meeting locally. "
        "A multi-meeting database can be added after the core application is complete."
    )


# ============================================================
# LIVE MEETING
# ============================================================

elif page == "Live Meeting":

    st.markdown(
        '<div class="page-title">Live Meeting</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Near-real-time meeting transcription.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                🔴 Live Meeting Mode
            </div>

            <p style="font-size:14px;line-height:1.7;color:#5e6273;">
                MeetSense can process microphone audio using Whisper.
                Speaker-aware live processing is designed as a rolling-window
                pipeline because speaker diarization is not truly instantaneous.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    live_file = TRANSCRIPT_DIR / "live_meeting.json"

    if live_file.exists():

        live_data = load_json(live_file)

        if isinstance(live_data, list):

            st.markdown(
                '<div class="section-heading">Live Transcript</div>',
                unsafe_allow_html=True,
            )

            for item in live_data:

                text = get_segment_text(item)

                if not text:
                    continue

                st.markdown(
                    f"""
                    <div class="transcript-item">
                        <div class="transcript-header">
                            <span class="transcript-speaker">
                                🎙️ Live
                            </span>
                            <span class="transcript-time">
                                {format_time(item.get("start", 0))}
                            </span>
                        </div>
                        <div class="transcript-text">
                            {text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:

        st.info(
            "No live transcript is available yet. "
            "The local microphone transcription module can be connected here next."
        )


# ============================================================
# TRANSCRIPT PAGE
# ============================================================

elif intelligence_page == "Transcript":

    st.markdown(
        '<div class="page-title">Meeting Transcript</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Timestamped transcription with speaker attribution.</div>',
        unsafe_allow_html=True,
    )

    search = st.text_input(
        "🔎 Search transcript",
        placeholder="Search words, sentences or speakers...",
    )

    filtered = transcript

    if search:

        query = search.lower()

        filtered = [
            item
            for item in transcript
            if query in get_segment_text(item).lower()
            or query in get_speaker_name(item).lower()
        ]

    st.caption(
        f"{len(filtered)} transcript segments"
    )

    for item in filtered:

        speaker = get_speaker_name(item)
        text = get_segment_text(item)

        st.markdown(
            f"""
            <div class="transcript-item">
                <div class="transcript-header">
                    <span class="transcript-speaker">
                        {speaker}
                    </span>
                    <span class="transcript-time">
                        {format_time(item.get("start",0))}
                        —
                        {format_time(item.get("end",0))}
                    </span>
                </div>

                <div class="transcript-text">
                    {text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# SUMMARY PAGE
# ============================================================

elif intelligence_page == "AI Summary":

    st.markdown(
        '<div class="page-title">AI Meeting Summary</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Transformer-generated summary of the meeting.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="summary-box">
            {summary_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading">🎯 Key Meeting Information</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Duration",
            format_time(duration),
        )

    with c2:
        st.metric(
            "Speakers",
            speaker_count,
        )

    with c3:
        st.metric(
            "Segments",
            segment_count,
        )


# ============================================================
# ACTION ITEMS
# ============================================================

elif intelligence_page == "Action Items":

    st.markdown(
        '<div class="page-title">Action Items</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Tasks and follow-up actions extracted from the meeting.</div>',
        unsafe_allow_html=True,
    )

    actions = []

    if isinstance(action_data, dict):

        actions = (
            action_data.get("action_items")
            or action_data.get("actions")
            or []
        )

    elif isinstance(action_data, list):

        actions = action_data

    if not actions:

        st.info(
            "No action items were found."
        )

    else:

        for index, action in enumerate(actions):

            if isinstance(action, dict):

                text = (
                    action.get("action")
                    or action.get("text")
                    or action.get("item")
                    or str(action)
                )

                speaker = (
                    action.get("speaker")
                    or action.get("owner")
                    or ""
                )

            else:

                text = str(action)
                speaker = ""

            st.markdown(
                f"""
                <div class="action-item">
                    <div class="action-title">
                        □ &nbsp; {text}
                    </div>
                    {
                        f'<div class="card-subtitle" style="margin-top:6px;">Speaker: {speaker}</div>'
                        if speaker else ""
                    }
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# DECISIONS
# ============================================================

elif intelligence_page == "Decisions":

    st.markdown(
        '<div class="page-title">Meeting Decisions</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Important decisions identified during the meeting.</div>',
        unsafe_allow_html=True,
    )

    decisions = []

    if isinstance(action_data, dict):

        decisions = (
            action_data.get("decisions")
            or action_data.get("decision_items")
            or []
        )

    if not decisions:

        st.info(
            "No decisions were extracted."
        )

    else:

        for decision in decisions:

            if isinstance(decision, dict):

                text = (
                    decision.get("decision")
                    or decision.get("text")
                    or str(decision)
                )

            else:

                text = str(decision)

            st.markdown(
                f"""
                <div class="decision">
                    💡 <b>Decision</b><br>
                    {text}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# KEYWORDS + TOPICS
# ============================================================

elif analytics_page == "Keywords & Topics":

    st.markdown(
        '<div class="page-title">Keywords & Topics</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">NLP analysis using TF-IDF and LDA topic modeling.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading">🔑 TF-IDF Keywords</div>',
        unsafe_allow_html=True,
    )

    keywords = []

    if isinstance(tfidf_data, dict):

        keywords = (
            tfidf_data.get("keywords")
            or tfidf_data.get("top_keywords")
            or tfidf_data.get("terms")
            or []
        )

    if keywords:

        for item in keywords:

            if isinstance(item, dict):

                word = (
                    item.get("term")
                    or item.get("keyword")
                    or item.get("word")
                    or ""
                )

                score = item.get("score")

                if score is not None:

                    st.markdown(
                        f"""
                        <div class="speaker-row">
                            <div style="display:flex;justify-content:space-between;">
                                <span class="speaker-name">
                                    {word}
                                </span>
                                <span class="speaker-time">
                                    {safe_float(score):.4f}
                                </span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                elif word:

                    st.markdown(
                        f'<span class="tag">{word}</span>',
                        unsafe_allow_html=True,
                    )

            else:

                st.markdown(
                    f'<span class="tag">{item}</span>',
                    unsafe_allow_html=True,
                )

    else:

        st.info(
            "No TF-IDF keyword data available."
        )

    st.markdown(
        '<div class="section-heading">🧠 LDA Topics</div>',
        unsafe_allow_html=True,
    )

    topics = []

    if isinstance(topic_data, dict):

        topics = (
            topic_data.get("topics")
            or topic_data.get("lda_topics")
            or []
        )

    elif isinstance(topic_data, list):

        topics = topic_data

    if topics:

        for index, topic in enumerate(topics, start=1):

            if isinstance(topic, dict):

                words = (
                    topic.get("words")
                    or topic.get("terms")
                    or topic.get("top_words")
                    or []
                )

                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-title">
                            Topic {index}
                        </div>
                    """,
                    unsafe_allow_html=True,
                )

                if isinstance(words, list):

                    for word in words:

                        if isinstance(word, dict):
                            word = (
                                word.get("word")
                                or word.get("term")
                                or str(word)
                            )

                        st.markdown(
                            f'<span class="tag">{word}</span>',
                            unsafe_allow_html=True,
                        )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-title">
                            Topic {index}
                        </div>
                        {topic}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:

        st.info(
            "No topic modeling data available."
        )


# ============================================================
# EVALUATION
# ============================================================

elif analytics_page == "Evaluation":

    st.markdown(
        '<div class="page-title">Model Evaluation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Quantitative evaluation of transcription, diarization, summarization and action extraction.</div>',
        unsafe_allow_html=True,
    )

    # ---------------- WER ----------------

    st.markdown(
        '<div class="section-heading">🎙️ Speech Recognition</div>',
        unsafe_allow_html=True,
    )

    wer_value = safe_float(
        wer_data.get("wer", 0)
        if isinstance(wer_data, dict)
        else 0
    )

    der_value = safe_float(
        der_data.get("der", 0)
        if isinstance(der_data, dict)
        else 0
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-name">Word Error Rate</div>
                <div class="metric-value">
                    {wer_value * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-name">Diarization Error Rate</div>
                <div class="metric-value">
                    {der_value * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------- ROUGE ----------------

    st.markdown(
        '<div class="section-heading">📝 Summarization</div>',
        unsafe_allow_html=True,
    )

    rouge1 = 0
    rouge2 = 0
    rougeL = 0

    if isinstance(rouge_data, dict):

        r1 = rouge_data.get("rouge1", {})
        r2 = rouge_data.get("rouge2", {})
        rl = rouge_data.get("rougeL", {})

        if isinstance(r1, dict):
            rouge1 = safe_float(
                r1.get("f1")
                or r1.get("fmeasure")
                or 0
            )

        if isinstance(r2, dict):
            rouge2 = safe_float(
                r2.get("f1")
                or r2.get("fmeasure")
                or 0
            )

        if isinstance(rl, dict):
            rougeL = safe_float(
                rl.get("f1")
                or rl.get("fmeasure")
                or 0
            )

    r1, r2, r3 = st.columns(3)

    with r1:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-name">ROUGE-1 F1</div>
                <div class="metric-value">
                    {rouge1 * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r2:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-name">ROUGE-2 F1</div>
                <div class="metric-value">
                    {rouge2 * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with r3:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-name">ROUGE-L F1</div>
                <div class="metric-value">
                    {rougeL * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------- ACTION ITEM ----------------

    st.markdown(
        '<div class="section-heading">✅ Action Item Extraction</div>',
        unsafe_allow_html=True,
    )

    action_f1 = 0
    action_precision = 0
    action_recall = 0

    if isinstance(action_eval_data, dict):

        action_f1 = safe_float(
            action_eval_data.get("f1")
            or action_eval_data.get("f1_score")
            or 0
        )

        action_precision = safe_float(
            action_eval_data.get("precision")
            or 0
        )

        action_recall = safe_float(
            action_eval_data.get("recall")
            or 0
        )

    a1, a2, a3 = st.columns(3)

    with a1:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-name">Precision</div>
                <div class="metric-value">
                    {action_precision * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a2:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-name">Recall</div>
                <div class="metric-value">
                    {action_recall * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a3:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-name">F1 Score</div>
                <div class="metric-value">
                    {action_f1 * 100:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# SPEAKERS
# ============================================================

elif analytics_page == "Speakers":

    st.markdown(
        '<div class="page-title">Speakers</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Speaker participation derived from diarization.</div>',
        unsafe_allow_html=True,
    )

    if not speaker_stats:

        st.info(
            "No speaker information is available yet."
        )

    else:

        for index, (speaker, seconds) in enumerate(
            speaker_stats.items(),
            start=1,
        ):

            percentage = (
                seconds / duration * 100
                if duration
                else 0
            )

            st.markdown(
                f"""
                <div class="card">
                    <div style="display:flex;justify-content:space-between;">
                        <div>
                            <div class="card-title">
                                👤 {speaker}
                            </div>
                            <div class="card-subtitle">
                                Speaker {index}
                            </div>
                        </div>

                        <div style="
                            font-size:20px;
                            font-weight:800;
                            color:#5142d8;
                        ">
                            {percentage:.1f}%
                        </div>
                    </div>

                    <div style="
                        margin-top:15px;
                        height:8px;
                        background:#ededf5;
                        border-radius:10px;
                    ">
                        <div style="
                            width:{min(percentage,100):.1f}%;
                            height:8px;
                            background:#6658df;
                            border-radius:10px;
                        "></div>
                    </div>

                    <div style="
                        margin-top:8px;
                        font-size:12px;
                        color:#777b8d;
                    ">
                        Speaking time: {format_time(seconds)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# DOWNLOAD SECTION
# ============================================================

st.markdown("---")

with st.expander("📥 Export Meeting Results"):

    download_columns = st.columns(4)

    files_for_download = [
        ("Speaker Transcript", SPEAKER_TRANSCRIPT),
        ("Summary", SUMMARY),
        ("NLP Analysis", NLP_PROCESSED),
        ("Evaluation - WER", WER),
    ]

    for column, (label, path) in zip(
        download_columns,
        files_for_download,
    ):

        with column:

            if path.exists():

                with open(
                    path,
                    "rb",
                ) as f:

                    st.download_button(
                        label,
                        data=f.read(),
                        file_name=path.name,
                        mime="application/json",
                        use_container_width=True,
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#9699a7;
        font-size:11px;
        margin-top:30px;
        padding:15px;
    ">
        MeetSense • Speaker-Aware Meeting Transcription & Summarization System
        <br>
        Whisper • PyAnnote • Silero VAD • NLTK • TF-IDF • LDA • T5
    </div>
    """,
    unsafe_allow_html=True,
)