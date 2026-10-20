import contextlib
import json
import os
import subprocess
import textwrap
import sys
import time
from pathlib import Path

import streamlit as st

_SRC_DIR = Path(__file__).resolve().parent / "src"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Local sounddevice mic path (Stage 4), not the browser/webrtc path --
# live_web.py and streamlit_webrtc remain unused here but untouched on
# disk as prior art. live_controller.py owns a worker thread that
# calls into live_pipeline.py's LiveMeetingSession; this script must
# never call VAD/ASR/diarization itself, only read controller state.
#
# NOT imported here at module level, deliberately: live_controller
# imports live_pipeline, which imports vad/transcription/diarization/
# summarization, which import torch/whisper/pyannote/transformers at
# THEIR module level. If this file imported live_controller here, every
# page load -- Dashboard, batch results, anything -- would pay that
# import cost (minutes, under memory pressure on this 8GB machine)
# before rendering a single pixel. Imported lazily instead, inside the
# Live Meeting page body, only once the user clicks Start.


# ============================================================
# THEME
# ============================================================
#
# Single source of truth for the whole app: st.session_state.dark_mode
# (a plain boolean, driven by exactly one st.toggle -- see "THEME
# SWITCH" below). Every other former theme mechanism (a separate
# unused "theme" string, an OS-prefers-color-scheme-only CSS block, a
# second sidebar radio control that had no visual effect, and a
# duplicated set of Python color constants) has been removed so
# nothing can fall out of sync with this one flag.

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

THEMES = {
    "light": {
        "bg": "#F4F6FA",
        "bg_secondary": "#EDEFF6",
        "sidebar": "#FFFFFF",
        "card": "#FFFFFF",
        "text": "#172033",
        "text_secondary": "#4B5468",
        "muted": "#6B7086",
        "border": "#E1E6EF",
        "accent": "#5B45E0",
        "accent_hover": "#4A36C9",
        "accent_soft": "#EEEAFE",
        "accent_solid": "#5B45E0",
        "accent_solid_hover": "#4A36C9",
        "success": "#0F7A5C",
        "success_bg": "#E6F7F1",
        "warning": "#92600A",
        "warning_bg": "#FFF4E0",
        "danger": "#B3261E",
        "danger_bg": "#FDECEA",
        "input": "#FFFFFF",
        "sidebar_hover": "#F4F1FF",
        "code_bg": "#F3F4F7",
        "shadow": "0 3px 15px rgba(25,25,60,0.06)",
    },
    "dark": {
        "bg": "#0F1117",
        "bg_secondary": "#171A23",
        "sidebar": "#12141C",
        "card": "#171A23",
        "text": "#F3F4F8",
        "text_secondary": "#C3C8D6",
        "muted": "#9AA0B4",
        "border": "#2C3140",
        "accent": "#9F8FFF",
        "accent_hover": "#B3A6FF",
        "accent_soft": "#241F45",
        "accent_solid": "#5B45E0",
        "accent_solid_hover": "#4A36C9",
        "success": "#4FD1A1",
        "success_bg": "#123328",
        "warning": "#F6C453",
        "warning_bg": "#3A2E12",
        "danger": "#F47777",
        "danger_bg": "#3A1919",
        "input": "#1B1F29",
        "sidebar_hover": "#1D2740",
        "code_bg": "#0F1726",
        "shadow": "0 8px 28px rgba(0,0,0,0.35)",
    },
}

theme = THEMES["dark" if st.session_state.dark_mode else "light"]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MeetSense — Meeting Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME VARIABLES (single CSS injection point)
# ============================================================
# Defines --bg-primary etc. as CSS custom properties from the THEMES
# dict above. Every component rule below (and throughout the file)
# reads these variables instead of hardcoding colors, so a single
# dark_mode flip updates everything at once.

st.markdown(
    f"""
<style>
:root {{
    --bg-primary: {theme['bg']};
    --bg-secondary: {theme['bg_secondary']};
    --bg-card: {theme['card']};
    --bg-sidebar: {theme['sidebar']};
    --sidebar-hover: {theme['sidebar_hover']};
    --text-primary: {theme['text']};
    --text-secondary: {theme['text_secondary']};
    --text-muted: {theme['muted']};
    --border: {theme['border']};
    --accent: {theme['accent']};
    --accent-hover: {theme['accent_hover']};
    --accent-solid: {theme['accent_solid']};
    --accent-solid-hover: {theme['accent_solid_hover']};
    --accent-soft: {theme['accent_soft']};
    --success: {theme['success']};
    --success-bg: {theme['success_bg']};
    --warning: {theme['warning']};
    --warning-bg: {theme['warning_bg']};
    --error: {theme['danger']};
    --error-bg: {theme['danger_bg']};
    --input-bg: {theme['input']};
    --input-text: {theme['text']};
    --code-bg: {theme['code_bg']};
    --shadow: {theme['shadow']};
}}
</style>
""",
    unsafe_allow_html=True,
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

# Experimental diarization backend (see README "Experimental Diarization
# Comparison" section) -- separate output files, never overwriting the
# production PyAnnote files above.
DIARIZATION_VBX = TRANSCRIPT_DIR / "diarization_vbx.json"

NLP_PROCESSED = NLP_DIR / "nlp_processed_transcript.json"
TFIDF = NLP_DIR / "tfidf_keywords.json"
TOPICS = NLP_DIR / "topic_modeling.json"
ACTION_ITEMS = NLP_DIR / "action_items_decisions.json"

SUMMARY = SUMMARY_DIR / "meeting_summary.json"

WER = EVAL_DIR / "wer_results.json"
DER = EVAL_DIR / "der_results.json"
DER_VBX = EVAL_DIR / "der_vbx.json"
ROUGE = EVAL_DIR / "rouge_results.json"
ACTION_EVAL = EVAL_DIR / "action_item_results.json"

MODEL_COMPARISON = EVAL_DIR / "model_comparison.json"
ABLATION_RESULTS = EVAL_DIR / "ablation_results.json"
ERROR_ANALYSIS = EVAL_DIR / "error_analysis.json"
SPEAKER_ANALYSIS = EVAL_DIR / "speaker_analysis.json"
EVIDENCE_LINKED_ITEMS = EVAL_DIR / "evidence_linked_items.json"


# ============================================================
# CUSTOM CSS
# ============================================================
#
# One static stylesheet. Every rule below reads the --bg-primary /
# --text-primary / etc. custom properties defined above (from
# THEMES + st.session_state.dark_mode) instead of hardcoding colors,
# so flipping dark_mode updates every component at once with no
# separate light/dark rule sets to keep in sync.

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--bg-primary);
    color: var(--text-primary);
}

/* Hide Streamlit default UI */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
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

/* ------------------------------------------------------------
   Sidebar
   ------------------------------------------------------------ */

section[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
section[data-testid="stSidebar"] [data-testid="stRadio"] label,
section[data-testid="stSidebar"] [data-testid="stRadio"] label * {
    color: var(--text-primary) !important;
    opacity: 1 !important;
}

/* Nav rows get a hover affordance and a clearly-visible selected state */
section[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 8px;
    padding: 2px 8px;
    transition: background-color .12s ease;
}

section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: var(--sidebar-hover) !important;
}

section[data-testid="stSidebar"] [role="radio"][aria-checked="true"],
section[data-testid="stSidebar"] [role="radio"][aria-checked="true"] * {
    color: var(--accent) !important;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"] [role="radio"][aria-checked="true"] {
    background: var(--accent-soft) !important;
    border-radius: 8px;
}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: var(--text-muted) !important;
    opacity: 1 !important;
}

[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}

[data-testid="stSidebarCollapsedControl"] button {
    color: var(--text-primary) !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    box-shadow: var(--shadow) !important;
}

[data-testid="stSidebarCollapsedControl"] svg {
    color: var(--text-primary) !important;
    fill: var(--text-primary) !important;
}

/* ------------------------------------------------------------
   Brand / top bar / titles
   ------------------------------------------------------------ */

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
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 21px;
    box-shadow: 0 5px 15px rgba(91,75,255,0.25);
    flex-shrink: 0;
}

.brand-name {
    font-size: 20px;
    font-weight: 800;
    color: var(--text-primary) !important;
}

.topbar {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 13px 18px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    box-shadow: var(--shadow);
}

.top-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-secondary);
}

.page-title {
    font-size: 31px;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 4px;
}

.page-subtitle {
    color: var(--text-secondary);
    font-size: 14px;
    margin-bottom: 25px;
}

.section-heading {
    font-size: 18px;
    font-weight: 750;
    color: var(--text-primary);
    margin-top: 12px;
    margin-bottom: 12px;
}

.soft-divider {
    height: 1px;
    background: var(--border);
    margin: 18px 0;
}

/* ------------------------------------------------------------
   Cards
   ------------------------------------------------------------ */

.card,
.meeting-header {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 18px;
    box-shadow: var(--shadow);
}

.card-title,
.meeting-name {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 12px;
}

.card-subtitle,
.meeting-meta {
    font-size: 12px;
    color: var(--text-muted);
}

.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 19px;
    min-height: 120px;
    box-shadow: var(--shadow);
}

.stat-icon { font-size: 20px; margin-bottom: 10px; }

.stat-number {
    font-size: 27px;
    font-weight: 800;
    color: var(--text-primary);
}

.stat-label {
    color: var(--text-muted);
    font-size: 12px;
    margin-top: 3px;
}

/* ------------------------------------------------------------
   Speakers / transcript
   ------------------------------------------------------------ */

.speaker-row {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 13px 15px;
    margin-bottom: 10px;
}

.speaker-name {
    font-size: 14px;
    font-weight: 700;
    color: var(--text-primary);
}

.speaker-time {
    font-size: 12px;
    color: var(--text-muted);
}

.transcript-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 15px 17px;
    margin-bottom: 10px;
    transition: border-color .15s ease, box-shadow .15s ease;
}

.transcript-item:hover {
    border-color: var(--accent);
    box-shadow: var(--shadow);
}

.transcript-header {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
}

.transcript-speaker {
    font-size: 13px;
    font-weight: 700;
    color: var(--accent);
}

.transcript-time {
    font-size: 11px;
    color: var(--text-muted);
}

.transcript-text {
    font-size: 14px;
    line-height: 1.65;
    color: var(--text-primary);
    word-wrap: break-word;
    overflow-wrap: break-word;
}

/* ------------------------------------------------------------
   Tags / status / evaluation metrics
   ------------------------------------------------------------ */

.tag {
    display: inline-block;
    padding: 7px 11px;
    margin: 4px 4px 4px 0;
    background: var(--accent-soft);
    color: var(--accent);
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

.status-pill {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 20px;
    background: var(--success-bg);
    color: var(--success);
    font-size: 11px;
    font-weight: 700;
}

.metric {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 15px;
    text-align: center;
}

.metric-name {
    color: var(--text-muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    font-size: 25px;
    font-weight: 800;
    color: var(--accent);
    margin-top: 5px;
}

/* ------------------------------------------------------------
   Action items / decisions / summary
   ------------------------------------------------------------ */

.action-item {
    padding: 14px;
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 9px;
    background: var(--bg-card);
}

.action-title {
    font-weight: 600;
    font-size: 13px;
    color: var(--text-primary);
}

.decision {
    background: var(--accent-soft);
    border-left: 4px solid var(--accent);
    padding: 14px 16px;
    border-radius: 9px;
    margin-bottom: 10px;
    font-size: 13px;
    line-height: 1.5;
    color: var(--text-primary);
}

.summary-box {
    background: var(--accent-soft);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
    font-size: 14px;
    line-height: 1.7;
    color: var(--text-primary);
}

/* ------------------------------------------------------------
   Native Streamlit widgets
   ------------------------------------------------------------ */

h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; }
p, li, label { color: var(--text-secondary); }

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: var(--text-primary);
}

.stButton > button,
.stDownloadButton > button {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    /* accent-solid (not accent) -- verified >=4.5:1 with white text in
       both themes. --accent itself is deliberately lighter in dark
       mode (for readable text on dark backgrounds), which would fail
       contrast as a button fill with white text on top. */
    background: var(--accent-solid) !important;
    color: #ffffff !important;
    border-color: var(--accent-solid) !important;
}

.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
    background: var(--accent-solid-hover) !important;
    color: #ffffff !important;
}

.stButton > button:disabled,
.stDownloadButton > button:disabled {
    opacity: 0.5 !important;
}

.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: var(--input-bg) !important;
    color: var(--input-text) !important;
    border-color: var(--border) !important;
}

.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stMultiSelect label,
.stNumberInput label {
    color: var(--text-primary) !important;
}

input::placeholder,
textarea::placeholder {
    color: var(--text-muted) !important;
}

[data-baseweb="select"] * {
    color: var(--input-text) !important;
}

[data-testid="stFileUploader"] {
    background: transparent !important;
}

[data-testid="stFileUploader"] * {
    color: var(--text-primary) !important;
}

/* The actual visible drag-and-drop box is a separate inner element,
   not just the outer stFileUploader wrapper above -- it needs its
   own background/border or it keeps Streamlit's own default styling
   regardless of the current theme. */
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-secondary) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: var(--text-primary) !important;
}

/* "200MB per file - WAV, MP3, ..." helper text: readable but
   secondary, matching the rest of the app's metadata style. */
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--text-secondary) !important;
}

/* "Browse files" button inside the dropzone. */
[data-testid="stFileUploaderDropzone"] button {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Uploaded file row (filename, size, remove icon). */
[data-testid="stFileUploaderFile"] {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
}

[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 12px 15px;
}

[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
}

button[data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
}

[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] * {
    color: var(--text-primary) !important;
}

[data-testid="stAlert"] {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

[data-testid="stAlert"] * {
    color: var(--text-primary) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
}

pre, code {
    background: var(--code-bg) !important;
    color: var(--text-primary) !important;
}

/* Focus visibility (accessibility) */
.stButton > button:focus-visible,
[data-baseweb="select"] > div:focus-within,
input:focus-visible,
textarea:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 1px;
}

/* ------------------------------------------------------------
   Responsive
   ------------------------------------------------------------ */

@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .page-title { font-size: 25px; }
    .topbar { flex-direction: column; align-items: flex-start; }
}

</style>
""",
    unsafe_allow_html=True,
)



# ============================================================
# HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def _load_json_cached(path_str, _mtime):
    # _mtime is only part of the cache key (so a re-processed file
    # invalidates the cache automatically); it isn't used directly.
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json(path):
    """Reads and parses a JSON file, cached per (path, modification
    time) so repeated reruns -- e.g. every theme toggle click -- don't
    re-read and re-parse every output file from disk each time. The
    mtime key means a freshly re-processed meeting is picked up
    immediately, not served stale."""

    try:
        if not path.exists():
            return None

        return _load_json_cached(str(path), path.stat().st_mtime)

    except Exception:
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

    # Three nav groups share ONE active page below (see
    # "active_page" right before the page-routing chain). Each
    # group's on_change marks itself as the active group, so
    # switching between GENERAL / INTELLIGENCE / ANALYTICS behaves
    # like a single combined nav instead of three independent ones.
    def _set_active_nav_group(group_name):
        st.session_state.active_nav_group = group_name

    if "active_nav_group" not in st.session_state:
        st.session_state.active_nav_group = "page"

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
        key="nav_page",
        on_change=_set_active_nav_group,
        args=("page",),
    )

    st.markdown("### INTELLIGENCE")

    intelligence_page = st.radio(
        "",
        [
            "Transcript",
            "AI Summary",
            "Action Items",
            "Decisions",
            "Meeting Report",
        ],
        label_visibility="collapsed",
        key="nav_intelligence",
        on_change=_set_active_nav_group,
        args=("intelligence",),
    )

    st.markdown("### ANALYTICS")

    analytics_page = st.radio(
        "",
        [
            "Keywords & Topics",
            "Speaker Analysis",
            "Evaluation",
            "Model Comparison",
            "Ablation Study",
            "Error Analysis",
            "Speakers",
        ],
        label_visibility="collapsed",
        key="nav_analytics",
        on_change=_set_active_nav_group,
        args=("analytics",),
    )

    st.markdown("---")

    st.caption("Speaker-Aware Meeting Intelligence")

    st.markdown(
        """
        <div style="
            background:var(--accent-soft);
            padding:13px;
            border-radius:12px;
            font-size:11px;
            color:var(--accent);
        ">
        <b>MeetSense AI</b><br>
        Whisper + PyAnnote + NLP + Transformers
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# THEME SWITCH
# ============================================================
# The one control that drives st.session_state.dark_mode, the single
# source of truth for every themed element (see "CUSTOM CSS" above).

_, theme_toggle_col = st.columns([8, 2])

with theme_toggle_col:
    st.toggle(
        "🌙 Dark mode",
        key="dark_mode",
        help="Switch between light and dark appearance.",
    )


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
der_vbx_data = load_json(DER_VBX) or {}
rouge_data = load_json(ROUGE) or {}
action_eval_data = load_json(ACTION_EVAL) or {}

model_comparison_data = load_json(MODEL_COMPARISON) or {}
ablation_data = load_json(ABLATION_RESULTS) or {}
error_analysis_data = load_json(ERROR_ANALYSIS) or {}
speaker_analysis_data = load_json(SPEAKER_ANALYSIS) or {}
evidence_linked_data = load_json(EVIDENCE_LINKED_ITEMS) or {}

speaker_stats = get_speaker_statistics(transcript)

duration = get_duration(transcript)

speaker_count = len(speaker_stats)

segment_count = len(transcript)


# ============================================================
# ACTIVE PAGE
# ============================================================
# page / intelligence_page / analytics_page are three separate
# st.radio() widgets that all exist in the sidebar simultaneously.
# active_nav_group (set via each radio's on_change) records which
# group the user actually interacted with, so exactly one page
# renders below instead of only ever the GENERAL group's selection.

_active_group = st.session_state.get("active_nav_group", "page")

if _active_group == "intelligence":
    active_page = intelligence_page
elif _active_group == "analytics":
    active_page = analytics_page
else:
    active_page = page


# ============================================================
# DASHBOARD
# ============================================================

if active_page == "Dashboard":

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
                            background:var(--border);
                            border-radius:10px;
                        ">
                            <div style="
                                width:{min(percentage,100):.1f}%;
                                height:5px;
                                background:var(--accent);
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

elif active_page == "Meeting":

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
                <p style="font-size:14px;line-height:1.7;color:var(--text-secondary);">
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
                <p style="font-size:14px;line-height:1.7;color:var(--text-secondary);">
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

elif active_page == "My Meetings":

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

elif active_page == "Live Meeting":

    st.markdown(
        '<div class="page-title">Live Meeting</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Incremental transcription, diarization, '
        "and summarization while the meeting is still running.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                🔴 Live Meeting Mode
            </div>
            <p style="font-size:14px;line-height:1.7;color:var(--text-secondary);">
                Audio is captured in rolling 20s windows (3s overlap) and run
                through VAD → Whisper → diarization → alignment → NLP →
                summarization as it arrives. Diarization re-clusters a capped,
                rolling buffer rather than each tiny window in isolation, to
                keep speaker labels consistent -- see PROJECT_REPORT for why.
                Processing speed is <b>not guaranteed to be real-time</b> on
                this hardware; the status strip below reports measured lag,
                not an assumption.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # Controller lifecycle: NOT constructed here. Constructing it
    # imports live_controller -> live_pipeline -> vad/transcription/
    # diarization/summarization -> torch/whisper/pyannote/
    # transformers, and loads every model. That only happens inside
    # the Start button handler below, the first time it's clicked --
    # never just from opening this tab. Once constructed, it's
    # cached in st.session_state, so later reruns (every widget
    # interaction) never reconstruct it or reload any model.
    # ------------------------------------------------------------

    from live_capture import list_input_devices  # no torch import -- cheap

    controller = st.session_state.get("live_controller")

    if controller is None and st.session_state.get(
        "live_controller_load_error"
    ):

        st.error(
            "Live-mode models failed to load on the last attempt. "
            "You can try Start again below."
        )

        st.caption(
            f"Details: {st.session_state['live_controller_load_error']}"
        )

    status = (
        controller.get_status()
        if controller is not None
        else {"running": False, "chunk_count": 0}
    )

    # --------------------------------------------------------
    # Controls: source selector + Start/Stop/Reset/Save. Always
    # rendered (even before a controller exists) so the page itself
    # never waits on model loading -- only clicking Start does.
    # --------------------------------------------------------

    source_col, device_col = st.columns([1, 2])

    with source_col:

        source_mode = st.radio(
            "Audio source",
            ["Simulated (replay file)", "Microphone"],
            index=0,
            disabled=status["running"],
            help=(
                "Simulated mode replays a saved recording at 1x "
                "speed through the same live pipeline -- the "
                "reliable fallback if microphone access fails "
                "during a demo."
            ),
        )

    simulated_file_path = None

    with device_col:

        if source_mode == "Simulated (replay file)":

            candidate_files = sorted(
                str(p)
                for p in Path("data/audio").glob("*.wav")
            ) + sorted(
                str(p)
                for p in Path("data/test_audio").glob("*.wav")
            )

            if candidate_files:

                simulated_file_path = st.selectbox(
                    "File to replay",
                    candidate_files,
                    disabled=status["running"],
                )

            else:

                st.warning("No .wav files found under data/.")

        else:

            try:
                mic_devices = list_input_devices()
            except Exception:
                mic_devices = []

            if mic_devices:
                st.caption(
                    "Input devices detected: "
                    + ", ".join(d["name"] for d in mic_devices)
                )
            else:
                st.warning(
                    "No microphone input device was detected. "
                    "Use Simulated mode instead."
                )

    start_col, stop_col, reset_col, save_col = st.columns(4)

    with start_col:

        if st.button(
            "▶ Start",
            disabled=status["running"],
            use_container_width=True,
        ):

            try:

                if controller is None:

                    with st.spinner(
                        "Loading live-mode models (VAD, Whisper, "
                        "diarization, summarizer) -- first time "
                        "this session, can take a while under "
                        "memory pressure..."
                    ):

                        from live_controller import LiveSessionController

                        controller = LiveSessionController()

                        st.session_state.live_controller = controller
                        st.session_state.pop(
                            "live_controller_load_error", None
                        )

                if source_mode == "Microphone":
                    controller.start(mode="microphone")
                else:
                    controller.start(
                        mode="simulated",
                        simulated_file_path=simulated_file_path,
                    )

                st.rerun()

            except Exception as error:

                st.session_state["live_start_error"] = str(error)

                if "live_controller" not in st.session_state:
                    st.session_state["live_controller_load_error"] = str(
                        error
                    )

    with stop_col:

        if st.button(
            "■ Stop",
            disabled=not status["running"],
            use_container_width=True,
        ):

            with st.spinner(
                "Stopping... if a chunk is mid-processing this "
                "waits for it to finish (seen up to ~160s during "
                "testing) -- a running model call cannot be "
                "safely interrupted."
            ):
                controller.stop()

            # Stop finalizes the session into a Meeting Report
            # (source="live"), reusing the session's own accumulated
            # transcript/summaries/topics -- no re-reading batch files.
            if controller.session.aligned_transcript:

                from meeting_report import build_report
                import time as _time

                live_meeting_id = (
                    "live_" + _time.strftime("%Y%m%d_%H%M%S")
                )

                _live_extractor_choice = st.session_state.get(
                    "extractor_choice", "rule_based"
                )

                try:

                    _spinner_ctx = (
                        st.spinner("Generating takeaways...")
                        if _live_extractor_choice == "llm"
                        else contextlib.nullcontext()
                    )

                    with _spinner_ctx:

                        live_report = build_report(
                            live_meeting_id,
                            transcript_segments=list(
                                controller.session.aligned_transcript
                            ),
                            chunk_summaries=list(
                                controller.session.chunk_summaries
                            ),
                            final_summary=controller.session.whole_summary,
                            topics=controller.session.topics_result.get(
                                "topics", []
                            ),
                            source="live",
                            extractor_name=_live_extractor_choice,
                            # Models are already loaded in this live
                            # session; only unload before an LLM call.
                            pre_extract_hook=controller.session.unload_models,
                        )

                    st.session_state["live_report_id"] = live_meeting_id

                    if live_report.extractor_used != _live_extractor_choice:
                        st.session_state["live_report_error"] = (
                            live_report.extractor_stats.get(
                                "fallback_notice",
                                f"Fell back to "
                                f"'{live_report.extractor_used}'.",
                            )
                        )

                except Exception as error:
                    st.session_state["live_report_error"] = str(error)

            st.rerun()

    with reset_col:

        if st.button(
            "⟲ Reset",
            disabled=status["running"] or controller is None,
            use_container_width=True,
        ):

            controller.reset()
            st.rerun()

    with save_col:

        if st.button(
            "💾 Save session",
            disabled=(
                controller is None
                or status["running"]
                or status["chunk_count"] == 0
            ),
            use_container_width=True,
        ):

            saved_paths = controller.save_session()
            st.success(f"Saved to {saved_paths['transcript']} and siblings.")

    if st.session_state.get("live_start_error"):

        st.error(
            "Could not start capture. If this is a microphone "
            "permission or device error, switch to Simulated mode "
            "above and try again."
        )
        st.caption(f"Details: {st.session_state['live_start_error']}")

    if st.session_state.get("live_report_id"):

        st.success(
            f"Meeting Report ready: "
            f"{st.session_state['live_report_id']} "
            f"(see the Meeting Report page)."
        )

    if st.session_state.get("live_report_error"):

        st.warning(
            f"Could not build the live Meeting Report: "
            f"{st.session_state['live_report_error']}"
        )

    st.divider()

    if controller is not None:

        # --------------------------------------------------------
        # Live-updating display. This fragment ONLY reads
        # controller.get_status() and renders -- it never calls
        # VAD/ASR/diarization. Re-runs on its own every ~2s without
        # re-running the whole page script.
        # --------------------------------------------------------

        @st.fragment(run_every="2s")
        def render_live_status():

            live_status = controller.get_status()

            # ---- status strip ----

            status_cols = st.columns(5)

            status_cols[0].metric(
                "Chunks processed", live_status["chunk_count"]
            )

            status_cols[1].metric(
                "Audio captured",
                format_time(live_status["captured_audio_seconds"]),
            )

            status_cols[2].metric(
                "Audio processed",
                format_time(live_status["processed_audio_seconds"]),
            )

            status_cols[3].metric(
                "Dropped chunks", live_status["dropped_chunk_count"]
            )

            last_log = live_status.get("last_chunk_log") or {}

            status_cols[4].metric(
                "Last chunk total",
                f"{last_log.get('total_ms', 0):.0f} ms",
            )

            if live_status["lag_seconds"] > 5:

                st.warning(
                    f"⚠️ LAGGING: processing is behind capture by "
                    f"{live_status['lag_seconds']:.1f}s. This is "
                    f"measured, not estimated."
                )

            elif live_status["running"]:

                st.markdown(
                    '<span class="status-pill">● Live -- keeping up '
                    "(lag "
                    f'{live_status["lag_seconds"]:.1f}s)</span>',
                    unsafe_allow_html=True,
                )

            if live_status["stopping"]:
                st.info("Stopping -- waiting for the current chunk to finish...")

            if live_status["last_error"]:
                st.error(f"Pipeline error on last chunk: {live_status['last_error']}")

            if last_log:

                with st.expander("Last chunk -- per-stage timing (ms)"):

                    st.json(
                        {
                            key: value
                            for key, value in last_log.items()
                            if key.endswith("_ms") or key.endswith("_error")
                        }
                    )

            # ---- rolling summary ----

            st.markdown(
                '<div class="section-heading">📝 Rolling Meeting Summary</div>',
                unsafe_allow_html=True,
            )

            if live_status["whole_summary"]:

                st.markdown(
                    f"""
                    <div class="summary-box">
                        {live_status["whole_summary"]}
                    </div>
                    <div class="card-subtitle">
                        As of chunk {live_status["chunk_count"]} /
                        {format_time(live_status["processed_audio_seconds"])}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.caption(
                    "Whole-meeting summary regenerates every "
                    f"{controller.session.recompute_every_n_chunks} chunks "
                    "-- none yet."
                )

            if live_status["chunk_summaries"]:

                with st.expander(
                    f"Chunk summaries so far "
                    f"({len(live_status['chunk_summaries'])})"
                ):

                    for index, chunk_summary in enumerate(
                        live_status["chunk_summaries"], start=1
                    ):
                        st.markdown(f"**Chunk {index}:** {chunk_summary}")

            # ---- decisions / action items ----

            items_col, decisions_col = st.columns(2)

            with items_col:

                st.markdown(
                    '<div class="section-heading">✅ Action Items</div>',
                    unsafe_allow_html=True,
                )

                if live_status["action_items"]:

                    for item in reversed(live_status["action_items"]):

                        deadline_html = (
                            f'<div class="card-subtitle">Deadline: {item["deadline"]}</div>'
                            if item.get("deadline")
                            else ""
                        )

                        st.markdown(
                            f"""
                            <div class="action-item">
                                <div class="action-title">□ {item['action']}</div>
                                <div class="card-subtitle">{item['speaker']}</div>
                                {deadline_html}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                else:
                    st.caption("None detected yet.")

            with decisions_col:

                st.markdown(
                    '<div class="section-heading">📌 Decisions</div>',
                    unsafe_allow_html=True,
                )

                if live_status["decisions"]:

                    for item in reversed(live_status["decisions"]):

                        st.markdown(
                            f"""
                            <div class="action-item">
                                <div class="action-title">{item['decision']}</div>
                                <div class="card-subtitle">{item['speaker']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                else:
                    st.caption("None detected yet.")

            # ---- keywords / topics ----

            st.markdown(
                '<div class="section-heading">🔑 Top Keywords & Topics</div>',
                unsafe_allow_html=True,
            )

            keywords = live_status["keywords"].get("top_keywords", [])
            topics = live_status["topics"].get("topics", [])

            if keywords or topics:

                keyword_col, topic_col = st.columns(2)

                with keyword_col:

                    if keywords:
                        st.markdown(
                            ", ".join(
                                k["term"] for k in keywords[:15]
                            )
                        )
                    else:
                        st.caption("No keywords yet.")

                with topic_col:

                    if topics:
                        for topic in topics:
                            words = ", ".join(
                                w["word"] for w in topic["top_words"][:6]
                            )
                            st.markdown(f"**Topic {topic['topic_id']}:** {words}")
                    else:
                        st.caption("No topics yet.")

            else:

                st.caption(
                    "Keywords/topics refit every "
                    f"{controller.session.recompute_every_n_chunks} chunks "
                    "over the growing transcript -- none yet."
                )

            # ---- growing transcript, newest first ----

            st.markdown(
                '<div class="section-heading">🗣️ Live Transcript '
                "(newest first)</div>",
                unsafe_allow_html=True,
            )

            transcript = live_status["transcript"]

            if transcript:

                for segment in reversed(transcript[-50:]):

                    st.markdown(
                        f"""
                        <div class="transcript-item">
                            <div class="transcript-header">
                                <span class="transcript-speaker">
                                    🎙️ {segment["speaker"]}
                                </span>
                                <span class="transcript-time">
                                    {format_time(segment["start"])}
                                </span>
                                <span class="transcript-time">
                                    {"speakers refined" if segment.get("speakers_refined") else "provisional speaker"}
                                </span>
                            </div>
                            <div class="transcript-text">
                                {segment["text"]}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            else:

                st.caption("No speech processed yet.")

        render_live_status()


# ============================================================
# TRANSCRIPT PAGE
# ============================================================

elif active_page == "Transcript":

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

    # Built as one combined HTML string and rendered with a single
    # st.markdown() call -- up to ~400 segments, so one call instead
    # of one per segment avoids per-element Streamlit overhead on
    # every rerun (including every theme toggle).
    segment_blocks = []

    for item in filtered:

        speaker = get_speaker_name(item)
        text = get_segment_text(item)

        segment_blocks.append(
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
            """
        )

    if segment_blocks:

        st.markdown(
            "".join(segment_blocks),
            unsafe_allow_html=True,
        )


# ============================================================
# SUMMARY PAGE
# ============================================================

elif active_page == "AI Summary":

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

elif active_page == "Action Items":

    st.markdown(
        '<div class="page-title">Action Items</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Tasks and follow-up actions extracted from the meeting.</div>',
        unsafe_allow_html=True,
    )

    evidence_action_items = evidence_linked_data.get("action_items", [])

    if evidence_action_items:

        # Evidence-linked rendering: every item traces back to its
        # source transcript segment (see src/evidence_linking.py).
        for item in evidence_action_items:

            deadline_html = (
                f'<div class="card-subtitle" style="margin-top:4px;">Deadline: {item["deadline"]}</div>'
                if item.get("deadline") else ""
            )

            st.markdown(
                f"""
                <div class="action-item">
                    <div class="action-title">
                        □ &nbsp; {item['text']}
                    </div>
                    <div class="card-subtitle" style="margin-top:6px;">
                        Speaker: {item['speaker_display_name']} &nbsp;·&nbsp;
                        Timestamp: {item['timestamp_formatted']}
                    </div>
                    {deadline_html}
                    <div style="font-size:12px;color:var(--text-muted);margin-top:8px;">
                        <b>Evidence:</b> "{item['evidence_text']}"
                    </div>
                    <div style="font-size:11px;color:var(--text-muted);margin-top:6px;">
                        Detection method: {item['detection_method_label']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:

        # Fallback: evidence_linked_items.json hasn't been generated
        # yet (run src/evidence_linking.py). Original rendering,
        # unchanged, so this page keeps working either way.
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

elif active_page == "Decisions":

    st.markdown(
        '<div class="page-title">Meeting Decisions</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Important decisions identified during the meeting.</div>',
        unsafe_allow_html=True,
    )

    evidence_decisions = evidence_linked_data.get("decisions", [])

    if evidence_decisions:

        # Evidence-linked rendering: every decision traces back to
        # its source transcript segment (see src/evidence_linking.py).
        for item in evidence_decisions:

            st.markdown(
                f"""
                <div class="decision">
                    💡 <b>Decision</b><br>
                    {item['text']}
                    <div class="card-subtitle" style="margin-top:6px;">
                        Speaker: {item['speaker_display_name']} &nbsp;·&nbsp;
                        Timestamp: {item['timestamp_formatted']}
                    </div>
                    <div style="font-size:12px;color:var(--text-muted);margin-top:8px;">
                        <b>Evidence:</b> "{item['evidence_text']}"
                    </div>
                    <div style="font-size:11px;color:var(--text-muted);margin-top:6px;">
                        Detection method: {item['detection_method_label']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:

        # Fallback: evidence_linked_items.json hasn't been generated
        # yet (run src/evidence_linking.py). Original rendering,
        # unchanged, so this page keeps working either way.
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
# MEETING REPORT
# ============================================================

elif active_page == "Meeting Report":

    st.markdown(
        '<div class="page-title">Meeting Report</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-subtitle">Speakers, takeaways, and '
        "evidence-linked action items.</div>",
        unsafe_allow_html=True,
    )

    from meeting_report import (
        build_report,
        load_report,
        save_report,
        list_report_ids,
        apply_speaker_renaming,
        render_markdown,
        render_html,
        available_extractors,
    )

    default_report_id = "meeting_001"

    existing_report_ids = list_report_ids()

    report_options = existing_report_ids or [default_report_id]

    build_col, extractor_col = st.columns([2, 1])

    with build_col:
        report_id = st.selectbox("Report", report_options, index=0)

    with extractor_col:
        # key="extractor_choice" persists this selection in
        # session_state so the Live Meeting Stop handler (a
        # different page) can reuse the same preference for
        # finalizing a live report with "the selected extractor".
        extractor_choice = st.selectbox(
            "Extractor",
            available_extractors(),
            index=0,
            key="extractor_choice",
        )

    if st.button("🔄 Build / rebuild from current batch outputs"):

        _live_controller = st.session_state.get("live_controller")

        _unload_hook = (
            _live_controller.session.unload_models
            if _live_controller is not None
            else None
        )

        try:

            if extractor_choice == "llm":
                with st.spinner("Generating takeaways..."):
                    built_report = build_report(
                        report_id,
                        source="recorded",
                        extractor_name=extractor_choice,
                        pre_extract_hook=_unload_hook,
                    )
            else:
                built_report = build_report(
                    report_id,
                    source="recorded",
                    extractor_name=extractor_choice,
                )

            if built_report.extractor_used != extractor_choice:
                st.warning(
                    built_report.extractor_stats.get(
                        "fallback_notice",
                        f"Fell back to '{built_report.extractor_used}'.",
                    )
                )
            else:
                st.success(
                    f"Built report '{report_id}' using '{extractor_choice}'."
                )

            st.rerun()

        except Exception as error:
            st.error(f"Could not build report: {error}")

    try:
        meeting_report = load_report(report_id)
    except FileNotFoundError:
        meeting_report = None

    if meeting_report is None:

        st.info(
            "No report yet for this meeting. Click "
            "'Build / rebuild' above."
        )

    else:

        st.markdown(
            f"**Date:** {meeting_report.date} &nbsp;&nbsp; "
            f"**Duration:** {format_time(meeting_report.duration_sec)} "
            f"&nbsp;&nbsp; **Source:** {meeting_report.source}"
        )

        if meeting_report.summary:

            st.markdown(
                '<div class="section-heading">📝 Summary</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="summary-box">{meeting_report.summary}</div>',
                unsafe_allow_html=True,
            )

        if meeting_report.speakers:

            st.markdown(
                '<div class="section-heading">🗣️ Speakers & '
                "Speaking Time</div>",
                unsafe_allow_html=True,
            )

            import pandas as pd

            chart_df = pd.DataFrame(
                {
                    "speaker": [s.name for s in meeting_report.speakers],
                    "seconds": [
                        s.speaking_time_sec
                        for s in meeting_report.speakers
                    ],
                }
            ).set_index("speaker")

            st.bar_chart(chart_df)

            with st.expander("Rename speakers"):

                renaming = {}

                for speaker in meeting_report.speakers:

                    new_name = st.text_input(
                        f"Rename {speaker.name}",
                        value=speaker.name,
                        key=f"rename_{report_id}_{speaker.name}",
                    )

                    if new_name != speaker.name:
                        renaming[speaker.name] = new_name

                if renaming and st.button("Apply renaming"):

                    meeting_report = apply_speaker_renaming(
                        meeting_report, renaming
                    )
                    save_report(meeting_report)

                    st.success(
                        "Renamed. Applied to speakers, transcript, "
                        "and all evidence."
                    )
                    st.rerun()

        for section_name, icon, section in [
            ("Decisions", "📌", meeting_report.decisions),
            ("Action Items", "✅", meeting_report.action_items),
            ("Deadlines", "⏰", meeting_report.deadlines),
            ("Open Questions", "❓", meeting_report.open_questions),
        ]:

            if section:

                st.markdown(
                    f'<div class="section-heading">{icon} '
                    f"{section_name}</div>",
                    unsafe_allow_html=True,
                )

                for item in section:

                    due = (
                        f" · due {item.due_date}"
                        if item.due_date
                        else ""
                    )

                    st.markdown(f"**{item.text}**{due}")

                    with st.expander(
                        f"Evidence — {item.evidence.speaker} @ "
                        f"{format_time(item.evidence.timestamp)}"
                    ):
                        st.markdown(f"> {item.evidence.quote}")

        st.markdown(
            '<div class="section-heading">⬇️ Export</div>',
            unsafe_allow_html=True,
        )

        export_col1, export_col2, export_col3 = st.columns(3)

        with export_col1:

            st.download_button(
                "JSON",
                data=json.dumps(
                    meeting_report.to_rendered_dict(),
                    indent=2,
                    ensure_ascii=False,
                ),
                file_name=f"{report_id}_report.json",
                mime="application/json",
            )

        with export_col2:

            st.download_button(
                "Markdown",
                data=render_markdown(meeting_report),
                file_name=f"{report_id}_report.md",
                mime="text/markdown",
            )

        with export_col3:

            st.download_button(
                "HTML",
                data=render_html(meeting_report),
                file_name=f"{report_id}_report.html",
                mime="text/html",
            )


# ============================================================
# KEYWORDS + TOPICS
# ============================================================

elif active_page == "Keywords & Topics":

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
# SPEAKER ANALYSIS
# ============================================================

elif active_page == "Speaker Analysis":

    st.markdown(
        '<div class="page-title">Speaker Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Objective, measurable per-speaker '
        'contribution -- speaking time, topics, keywords, questions, '
        'decisions and action items.</div>',
        unsafe_allow_html=True,
    )

    if not speaker_analysis_data:

        st.info(
            "No speaker analysis data available. Run "
            "src/speaker_analysis.py to generate it."
        )

    else:

        st.caption(
            f"⚠️ {speaker_analysis_data.get('objective_measurement_only', '')}"
        )

        for speaker in speaker_analysis_data.get("speakers", []):

            topic_labels = []

            for t in speaker.get("topics", []):

                label = ", ".join(t["top_words"]) or f"Topic {t['topic_id']}"
                topic_labels.append(label)

            topics_html = "".join(
                f'<span class="tag">{label}</span>' for label in topic_labels
            ) or "<span class=\"card-subtitle\">No dominant topics.</span>"

            keywords_html = "".join(
                f'<span class="tag">{kw.get("term")}</span>'
                for kw in speaker.get("keywords", [])
            ) or "<span class=\"card-subtitle\">No keywords.</span>"

            st.markdown(
                f"""
                <div class="card">
                    <div style="display:flex;justify-content:space-between;align-items:baseline;">
                        <div class="card-title">👤 {speaker['display_name']}</div>
                        <div style="font-size:20px;font-weight:800;color:var(--accent);">
                            {speaker['participation_percentage']:.1f}%
                        </div>
                    </div>

                    <div style="
                        margin-top:10px;
                        height:8px;
                        background:var(--border);
                        border-radius:10px;
                    ">
                        <div style="
                            width:{min(speaker['participation_percentage'], 100):.1f}%;
                            height:8px;
                            background:var(--accent);
                            border-radius:10px;
                        "></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4, m5 = st.columns(5)

            with m1:
                st.markdown(
                    f"""<div class="metric">
                        <div class="metric-name">Speaking time</div>
                        <div class="metric-value">{speaker['speaking_time_formatted']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            with m2:
                st.markdown(
                    f"""<div class="metric">
                        <div class="metric-name">Utterances</div>
                        <div class="metric-value">{speaker['utterance_count']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            with m3:
                st.markdown(
                    f"""<div class="metric">
                        <div class="metric-name">Questions</div>
                        <div class="metric-value">{speaker['question_count']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            with m4:
                st.markdown(
                    f"""<div class="metric">
                        <div class="metric-name">Decisions</div>
                        <div class="metric-value">{speaker['decision_count']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            with m5:
                st.markdown(
                    f"""<div class="metric">
                        <div class="metric-name">Action items</div>
                        <div class="metric-value">{speaker['action_item_count']}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="card-subtitle" style="margin-top:10px;">Topics</div>{topics_html}',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="card-subtitle" style="margin-top:10px;">Keywords</div>{keywords_html}',
                unsafe_allow_html=True,
            )

            contributions = speaker.get("notable_contributions", [])

            if contributions:

                st.markdown(
                    '<div class="card-subtitle" style="margin-top:10px;">'
                    'Notable contributions (own extracted decisions/action items)</div>',
                    unsafe_allow_html=True,
                )

                for item in contributions:

                    label = "💡 Decision" if item["type"] == "decision" else "□ Action item"

                    st.markdown(
                        f'<div style="font-size:13px;margin-top:4px;">{label}: {item["text"]}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        if speaker_analysis_data.get("methodology"):

            with st.expander("Methodology"):

                for key, value in speaker_analysis_data["methodology"].items():
                    st.markdown(f"**{key.replace('_', ' ').title()}**: {value}")


# ============================================================
# EVALUATION
# ============================================================

elif active_page == "Evaluation":

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

    # Diarization Method selector -- PyAnnote is the production backend
    # and is always available; VBx (experimental) only appears once its
    # own DER results exist (i.e. after running
    # `python run_diarization_experiment.py --method vbx`), so the
    # toggle never offers a method with nothing to show.
    diarization_method = "PyAnnote (production)"

    if der_vbx_data:

        diarization_method = st.selectbox(
            "Diarization Method",
            ["PyAnnote (production)", "VBx (experimental)"],
            key="diarization_method_selector",
        )

    if diarization_method.startswith("VBx"):

        der_value = safe_float(der_vbx_data.get("der", 0))

        st.caption(
            "VBx is an experimental, research-comparison-only "
            "diarization backend. It received different upstream "
            "segmentation input than PyAnnote (externally supplied "
            "VAD vs. PyAnnote's own internal segmentation), so this "
            "DER is not a like-for-like clustering-quality comparison "
            "-- see README.md for details."
        )

    else:

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
# MODEL COMPARISON
# ============================================================

elif active_page == "Model Comparison":

    st.markdown(
        '<div class="page-title">Model Comparison</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Comparing installed model variants on this recording.</div>',
        unsafe_allow_html=True,
    )

    if not model_comparison_data:

        st.info(
            "No model comparison data available. Run "
            "src/model_comparison.py to generate it."
        )

    else:

        for note in model_comparison_data.get("limitations", []):
            st.caption(f"⚠️ {note}")

        st.markdown(
            '<div class="section-heading">🎙️ ASR: Whisper base vs small</div>',
            unsafe_allow_html=True,
        )

        asr = model_comparison_data.get("asr_comparison", {})
        asr_models = asr.get("models", {})

        if asr_models:

            asr_cols = st.columns(len(asr_models))

            for col, (model_key, model_info) in zip(asr_cols, asr_models.items()):

                with col:

                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="card-title">{model_key}</div>
                            <div class="metric">
                                <div class="metric-name">WER</div>
                                <div class="metric-value">
                                    {safe_float(model_info.get("wer_percentage")):.2f}%
                                </div>
                            </div>
                            <div class="metric">
                                <div class="metric-name">Transcribe time</div>
                                <div class="metric-value">
                                    {safe_float(model_info.get("transcribe_seconds")):.1f}s
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            cross_check = asr.get("cross_check_against_wer_results_json", {})

            if cross_check.get("available"):

                st.caption(
                    "Cross-check vs the standalone WER evaluation: "
                    f"{safe_float(cross_check.get('stored_wer_percentage')):.2f}% stored, "
                    f"{safe_float(cross_check.get('fresh_base_wer_percentage')):.2f}% "
                    "freshly re-run base model."
                )

        else:

            st.info("No ASR comparison data available.")

        st.markdown(
            '<div class="section-heading">📝 Summarization: T5-small vs TF-IDF extractive</div>',
            unsafe_allow_html=True,
        )

        summarization_comparison = model_comparison_data.get(
            "summarization_comparison", {}
        )
        sum_models = summarization_comparison.get("models", {})

        if sum_models:

            sum_cols = st.columns(len(sum_models))

            for col, (model_key, model_info) in zip(sum_cols, sum_models.items()):

                with col:

                    rouge1 = model_info.get("rouge", {}).get("rouge1", {})

                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="card-title">{model_key}</div>
                            <div class="metric">
                                <div class="metric-name">ROUGE-1 F1</div>
                                <div class="metric-value">
                                    {safe_float(rouge1.get("fmeasure")) * 100:.2f}%
                                </div>
                            </div>
                            <div class="metric">
                                <div class="metric-name">Summary length</div>
                                <div class="metric-value">
                                    {model_info.get("summary_word_count", "—")} words
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if summarization_comparison.get("note"):
                st.caption(summarization_comparison["note"])

        else:

            st.info("No summarization comparison data available.")

        st.markdown(
            '<div class="section-heading">🗣️ Diarization</div>',
            unsafe_allow_html=True,
        )

        diarization_comparison = model_comparison_data.get(
            "diarization_comparison", {}
        )

        st.info(
            diarization_comparison.get(
                "reason",
                "No diarization model comparison available.",
            )
        )


# ============================================================
# ABLATION STUDY
# ============================================================

elif active_page == "Ablation Study":

    st.markdown(
        '<div class="page-title">Ablation Study</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Whisper only → + VAD → + diarization → full pipeline.</div>',
        unsafe_allow_html=True,
    )

    if not ablation_data:

        st.info(
            "No ablation data available. Run src/ablation_study.py "
            "to generate it."
        )

    else:

        for note in ablation_data.get("limitations", []):
            st.caption(f"⚠️ {note}")

        stage_labels = {
            "whisper_only": "1 · Whisper only",
            "whisper_plus_vad": "2 · + VAD",
            "whisper_vad_diarization": "3 · + Diarization",
            "full_pipeline": "4 · Full pipeline",
        }

        metric_specs = [
            ("WER", "wer_percentage", "%"),
            ("DER", "der_percentage", "%"),
            ("Action F1", "action_item_f1", ""),
            ("ROUGE-1 F1", "rouge1_f1", ""),
        ]

        for stage in ablation_data.get("stages", []):

            stage_key = stage.get("stage", "")
            label = stage_labels.get(stage_key, stage_key)

            st.markdown(
                f'<div class="section-heading">{label}</div>',
                unsafe_allow_html=True,
            )

            st.caption(stage.get("description", ""))

            metrics = stage.get("metrics", {})
            not_applicable = stage.get("not_applicable", {})

            metric_cols = st.columns(len(metric_specs))

            for col, (name, key, suffix) in zip(metric_cols, metric_specs):

                with col:

                    value = metrics.get(key)

                    if value is None:
                        display = "—"
                    elif suffix == "%":
                        display = f"{safe_float(value):.2f}%"
                    else:
                        display = f"{safe_float(value):.4f}"

                    st.markdown(
                        f"""
                        <div class="metric">
                            <div class="metric-name">{name}</div>
                            <div class="metric-value">{display}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if not_applicable:

                reasons = " · ".join(
                    f"{k}: {v}" for k, v in not_applicable.items()
                )

                st.caption(f"Not applicable — {reasons}")

            if stage.get("note"):
                st.caption(stage["note"])


# ============================================================
# ERROR ANALYSIS
# ============================================================

elif active_page == "Error Analysis":

    st.markdown(
        '<div class="page-title">Error Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">Structured error findings across ASR, diarization, NLP, action items and summarization.</div>',
        unsafe_allow_html=True,
    )

    if not error_analysis_data:

        st.info(
            "No error analysis data available. Run "
            "src/error_analysis.py to generate it."
        )

    else:

        for note in error_analysis_data.get("limitations", []):
            st.caption(f"⚠️ {note}")

        # ---------------- ASR ----------------

        asr_errors = error_analysis_data.get("asr", {})

        with st.expander("🎙️ ASR errors", expanded=True):

            word_errors = asr_errors.get("word_level_errors", {})
            wer_info = asr_errors.get("word_error_rate", {})

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Substitutions</div>
                        <div class="metric-value">{word_errors.get("substitutions", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Insertions</div>
                        <div class="metric-value">{word_errors.get("insertions", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Deletions (missed)</div>
                        <div class="metric-value">{word_errors.get("deletions", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                "Punctuation-attributable WER delta: "
                f"{safe_float(wer_info.get('punctuation_attributable_delta_points')):.2f} "
                "percentage points"
            )

            sub_examples = word_errors.get("substitution_examples", [])

            if sub_examples:

                st.markdown("**Substitution examples**")

                for ex in sub_examples[:8]:
                    st.markdown(
                        f'<span class="tag">{ex.get("reference")} → {ex.get("hypothesis")}</span>',
                        unsafe_allow_html=True,
                    )

            filler_counts = asr_errors.get("filler_word_usage", {}).get("counts", {})

            if filler_counts:

                st.markdown("**Filler word usage (reference vs hypothesis)**")

                for term, counts in filler_counts.items():
                    st.caption(
                        f"'{term}' — reference: {counts.get('reference')}, "
                        f"hypothesis: {counts.get('hypothesis')}"
                    )

        # ---------------- Diarization ----------------

        diarization_errors = error_analysis_data.get("diarization", {})

        with st.expander("🗣️ Diarization errors"):

            confusion = diarization_errors.get("speaker_confusion", {})
            overlap = diarization_errors.get("overlapping_speech", {})
            boundary = diarization_errors.get("boundary_confidence", {})

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Confusion (s)</div>
                        <div class="metric-value">{safe_float(confusion.get("confusion_seconds")):.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Overlapping turn pairs</div>
                        <div class="metric-value">{overlap.get("overlapping_turn_pairs_found", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Low-confidence boundaries</div>
                        <div class="metric-value">{safe_float(boundary.get("segments_below_50pct_overlap_percentage")):.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if boundary.get("methodology"):
                st.caption(boundary["methodology"])

        # ---------------- NLP ----------------

        nlp_errors = error_analysis_data.get("nlp", {})

        with st.expander("🧠 NLP errors"):

            ner = nlp_errors.get("ner_false_positives", {})
            keyword_noise = nlp_errors.get("keyword_noise", {})
            topic_quality = nlp_errors.get("topic_quality", {})

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">NER false positives</div>
                        <div class="metric-value">{ner.get("flagged_as_likely_false_positive", "—")}/{ner.get("total_entities", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Keyword noise</div>
                        <div class="metric-value">{keyword_noise.get("flagged_as_discourse_noise", "—")}/{keyword_noise.get("top_keywords_checked", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Avg topic overlap</div>
                        <div class="metric-value">{safe_float(topic_quality.get("average_pairwise_jaccard_similarity")):.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if keyword_noise.get("root_cause"):
                st.caption(keyword_noise["root_cause"])

            flagged_keywords = keyword_noise.get("examples", [])

            if flagged_keywords:

                st.markdown("**Flagged keywords (discourse noise)**")

                for kw in flagged_keywords:
                    st.markdown(
                        f'<span class="tag">{kw.get("term")} ({safe_float(kw.get("score")):.1f})</span>',
                        unsafe_allow_html=True,
                    )

        # ---------------- Action items ----------------

        action_item_errors = error_analysis_data.get("action_items", {})

        with st.expander("✅ Action item errors"):

            unverified = action_item_errors.get("unverified_extractions", {})

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Total extracted</div>
                        <div class="metric-value">{unverified.get("total_extracted_full_meeting", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Verified vs reference</div>
                        <div class="metric-value">{unverified.get("evaluated_against_reference", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Unverified</div>
                        <div class="metric-value">{unverified.get("unverified_count", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if unverified.get("note"):
                st.caption(unverified["note"])

            coverage_gap_examples = action_item_errors.get(
                "missed_items_coverage_gap", {}
            ).get("false_negative_examples", [])

            if coverage_gap_examples:

                st.markdown("**Missed reference items**")

                for ex in coverage_gap_examples:

                    cause = ex.get("likely_cause", "").replace("_", " ").title()

                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="card-title">{cause}</div>
                            {ex.get("missed_reference_item", "")}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ---------------- Summarization ----------------

        summarization_errors = error_analysis_data.get("summarization", {})

        with st.expander("📝 Summarization errors"):

            length_info = summarization_errors.get("length", {})
            missing_info = summarization_errors.get("missing_info", {})
            unsupported_info = summarization_errors.get(
                "possible_unsupported_details", {}
            )

            c1, c2 = st.columns(2)

            with c1:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Compression ratio</div>
                        <div class="metric-value">{length_info.get("compression_ratio", "—")}:1</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c2:
                st.markdown(
                    f"""
                    <div class="metric">
                        <div class="metric-name">Summary words</div>
                        <div class="metric-value">{length_info.get("summary_word_count", "—")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            missing_facts = missing_info.get("missing_facts", [])

            if missing_facts:

                st.markdown(
                    "**Missing facts** (present in the manually-verified "
                    "meeting opening, absent from the generated summary)"
                )

                for fact in missing_facts:
                    st.markdown(
                        f'<span class="tag">{fact.replace("_", " ")}</span>',
                        unsafe_allow_html=True,
                    )

            unsupported_tokens = unsupported_info.get("candidate_tokens", [])

            if unsupported_tokens:

                st.markdown("**Possible unsupported details**")

                for token in unsupported_tokens:
                    st.markdown(
                        f'<span class="tag">{token}</span>',
                        unsafe_allow_html=True,
                    )

            else:

                st.caption(
                    "No unsupported (non-source) proper nouns/numbers "
                    "detected in the summary."
                )


# ============================================================
# SPEAKERS
# ============================================================

elif active_page == "Speakers":

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
                            color:var(--accent);
                        ">
                            {percentage:.1f}%
                        </div>
                    </div>

                    <div style="
                        margin-top:15px;
                        height:8px;
                        background:var(--border);
                        border-radius:10px;
                    ">
                        <div style="
                            width:{min(percentage,100):.1f}%;
                            height:8px;
                            background:var(--accent);
                            border-radius:10px;
                        "></div>
                    </div>

                    <div style="
                        margin-top:8px;
                        font-size:12px;
                        color:var(--text-muted);
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

# Always-visible section -- not a collapsible st.expander, so the
# four export actions never require a click to reveal.
st.markdown(
    '<div class="section-heading">📥 Export Meeting Results</div>',
    unsafe_allow_html=True,
)

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
        color:var(--text-muted);
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
