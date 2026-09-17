import json
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st


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