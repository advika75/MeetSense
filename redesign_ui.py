from pathlib import Path

APP = Path("app.py")
BACKUP = Path("app_before_ui_redesign.py")

if not APP.exists():
    raise SystemExit("ERROR: app.py was not found.")

if not BACKUP.exists():
    BACKUP.write_text(APP.read_text())

text = APP.read_text()

# ------------------------------------------------------------
# 1. Add session-state landing page control
# ------------------------------------------------------------

marker = "st.set_page_config("

if "MEETSENSE_UI_VERSION" not in text:
    insert_after = "import streamlit as st\n"

    addition = r'''

# ============================================================
# MEETSENSE UI STATE
# ============================================================

MEETSENSE_UI_VERSION = "2.0"

if "meetsense_started" not in st.session_state:
    st.session_state.meetsense_started = False

if "meetsense_theme" not in st.session_state:
    st.session_state.meetsense_theme = "light"

'''
    text = text.replace(insert_after, insert_after + addition, 1)

# ------------------------------------------------------------
# 2. Change page configuration
# ------------------------------------------------------------

old_config = '''st.set_page_config(
    page_title="MeetSense",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)'''

new_config = '''st.set_page_config(
    page_title="MeetSense — Meeting Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)'''

text = text.replace(old_config, new_config, 1)

# ------------------------------------------------------------
# 3. Add landing-page CSS immediately before existing CSS
# ------------------------------------------------------------

css_marker = '<style>'

landing_css = r'''
<style>

/* ==========================================================
   MEETSENSE LANDING + THEME SYSTEM
   ========================================================== */

:root {
    --ms-bg: #f5f6f8;
    --ms-surface: #ffffff;
    --ms-surface-2: #f0f1f4;
    --ms-text: #171923;
    --ms-text-secondary: #626979;
    --ms-border: #e1e3e8;
    --ms-accent: #635bff;
    --ms-accent-soft: #efedff;
}

.ms-dark {
    --ms-bg: #17181c;
    --ms-surface: #202228;
    --ms-surface-2: #292c33;
    --ms-text: #f5f6f8;
    --ms-text-secondary: #aeb3bf;
    --ms-border: #383b44;
    --ms-accent: #8b7cff;
    --ms-accent-soft: #302d4b;
}

.stApp {
    background: var(--ms-bg);
}

.ms-landing {
    min-height: 88vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 45px;
}

.ms-landing-shell {
    width: min(1200px, 96vw);
    min-height: 680px;
    display: grid;
    grid-template-columns: 1.05fr 0.95fr;
    background: var(--ms-surface);
    border: 1px solid var(--ms-border);
    border-radius: 28px;
    overflow: hidden;
    box-shadow: 0 25px 80px rgba(0,0,0,0.08);
}

.ms-landing-visual {
    background:
        radial-gradient(circle at 30% 30%, rgba(99,91,255,.13), transparent 35%),
        linear-gradient(145deg, var(--ms-surface-2), var(--ms-bg));
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 55px;
}

.ms-visual-card {
    width: 100%;
    max-width: 480px;
    aspect-ratio: 1 / 1;
    border-radius: 24px;
    border: 1px solid var(--ms-border);
    background: var(--ms-surface);
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
}

.ms-visual-circle {
    width: 250px;
    height: 250px;
    border-radius: 50%;
    background: var(--ms-accent-soft);
    position: absolute;
}

.ms-visual-icon {
    position: relative;
    z-index: 2;
    font-size: 115px;
}

.ms-wave {
    position: absolute;
    bottom: 70px;
    left: 15%;
    right: 15%;
    height: 55px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 5px;
}

.ms-wave span {
    width: 4px;
    border-radius: 10px;
    background: var(--ms-accent);
    opacity: .75;
}

.ms-wave span:nth-child(1) { height: 15px; }
.ms-wave span:nth-child(2) { height: 28px; }
.ms-wave span:nth-child(3) { height: 42px; }
.ms-wave span:nth-child(4) { height: 24px; }
.ms-wave span:nth-child(5) { height: 55px; }
.ms-wave span:nth-child(6) { height: 35px; }
.ms-wave span:nth-child(7) { height: 48px; }
.ms-wave span:nth-child(8) { height: 22px; }
.ms-wave span:nth-child(9) { height: 38px; }
.ms-wave span:nth-child(10) { height: 17px; }

.ms-landing-content {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 70px;
    color: var(--ms-text);
}

.ms-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 70px;
}

.ms-brand-logo {
    width: 44px;
    height: 44px;
    border-radius: 13px;
    background: var(--ms-accent);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 23px;
    font-weight: 800;
}

.ms-brand-name {
    font-size: 21px;
    font-weight: 800;
}

.ms-eyebrow {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .14em;
    color: var(--ms-accent);
    margin-bottom: 18px;
}

.ms-landing-title {
    font-size: clamp(42px, 5vw, 68px);
    line-height: 1.02;
    letter-spacing: -0.045em;
    font-weight: 800;
    margin-bottom: 25px;
    color: var(--ms-text);
}

.ms-landing-subtitle {
    font-size: 17px;
    line-height: 1.7;
    color: var(--ms-text-secondary);
    max-width: 500px;
    margin-bottom: 38px;
}

.ms-theme-label {
    color: var(--ms-text-secondary);
    font-size: 13px;
    margin-bottom: 15px;
}

.ms-feature-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 38px;
}

.ms-feature {
    border: 1px solid var(--ms-border);
    background: var(--ms-surface-2);
    border-radius: 999px;
    padding: 9px 13px;
    font-size: 12px;
    color: var(--ms-text-secondary);
}

@media (max-width: 900px) {
    .ms-landing-shell {
        grid-template-columns: 1fr;
    }

    .ms-landing-visual {
        min-height: 360px;
    }

    .ms-landing-content {
        padding: 40px;
    }

    .ms-brand {
        margin-bottom: 40px;
    }
}

/* Force readable Streamlit sidebar text */

section[data-testid="stSidebar"] {
    background: var(--ms-surface) !important;
}

section[data-testid="stSidebar"] * {
    color: var(--ms-text) !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: var(--ms-text) !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--ms-text) !important;
}

</style>
'''

if "MEETSENSE UI STATE" in text and ".ms-landing-shell" not in text:
    text = text.replace(css_marker, css_marker + landing_css, 1)

# ------------------------------------------------------------
# 4. Insert landing page before sidebar
# ------------------------------------------------------------

sidebar_marker = "# ============================================================\n# SIDEBAR\n# ============================================================"

if sidebar_marker not in text:
    raise SystemExit("ERROR: Sidebar marker not found. No changes made.")

if "ms-landing-shell" not in text:
    landing_block = r'''

# ============================================================
# LANDING PAGE
# ============================================================

if not st.session_state.meetsense_started:

    theme = st.session_state.meetsense_theme

    if theme == "dark":
        st.markdown(
            '<div class="ms-dark">',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="ms-landing">

            <div class="ms-landing-shell">

                <div class="ms-landing-visual">

                    <div class="ms-visual-card">

                        <div class="ms-visual-circle"></div>

                        <div class="ms-visual-icon">
                            🎙️
                        </div>

                        <div class="ms-wave">
                            <span></span>
                            <span></span>
                            <span></span>
                            <span></span>
                            <span></span>
                            <span></span>
                            <span></span>
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>

                    </div>

                </div>

                <div class="ms-landing-content">

                    <div class="ms-brand">
                        <div class="ms-brand-logo">∞</div>
                        <div class="ms-brand-name">MeetSense</div>
                    </div>

                    <div class="ms-eyebrow">
                        SPEAKER-AWARE MEETING INTELLIGENCE
                    </div>

                    <div class="ms-landing-title">
                        Meeting intelligence,<br>
                        simplified.
                    </div>

                    <div class="ms-landing-subtitle">
                        Turn meeting recordings into speaker-aware
                        transcripts, concise summaries, topics,
                        decisions and actionable tasks.
                    </div>

                    <div class="ms-feature-row">
                        <div class="ms-feature">Whisper ASR</div>
                        <div class="ms-feature">Speaker Diarization</div>
                        <div class="ms-feature">NLP Analysis</div>
                        <div class="ms-feature">AI Summary</div>
                    </div>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # Theme controls
    theme_col1, theme_col2, theme_col3 = st.columns([2, 1, 1])

    with theme_col1:
        st.markdown(
            '<div class="ms-theme-label">Appearance</div>',
            unsafe_allow_html=True,
        )

    with theme_col2:
        if st.button(
            "☀ Light",
            use_container_width=True,
            type="secondary" if theme != "light" else "primary",
        ):
            st.session_state.meetsense_theme = "light"
            st.rerun()

    with theme_col3:
        if st.button(
            "🌙 Dark",
            use_container_width=True,
            type="secondary" if theme != "dark" else "primary",
        ):
            st.session_state.meetsense_theme = "dark"
            st.rerun()

    st.write("")

    start_col1, start_col2, start_col3 = st.columns([1, 1, 1])

    with start_col2:
        if st.button(
            "Start  →",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.meetsense_started = True
            st.rerun()

    st.stop()


'''
    text = text.replace(sidebar_marker, landing_block + sidebar_marker, 1)

APP.write_text(text)

print("SUCCESS: MeetSense UI changes applied.")
print("Backup:", BACKUP)
print("Modified:", APP)
