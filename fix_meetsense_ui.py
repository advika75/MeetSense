from pathlib import Path

APP = Path("app.py")
BACKUP = Path("app_before_ui_redesign.py")

if not BACKUP.exists():
    raise SystemExit("ERROR: app_before_ui_redesign.py not found.")

# ------------------------------------------------------------
# RESTORE CLEAN ORIGINAL
# ------------------------------------------------------------

original = BACKUP.read_text()
text = original

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

imports_marker = "import streamlit as st\n"

state_code = '''

# ============================================================
# MEETSENSE UI STATE
# ============================================================

if "meetsense_started" not in st.session_state:
    st.session_state.meetsense_started = False

if "meetsense_theme" not in st.session_state:
    st.session_state.meetsense_theme = "light"

'''

if "meetsense_started" not in text:
    text = text.replace(
        imports_marker,
        imports_marker + state_code,
        1
    )

# ------------------------------------------------------------
# PAGE CONFIG
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
# FIND THE EXISTING CSS STYLE TAG
# ------------------------------------------------------------

style_start = text.find("<style>")

if style_start == -1:
    raise SystemExit("ERROR: Existing <style> block not found.")

css_insert_position = style_start + len("<style>")

# IMPORTANT:
# We insert CSS INSIDE the existing style block.
# We do NOT insert another <style> tag.

theme_css = r'''

/* ============================================================
   MEETSENSE THEME SYSTEM
   ============================================================ */

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

/* ------------------------------------------------------------
   LANDING PAGE
   ------------------------------------------------------------ */

.ms-landing {
    min-height: 82vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 30px 10px 50px 10px;
}

.ms-landing-shell {
    width: min(1250px, 96vw);
    min-height: 680px;

    display: grid;
    grid-template-columns: 1fr 1fr;

    background: var(--ms-surface);
    border: 1px solid var(--ms-border);
    border-radius: 26px;

    overflow: hidden;

    box-shadow:
        0 25px 80px rgba(0,0,0,0.08);
}

.ms-landing-visual {
    background:
        radial-gradient(
            circle at 50% 45%,
            rgba(99,91,255,0.13),
            transparent 38%
        ),
        var(--ms-surface-2);

    display: flex;
    align-items: center;
    justify-content: center;

    padding: 50px;
}

.ms-visual-card {
    width: min(450px, 90%);
    aspect-ratio: 1 / 1;

    border-radius: 26px;

    background:
        radial-gradient(
            circle at 50% 45%,
            rgba(99,91,255,0.10),
            transparent 50%
        ),
        var(--ms-surface);

    border: 1px solid var(--ms-border);

    display: flex;
    align-items: center;
    justify-content: center;

    position: relative;
    overflow: hidden;
}

.ms-visual-circle {
    width: 265px;
    height: 265px;

    border-radius: 50%;

    background: var(--ms-accent-soft);

    position: absolute;
}

.ms-visual-icon {
    position: relative;
    z-index: 2;

    width: 150px;
    height: 150px;

    border-radius: 45px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: var(--ms-text);
    color: var(--ms-surface);

    font-size: 65px;

    box-shadow:
        0 20px 50px rgba(0,0,0,0.16);
}

.ms-wave {
    position: absolute;

    bottom: 75px;
    left: 18%;
    right: 18%;

    height: 55px;

    display: flex;
    align-items: center;
    justify-content: center;

    gap: 5px;

    z-index: 3;
}

.ms-wave span {
    width: 4px;
    border-radius: 10px;
    background: var(--ms-accent);
    opacity: .8;
}

.ms-wave span:nth-child(1) { height: 14px; }
.ms-wave span:nth-child(2) { height: 26px; }
.ms-wave span:nth-child(3) { height: 40px; }
.ms-wave span:nth-child(4) { height: 25px; }
.ms-wave span:nth-child(5) { height: 53px; }
.ms-wave span:nth-child(6) { height: 35px; }
.ms-wave span:nth-child(7) { height: 45px; }
.ms-wave span:nth-child(8) { height: 24px; }
.ms-wave span:nth-child(9) { height: 36px; }
.ms-wave span:nth-child(10) { height: 16px; }

/* ------------------------------------------------------------
   LANDING CONTENT
   ------------------------------------------------------------ */

.ms-landing-content {
    display: flex;
    flex-direction: column;
    justify-content: center;

    padding: 65px;

    color: var(--ms-text);
}

.ms-brand {
    display: flex;
    align-items: center;
    gap: 12px;

    margin-bottom: 65px;
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
    color: var(--ms-text);
}

.ms-eyebrow {
    font-size: 12px;
    font-weight: 700;

    letter-spacing: .14em;

    color: var(--ms-accent);

    margin-bottom: 18px;
}

.ms-landing-title {
    font-size: clamp(43px, 5vw, 68px);

    line-height: 1.03;

    letter-spacing: -0.045em;

    font-weight: 800;

    color: var(--ms-text);

    margin-bottom: 24px;
}

.ms-landing-subtitle {
    font-size: 17px;

    line-height: 1.7;

    color: var(--ms-text-secondary);

    max-width: 510px;

    margin-bottom: 32px;
}

.ms-feature-row {
    display: flex;

    gap: 9px;

    flex-wrap: wrap;

    margin-bottom: 35px;
}

.ms-feature {
    border: 1px solid var(--ms-border);

    background: var(--ms-surface-2);

    border-radius: 999px;

    padding: 8px 12px;

    font-size: 12px;

    color: var(--ms-text-secondary);
}

/* ------------------------------------------------------------
   THEME CONTROLS
   ------------------------------------------------------------ */

.ms-theme-title {
    font-size: 12px;
    font-weight: 600;

    color: var(--ms-text-secondary);

    margin-bottom: 8px;
}

/* ------------------------------------------------------------
   SIDEBAR READABILITY
   ------------------------------------------------------------ */

section[data-testid="stSidebar"] {
    background: var(--ms-surface) !important;

    border-right:
        1px solid var(--ms-border) !important;
}

section[data-testid="stSidebar"] * {
    color: var(--ms-text) !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--ms-text) !important;
}

@media (max-width: 900px) {

    .ms-landing-shell {
        grid-template-columns: 1fr;
    }

    .ms-landing-visual {
        min-height: 390px;
    }

    .ms-landing-content {
        padding: 45px;
    }

    .ms-brand {
        margin-bottom: 40px;
    }
}

'''

text = (
    text[:css_insert_position]
    + theme_css
    + text[css_insert_position:]
)

# ------------------------------------------------------------
# LANDING PAGE
# ------------------------------------------------------------

sidebar_marker = '''# ============================================================
# SIDEBAR
# ============================================================'''

if sidebar_marker not in text:
    raise SystemExit("ERROR: Sidebar marker not found.")

landing_page = r'''

# ============================================================
# MEETSENSE LANDING PAGE
# ============================================================

if not st.session_state.meetsense_started:

    current_theme = st.session_state.meetsense_theme

    wrapper_class = "ms-dark" if current_theme == "dark" else ""

    st.markdown(
        f'<div class="{wrapper_class}">',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="ms-landing">

            <div class="ms-landing-shell">

                <!-- LEFT VISUAL -->

                <div class="ms-landing-visual">

                    <div class="ms-visual-card">

                        <div class="ms-visual-circle"></div>

                        <div class="ms-visual-icon">
                            🎙
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

                <!-- RIGHT CONTENT -->

                <div class="ms-landing-content">

                    <div class="ms-brand">

                        <div class="ms-brand-logo">
                            ∞
                        </div>

                        <div class="ms-brand-name">
                            MeetSense
                        </div>

                    </div>

                    <div class="ms-eyebrow">
                        SPEAKER-AWARE MEETING INTELLIGENCE
                    </div>

                    <div class="ms-landing-title">
                        Meeting intelligence,
                        <br>
                        simplified.
                    </div>

                    <div class="ms-landing-subtitle">
                        Turn meeting recordings into speaker-aware
                        transcripts, concise summaries, topics,
                        decisions and actionable tasks.
                    </div>

                    <div class="ms-feature-row">

                        <div class="ms-feature">
                            Whisper ASR
                        </div>

                        <div class="ms-feature">
                            Speaker Diarization
                        </div>

                        <div class="ms-feature">
                            NLP Analysis
                        </div>

                        <div class="ms-feature">
                            AI Summary
                        </div>

                    </div>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ms-theme-title">Appearance</div>',
        unsafe_allow_html=True
    )

    theme_left, theme_right, start_col = st.columns(
        [1, 1, 2]
    )

    with theme_left:

        if st.button(
            "☀ Light",
            use_container_width=True,
            type=(
                "primary"
                if current_theme == "light"
                else "secondary"
            ),
        ):
            st.session_state.meetsense_theme = "light"
            st.rerun()

    with theme_right:

        if st.button(
            "🌙 Dark",
            use_container_width=True,
            type=(
                "primary"
                if current_theme == "dark"
                else "secondary"
            ),
        ):
            st.session_state.meetsense_theme = "dark"
            st.rerun()

    with start_col:

        if st.button(
            "Start  →",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.meetsense_started = True
            st.rerun()

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    st.stop()

'''

text = text.replace(
    sidebar_marker,
    landing_page + "\n" + sidebar_marker,
    1
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

APP.write_text(text)

print()
print("==============================================")
print(" MeetSense UI FIX COMPLETE")
print("==============================================")
print()
print("Restored from:")
print("  app_before_ui_redesign.py")
print()
print("Updated:")
print("  app.py")
print()
print("Next: run python -m py_compile app.py")
print()
