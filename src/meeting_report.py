"""
Generic meeting-report foundation (no LLM yet). Works for either a
batch ("recorded") or live meeting, since it only consumes the same
speaker-aware transcript shape both paths already produce.
RuleBasedExtractor wraps the existing regex logic in
action_item_extraction.py -- no domain-specific rules added here.
"""

import html as html_lib
import json
import os
import sys
from datetime import datetime
from typing import Callable, Optional

from pydantic import BaseModel, Field

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from action_item_extraction import (
    get_text,
    get_speaker,
    get_start_time,
    contains_pattern,
    extract_deadline,
    clean_action_text,
    ACTION_PATTERNS,
    DECISION_PATTERNS,
)


OUTPUT_ROOT = "outputs/reports"


# ============================================================
# MODELS
# ============================================================

class Evidence(BaseModel):
    speaker: str
    timestamp: float
    quote: str


class TakeawayItem(BaseModel):
    text: str
    owner: Optional[str] = None
    due_date: Optional[str] = None  # resolved ISO date, if any
    due_date_text: Optional[str] = None  # original phrase ("by Friday")
    confidence: Optional[float] = None
    evidence: Evidence


class Speaker(BaseModel):
    name: str
    speaking_time_sec: float
    share_pct: float


class MeetingReport(BaseModel):

    meeting_id: str
    title: str | None = None
    source: str  # "live" | "recorded"
    date: str
    duration_sec: float
    speakers: list[Speaker] = Field(default_factory=list)
    summary: str = ""
    chunk_summaries: list[str] = Field(default_factory=list)
    topics: list[dict] = Field(default_factory=list)
    decisions: list[TakeawayItem] = Field(default_factory=list)
    action_items: list[TakeawayItem] = Field(default_factory=list)
    deadlines: list[TakeawayItem] = Field(default_factory=list)
    open_questions: list[TakeawayItem] = Field(default_factory=list)
    transcript: list[dict] = Field(default_factory=list)
    extractor_used: str = "rule_based"
    extractor_stats: dict = Field(default_factory=dict)

    def to_rendered_dict(self):
        """Dict form with empty sections omitted (JSON/MD/HTML export)."""

        data = self.model_dump()

        return {
            key: value
            for key, value in data.items()
            if value not in ([], "", None)
        }


# ============================================================
# EXTRACTOR REGISTRY
# ============================================================

class TakeawayExtractor:

    name = "base"

    def is_available(self):
        """Returns (ok: bool, resolved_model_or_message: str). Base
        extractors (e.g. rule-based) have no external dependency and
        are always available."""

        return True, "ok"

    def extract(self, transcript_segments, meeting_date, chunk_summaries=None):
        """Returns a dict with keys decisions/action_items/deadlines/
        open_questions (each a list of TakeawayItem), and optionally
        "summary" (str) and "stats" (dict, e.g. items dropped)."""

        raise NotImplementedError


_EXTRACTOR_REGISTRY: dict[str, Callable[[], TakeawayExtractor]] = {}


def register_extractor(name):

    def decorator(cls):
        _EXTRACTOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_extractor(name):

    if name not in _EXTRACTOR_REGISTRY:

        raise KeyError(
            f"Unknown extractor '{name}'. "
            f"Available: {list(_EXTRACTOR_REGISTRY)}"
        )

    return _EXTRACTOR_REGISTRY[name]()


def available_extractors():
    return list(_EXTRACTOR_REGISTRY)


# ============================================================
# RULE-BASED EXTRACTOR (wraps existing regex logic -- no new rules)
# ============================================================

@register_extractor("rule_based")
class RuleBasedExtractor(TakeawayExtractor):

    name = "rule_based"

    def extract(self, transcript_segments, meeting_date, chunk_summaries=None):

        decisions = []
        action_items = []

        for segment in transcript_segments:

            text = get_text(segment)

            if not text:
                continue

            speaker = get_speaker(segment)
            start_time = get_start_time(segment) or 0.0
            deadline = extract_deadline(text)
            cleaned = clean_action_text(text)

            evidence = Evidence(
                speaker=speaker,
                timestamp=float(start_time),
                quote=cleaned,
            )

            if contains_pattern(text, ACTION_PATTERNS):

                action_items.append(
                    TakeawayItem(
                        text=cleaned,
                        due_date=deadline,
                        evidence=evidence,
                    )
                )

            if contains_pattern(text, DECISION_PATTERNS):

                decisions.append(
                    TakeawayItem(
                        text=cleaned,
                        evidence=evidence,
                    )
                )

        return {
            "decisions": decisions,
            "action_items": action_items,
            "deadlines": [
                item for item in action_items if item.due_date
            ],
            # Rule-based regex has no open-question detector.
            "open_questions": [],
        }


# ============================================================
# BUILD REPORT
# ============================================================

def compute_speaking_time(transcript_segments):

    totals = {}

    for segment in transcript_segments:

        speaker = segment.get("speaker", "UNKNOWN")

        duration = max(
            0.0,
            float(segment.get("end", 0)) - float(segment.get("start", 0)),
        )

        totals[speaker] = totals.get(speaker, 0.0) + duration

    total_speaking = sum(totals.values()) or 1.0

    return [
        Speaker(
            name=speaker,
            speaking_time_sec=round(duration, 2),
            share_pct=round(100 * duration / total_speaking, 1),
        )
        for speaker, duration in sorted(
            totals.items(), key=lambda kv: -kv[1]
        )
    ]


def build_report(
    meeting_id,
    transcript_segments=None,
    chunk_summaries=None,
    final_summary=None,
    topics=None,
    source="recorded",
    meeting_date=None,
    extractor_name="rule_based",
    transcript_path="outputs/transcripts/speaker_transcript.json",
    summary_path="outputs/summaries/meeting_summary.json",
    topics_path="outputs/nlp/topic_modeling.json",
    pre_extract_hook=None,
    title=None,
):
    """Builds a MeetingReport either from explicitly-passed data (the
    live path, source="live") or from the existing batch output
    files (the recorded path, source="recorded"), saves it to
    outputs/reports/<meeting_id>/report.json, and returns it.

    If extractor_name's extractor reports itself unavailable (e.g.
    the "llm" extractor when Ollama isn't reachable), this falls
    back to "rule_based" and records that in report.extractor_used,
    rather than failing the whole report build.

    pre_extract_hook, if given, is called with no arguments right
    before extraction -- e.g. to unload heavy models (Whisper/
    pyannote) before an LLM extractor call on a memory-constrained
    machine. Only called for extractors other than "rule_based",
    since rule-based needs no such unload.

    title, if given, becomes the report's user-facing display name.
    If omitted, a title set on an EXISTING report with this
    meeting_id is preserved across a rebuild (rebuilding re-runs
    extraction but shouldn't silently erase a name the user already
    set)."""

    if meeting_date is None:
        meeting_date = datetime.now().strftime("%Y-%m-%d")

    if title is None:
        try:
            title = load_report(meeting_id).title
        except FileNotFoundError:
            title = None

    if transcript_segments is None:

        with open(transcript_path, "r", encoding="utf-8") as file:
            transcript_segments = json.load(file)

    if chunk_summaries is None or final_summary is None:

        if os.path.exists(summary_path):

            with open(summary_path, "r", encoding="utf-8") as file:
                summary_data = json.load(file)

            if chunk_summaries is None:
                chunk_summaries = summary_data.get("chunk_summaries", [])

            if final_summary is None:
                final_summary = summary_data.get("final_summary", "")

        else:

            chunk_summaries = chunk_summaries or []
            final_summary = final_summary or ""

    if topics is None:

        if os.path.exists(topics_path):

            with open(topics_path, "r", encoding="utf-8") as file:
                topics_data = json.load(file)

            topics = topics_data.get("topics", [])

        else:
            topics = []

    duration_sec = max(
        (float(segment.get("end", 0)) for segment in transcript_segments),
        default=0.0,
    )

    speakers = compute_speaking_time(transcript_segments)

    extractor = get_extractor(extractor_name)

    extractor_used = extractor_name
    fallback_notice = None

    ok, availability_message = extractor.is_available()

    if not ok:

        fallback_notice = (
            f"Extractor '{extractor_name}' unavailable "
            f"({availability_message}); used 'rule_based' instead."
        )

        extractor = get_extractor("rule_based")
        extractor_used = "rule_based"

    else:

        if extractor_name != "rule_based" and pre_extract_hook is not None:
            pre_extract_hook()

    extracted = extractor.extract(
        transcript_segments, meeting_date, chunk_summaries=chunk_summaries
    )

    if extracted.get("summary"):
        final_summary = extracted["summary"]

    extractor_stats = extracted.get("stats", {})

    if fallback_notice:
        extractor_stats["fallback_notice"] = fallback_notice

    report = MeetingReport(
        meeting_id=meeting_id,
        title=title,
        source=source,
        date=meeting_date,
        duration_sec=round(duration_sec, 2),
        speakers=speakers,
        summary=final_summary,
        chunk_summaries=chunk_summaries,
        topics=topics,
        decisions=extracted["decisions"],
        action_items=extracted["action_items"],
        deadlines=extracted["deadlines"],
        open_questions=extracted["open_questions"],
        transcript=transcript_segments,
        extractor_used=extractor_used,
        extractor_stats=extractor_stats,
    )

    save_report(report)

    return report


def save_report(report):

    report_dir = os.path.join(OUTPUT_ROOT, report.meeting_id)
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, "report.json")

    with open(report_path, "w", encoding="utf-8") as file:

        json.dump(
            report.to_rendered_dict(),
            file,
            indent=2,
            ensure_ascii=False,
        )

    return report_path


def load_report(meeting_id):

    report_path = os.path.join(OUTPUT_ROOT, meeting_id, "report.json")

    with open(report_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return MeetingReport.model_validate(data)


def list_report_ids():

    if not os.path.isdir(OUTPUT_ROOT):
        return []

    return sorted(
        name
        for name in os.listdir(OUTPUT_ROOT)
        if os.path.exists(os.path.join(OUTPUT_ROOT, name, "report.json"))
    )


def apply_speaker_renaming(report, renaming):
    """renaming: dict old_name -> new_name. Applied to speakers,
    transcript, and every evidence.speaker across all takeaway
    sections, so a rename is consistent everywhere in the report."""

    for speaker in report.speakers:
        if speaker.name in renaming:
            speaker.name = renaming[speaker.name]

    for segment in report.transcript:
        if segment.get("speaker") in renaming:
            segment["speaker"] = renaming[segment["speaker"]]

    for section in (
        report.decisions,
        report.action_items,
        report.deadlines,
        report.open_questions,
    ):
        for item in section:
            if item.evidence.speaker in renaming:
                item.evidence.speaker = renaming[item.evidence.speaker]

    return report


# ============================================================
# EXPORT RENDERERS
# ============================================================

def render_markdown(report):

    lines = [
        f"# Meeting Report: {report.title or report.meeting_id}",
        "",
        f"**Date:** {report.date}  ",
        f"**Duration:** {report.duration_sec:.0f}s  ",
        f"**Source:** {report.source}",
        "",
    ]

    if report.summary:
        lines += ["## Summary", "", report.summary, ""]

    if report.speakers:

        lines += ["## Speakers", ""]

        for speaker in report.speakers:

            lines.append(
                f"- **{speaker.name}**: "
                f"{speaker.speaking_time_sec:.0f}s "
                f"({speaker.share_pct:.1f}%)"
            )

        lines.append("")

    for section_name, section in [
        ("Decisions", report.decisions),
        ("Action Items", report.action_items),
        ("Deadlines", report.deadlines),
        ("Open Questions", report.open_questions),
    ]:

        if section:

            lines += [f"## {section_name}", ""]

            for item in section:

                due = f" (due: {item.due_date})" if item.due_date else ""

                lines.append(
                    f"- {item.text}{due} "
                    f"— *{item.evidence.speaker} @ "
                    f"{item.evidence.timestamp:.0f}s*"
                )

                lines.append(f"  > {item.evidence.quote}")

            lines.append("")

    return "\n".join(lines)


def render_html(report):
    """Minimal, dependency-free Markdown->HTML (no markdown library):
    wraps the rendered Markdown in a styled <pre> block."""

    escaped = html_lib.escape(render_markdown(report))

    return (
        "<html><head><meta charset='utf-8'></head>"
        "<body style='font-family:-apple-system,sans-serif;"
        "max-width:800px;margin:40px auto;line-height:1.6;'>"
        f"<pre style='white-space:pre-wrap;font-family:inherit;'>"
        f"{escaped}</pre></body></html>"
    )


# ============================================================
# AUTO-REGISTER OPTIONAL EXTRACTORS
# ============================================================
# Imported here (after TakeawayExtractor/register_extractor/
# TakeawayItem/Evidence are defined above) so llm_extractor.py's
# `from meeting_report import ...` resolves against this
# already-partially-initialized module -- importing
# meeting_report is then enough to make "llm" show up in
# available_extractors(), with no separate import needed at call
# sites. No heavy (torch-adjacent) dependency is involved here --
# llm_extractor only needs requests/pydantic/rapidfuzz/dateparser.

try:
    import llm_extractor  # noqa: F401
except Exception as _llm_extractor_import_error:
    _llm_extractor_import_error_message = str(_llm_extractor_import_error)
else:
    _llm_extractor_import_error_message = None

try:
    import hybrid_extractor  # noqa: F401
except Exception as _hybrid_extractor_import_error:
    _hybrid_extractor_import_error_message = str(_hybrid_extractor_import_error)
else:
    _hybrid_extractor_import_error_message = None
