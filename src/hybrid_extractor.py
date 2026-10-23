"""
Hybrid takeaway extractor: high-recall candidate detection, then a
short per-candidate LLM classification. Registers as "hybrid" in the
meeting_report extractor registry (imported at the bottom of
meeting_report.py, next to llm_extractor).

Why: the chunk-level "llm" extractor asks the model to find everything
in ~9 minutes of transcript at once and then relies on heavy
dedupe/routing heuristics. Here the transcript is first scanned with
loose commitment/decision/date cues (recall), and each candidate gets
its own small, temperature-0 classification call over the candidate
line +-3 neighbours (precision). Nothing here is meeting-specific.

Steps:
  1. detect_candidates(): cue regexes + dateparser-validated date phrases.
  2. classify_candidate(): one JSON call -> action_item | decision |
     discussion | none, with owner/task/due_text/line_ids.
  3. merge_classified(): merge candidates sharing line_ids, or within
     +-10s with highly similar task text (same type only).
  4. resolve due dates with dateparser relative to the meeting date;
     evidence quotes are always the real transcript lines.
"""

import json
import re
import subprocess
from datetime import datetime

from rapidfuzz import fuzz

try:
    import dateparser
except ImportError:
    dateparser = None

from meeting_report import (
    TakeawayExtractor,
    register_extractor,
    TakeawayItem,
    Evidence,
)
from llm_extractor import (
    OLLAMA_MODEL,
    OLLAMA_URL,
    call_ollama,
    check_ollama_available,
    meaningful_tokens,
    resolve_due_date,
)

import requests


CONTEXT_LINES = 3
MERGE_TIME_WINDOW_SECONDS = 10
MERGE_TEXT_SIMILARITY = 75  # rapidfuzz token_sort_ratio, 0-100
MERGE_SHARED_LINE_FRACTION = 0.5
CLASSIFY_NUM_PREDICT = 200
CLASSIFY_KEEP_ALIVE = "10m"  # keep model resident for the whole run
TYPES = {"action_item", "decision", "discussion", "none"}

# ------------------------------------------------------------------
# 1. CANDIDATE DETECTION (deliberately loose)
# ------------------------------------------------------------------

COMMITMENT_CUES = [
    r"\bhave to\b", r"\bhas to\b", r"\bhad to\b", r"\bneed to\b",
    r"\bneeds to\b", r"\bwill\b", r"\bwon't\b", r"\bi'll\b", r"\bwe'll\b",
    r"\byou'll\b", r"\bthey'll\b", r"\bgoing to\b", r"\bgonna\b",
    r"\bshould\b", r"\bmust\b", r"\blet's\b", r"\blet us\b",
    r"\bassigned\b", r"\bresponsible\b", r"\bdeadline\b", r"\bdue\b",
    r"\bby (?:the )?(?:end of|next|this|tomorrow|today|tonight|monday|"
    r"tuesday|wednesday|thursday|friday|saturday|sunday|\d)",
    r"\btake care of\b", r"\bfollow up\b", r"\baction item\b",
]

# Decision cues are added so decisions stay reachable; the classifier
# still decides the type.
DECISION_CUES = [
    r"\bagree[ds]?\b", r"\bdecid(?:e|ed|ing)\b", r"\bgo(?:ing)? with\b",
    r"\bsettled\b", r"\bfinal\b", r"\bapproved?\b", r"\bconfirmed?\b",
]

CUE_PATTERN = re.compile(
    "|".join(COMMITMENT_CUES + DECISION_CUES), re.IGNORECASE
)

DATE_PHRASE_PATTERN = re.compile(
    r"\b(?:by |before |on |until |due )?"
    r"(?:(?:next|this|last) (?:week|month|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday|quarter)|"
    r"tomorrow|tonight|today|later today|end of (?:the )?(?:day|week|month|"
    r"quarter|year)|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"(?:january|february|march|april|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
    r"\.? \d{1,2}(?:st|nd|rd|th)?|"
    r"\d{1,2}(?:st|nd|rd|th)? of (?:january|february|march|april|may|june|"
    r"july|august|september|october|november|december)|"
    r"in (?:a|an|one|two|three|four|five|six|\d+) (?:day|week|month|year)s?)\b",
    re.IGNORECASE,
)


def detect_date_phrase(text):
    """LLM-free date detector: a loose phrase regex whose match must
    also be parseable by dateparser (so only real dates survive)."""

    if dateparser is None:
        return None

    for match in DATE_PHRASE_PATTERN.finditer(text):
        phrase = match.group(0).strip()
        try:
            parsed = dateparser.parse(re.sub(r"^(?:by|before|on|until|due) ", "", phrase, flags=re.I))
        except Exception:
            parsed = None
        if parsed is not None:
            return phrase

    return None


def detect_candidates(segments):
    """Returns sorted line indices that look like commitments,
    decisions, or carry a date. A cue may sit in the line itself or in
    the join of this and the next line (ASR splits sentences mid-way)."""

    candidates = []

    for index, segment in enumerate(segments):

        text = str(segment.get("text", "")).strip()

        if not text:
            continue

        next_text = (
            str(segments[index + 1].get("text", "")).strip()
            if index + 1 < len(segments) else ""
        )

        if CUE_PATTERN.search(text) or detect_date_phrase(f"{text} {next_text}"):
            candidates.append(index)

    return candidates


# ------------------------------------------------------------------
# 2. CLASSIFIER
# ------------------------------------------------------------------

CLASSIFY_PROMPT = """You label ONE line of a meeting transcript: the line marked ">>". Use the surrounding lines only as context. Use only what is explicitly said; do not invent.

Types:
- "action_item": a specific person or group commits to, or is assigned, a concrete thing to do.
- "decision": the group explicitly agreed or settled on something.
- "discussion": a suggestion, idea, or "we should consider/maybe" that nobody committed to.
- "none": background information, description, questions, reactions, or other chatter. Most lines are "none". A line is not a "discussion" just because it mentions a topic.

Return JSON only: {{"reason": at most 12 words quoting what the marked line itself says, "type": ..., "owner": string or null, "task": short restatement or "", "due_text": exact timing phrase or null, "line_ids": [integer ids of lines that support it]}}

Example 1
[L30] EVA (02:40): The weekly sync is a bit long lately.
>>[L31] FIN (02:44): We should probably think about shortening the weekly sync at some point.
[L32] EVA (02:49): Maybe, I'm not sure yet.
Answer: {{"reason": "suggests shortening, nobody commits", "type": "discussion", "owner": null, "task": "Consider shortening the weekly sync", "due_text": null, "line_ids": [31]}}

Example 2
[L20] CAL (01:02): We looked at the three vendors again.
>>[L21] DEE (01:06): Right, so we're going with the second vendor for the contract.
[L22] CAL (01:09): Yes, everybody is fine with that.
Answer: {{"reason": "group settles on the second vendor", "type": "decision", "owner": null, "task": "Use the second vendor for the contract", "due_text": null, "line_ids": [21, 22]}}

Example 3
[L10] ANA (00:10): Okay, who can update the pricing sheet?
>>[L11] BEN (00:13): I'll update the pricing sheet and send it to the team by Thursday.
[L12] ANA (00:17): Great, thanks.
Answer: {{"reason": "Ben says I'll update it by Thursday", "type": "action_item", "owner": "Ben", "task": "Update the pricing sheet and send it to the team", "due_text": "by Thursday", "line_ids": [11]}}

Now label the marked line.
{window}
Answer:"""


def format_window(segments, index, context=CONTEXT_LINES):

    start = max(0, index - context)
    end = min(len(segments), index + context + 1)
    lines = []

    for line_index in range(start, end):

        segment = segments[line_index]
        minutes, seconds = divmod(int(float(segment.get("start", 0))), 60)
        marker = ">>" if line_index == index else ""
        lines.append(
            f"{marker}[L{line_index}] {segment.get('speaker', 'UNKNOWN')} "
            f"({minutes:02d}:{seconds:02d}): "
            f"{str(segment.get('text', '')).strip()}"
        )

    return "\n".join(lines), start, end


def classify_candidate(segments, index, model):
    """Returns a dict {type, owner, task, due_text, line_ids} with
    line_ids restricted to the shown window, or None on failure."""

    window_text, start, end = format_window(segments, index)
    prompt = CLASSIFY_PROMPT.format(window=window_text)

    for _ in range(2):

        try:
            raw = call_ollama(
                prompt, model, keep_alive=CLASSIFY_KEEP_ALIVE,
                num_predict=CLASSIFY_NUM_PREDICT, num_ctx_floor=2048,
            )
            data = json.loads(raw)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        item_type = data.get("type")
        if item_type not in TYPES:
            continue

        line_ids = sorted({
            line_id for line_id in (data.get("line_ids") or [])
            if isinstance(line_id, int) and start <= line_id < end
        }) or [index]

        def clean(value):
            return value.strip() if isinstance(value, str) and value.strip() else None

        return {
            "type": item_type,
            "owner": clean(data.get("owner")),
            "task": clean(data.get("task")) or "",
            "due_text": clean(data.get("due_text")),
            "line_ids": line_ids,
            "candidate": index,
        }

    return None


# ------------------------------------------------------------------
# 3. MERGE + GROUNDING
# ------------------------------------------------------------------

def is_grounded(task, segments, line_ids):
    """Same sanity rule as the chunk extractor: the restated task must
    share vocabulary with the lines it cites."""

    quote = " ".join(str(segments[i].get("text", "")) for i in line_ids)
    task_tokens = meaningful_tokens(task)

    if not task_tokens:
        return False

    overlap = len(task_tokens & meaningful_tokens(quote)) / len(task_tokens)
    fuzzy = fuzz.partial_ratio(task, quote) / 100.0

    return overlap >= 0.3 or fuzzy >= 0.5


def same_item(left, right, segments):

    if left["type"] != right["type"]:
        return False

    left_lines, right_lines = set(left["line_ids"]), set(right["line_ids"])
    shared = len(left_lines & right_lines) / min(len(left_lines), len(right_lines))
    if shared >= MERGE_SHARED_LINE_FRACTION:
        return True

    left_time = float(segments[min(left["line_ids"])].get("start", 0))
    right_time = float(segments[min(right["line_ids"])].get("start", 0))

    return (
        abs(left_time - right_time) <= MERGE_TIME_WINDOW_SECONDS
        and fuzz.token_sort_ratio(left["task"], right["task"]) >= MERGE_TEXT_SIMILARITY
    )


def merge_classified(classified, segments):

    merged = []

    for item in sorted(classified, key=lambda entry: min(entry["line_ids"])):

        target = next(
            (existing for existing in merged if same_item(existing, item, segments)),
            None,
        )

        if target is None:
            merged.append(dict(item))
            continue

        target["line_ids"] = sorted(set(target["line_ids"]) | set(item["line_ids"]))
        target["owner"] = target["owner"] or item["owner"]
        target["due_text"] = target["due_text"] or item["due_text"]
        if len(item["task"]) > len(target["task"]):
            target["task"] = item["task"]

    return merged


def to_takeaway(entry, segments, meeting_date):

    line_ids = entry["line_ids"]
    cited = [segments[i] for i in line_ids]
    quote = " ".join(
        str(segment.get("text", "")).strip() for segment in cited
        if str(segment.get("text", "")).strip()
    )

    due_text = entry["due_text"] or detect_date_phrase(quote)

    return TakeawayItem(
        text=entry["task"],
        owner=entry["owner"],
        due_date_text=due_text,
        due_date=resolve_due_date(due_text, meeting_date) if due_text else None,
        evidence=Evidence(
            speaker=str(cited[0].get("speaker", "UNKNOWN")),
            timestamp=float(cited[0].get("start", 0)),
            quote=quote,
        ),
    )


def unload_model(model):
    """Release the model from Ollama's memory (8GB machine)."""

    try:
        requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
    except Exception:
        pass


# ------------------------------------------------------------------
# EXTRACTOR
# ------------------------------------------------------------------

@register_extractor("hybrid")
class HybridExtractor(TakeawayExtractor):

    name = "hybrid"

    def __init__(self):
        self.last_run_stats = {}

    def is_available(self):
        return check_ollama_available()

    def extract(self, transcript_segments, meeting_date, chunk_summaries=None):

        ok, model = self.is_available()

        if not ok:
            raise RuntimeError(model)

        segments = transcript_segments
        candidates = detect_candidates(segments)

        classified = []
        failed = 0
        counts = {"action_item": 0, "decision": 0, "discussion": 0, "none": 0}

        try:
            for index in candidates:

                result = classify_candidate(segments, index, model)

                if result is None:
                    failed += 1
                    continue

                counts[result["type"]] += 1

                if result["type"] == "none":
                    continue

                if not is_grounded(result["task"], segments, result["line_ids"]):
                    counts["dropped_ungrounded"] = counts.get("dropped_ungrounded", 0) + 1
                    continue

                classified.append(result)
        finally:
            unload_model(model)

        merged = merge_classified(classified, segments)

        buckets = {"action_item": [], "decision": [], "discussion": []}
        for entry in merged:
            buckets[entry["type"]].append(to_takeaway(entry, segments, meeting_date))

        action_items = buckets["action_item"]
        decisions = buckets["decision"]
        open_questions = buckets["discussion"]
        deadlines = [
            item for item in action_items + decisions if item.due_date_text
        ]

        self.last_run_stats = {
            "model": model,
            "lines": len(segments),
            "candidates": len(candidates),
            "classification_failures": failed,
            "classified_counts": counts,
            "merged_items": len(merged),
        }

        return {
            "decisions": decisions,
            "action_items": action_items,
            "deadlines": deadlines,
            "open_questions": open_questions,
            "summary": "",
            "stats": self.last_run_stats,
        }
