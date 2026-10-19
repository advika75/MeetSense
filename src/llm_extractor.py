"""
Local-LLM takeaway extractor (Ollama). Registers as "llm" in the
meeting_report extractor registry (see the import at the bottom of
meeting_report.py). Works for any meeting type -- the prompt asks
only for explicit commitments/decisions/deadlines actually present
in the chunk; nothing here is domain-specific.

Map-reduce pipeline:
  1. chunk_transcript(): split into ~8-10 min windows with overlap.
  2. extract_chunk(): ask Ollama per chunk, JSON mode, temperature 0.
  3. validate + anti-hallucination: pydantic parse (retry once on
     invalid JSON), then ground items to numbered transcript lines.
  4. reduce_items(): merge across chunks, dedupe near-duplicates by
     fuzzy text similarity, keep the earliest evidence.
  5. resolve_due_dates(): dateparser, relative to the meeting date.
  6. summarize(): ask Ollama for a 3-5 sentence overview from the
     existing chunk summaries (no transcript re-read needed).
"""

import json
import os
import re
from datetime import datetime
from typing import Optional

import requests
from pydantic import BaseModel, ValidationError
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


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_FALLBACK_MODEL = os.environ.get(
    "OLLAMA_FALLBACK_MODEL", "qwen2.5:1.5b"
)
# Release the model from Ollama's memory right after use -- this is
# an 8GB machine and the model shouldn't sit resident between calls.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "0")

CHUNK_SECONDS = 9 * 60  # ~8-10 minutes
CHUNK_OVERLAP_SECONDS = 30
MAX_CHUNK_CHARS = 6000  # rough ~1500-token budget for the transcript excerpt

FUZZY_QUOTE_THRESHOLD = 0.85  # fraction, per spec
DEDUP_SIMILARITY_THRESHOLD = 90  # rapidfuzz score, 0-100
EVIDENCE_TEXT_SIMILARITY_THRESHOLD = 70  # normalized score, 0-100

REQUEST_TIMEOUT_SECONDS = 300
MAX_OUTPUT_TOKENS = 1200


# ============================================================
# RAW (UNVALIDATED-AGAINST-TRANSCRIPT) MODEL OUTPUT SCHEMA
# ============================================================

class RawTakeaway(BaseModel):
    text: str
    owner: Optional[str] = None
    due_date_text: Optional[str] = None
    line_ids: list[int] = []


class RawChunkResult(BaseModel):
    """Raw category lists, kept as list[dict] rather than
    list[RawTakeaway] here: a single malformed item inside one
    category must not invalidate every other item in the chunk. Each
    dict is validated into RawTakeaway individually in
    parse_chunk_items(), so one bad item is just dropped and counted,
    not a reason to discard the whole chunk's real findings."""

    decisions: list[dict] = []
    action_items: list[dict] = []
    deadlines: list[dict] = []
    open_questions: list[dict] = []


def parse_chunk_items(raw_dicts, schema_drop_counter):

    items = []

    for raw_dict in raw_dicts:

        try:
            items.append(RawTakeaway.model_validate(raw_dict))
        except ValidationError:
            schema_drop_counter["dropped"] += 1

    return items


# ============================================================
# OLLAMA AVAILABILITY
# ============================================================

def check_ollama_available(model=None):
    """Returns (ok: bool, resolved_model_or_message: str)."""

    model = model or OLLAMA_MODEL

    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        response.raise_for_status()
    except Exception as error:
        return False, f"Ollama unreachable at {OLLAMA_URL}: {error}"

    available_models = [
        entry.get("name", "")
        for entry in response.json().get("models", [])
    ]

    if model in available_models:
        return True, model

    if OLLAMA_FALLBACK_MODEL in available_models:
        return True, OLLAMA_FALLBACK_MODEL

    return False, (
        f"Neither '{model}' nor fallback '{OLLAMA_FALLBACK_MODEL}' "
        f"is pulled in Ollama. Available: {available_models}"
    )


def call_ollama(
    prompt, model, keep_alive=OLLAMA_KEEP_ALIVE,
    num_predict=MAX_OUTPUT_TOKENS,
):

    estimated_prompt_tokens = (len(prompt) + 3) // 4
    num_ctx = max(4096, estimated_prompt_tokens + num_predict + 256)

    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
        "keep_alive": keep_alive,
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    return response.json().get("response", "")


# ============================================================
# CHUNKING
# ============================================================

def chunk_transcript(
    transcript_segments,
    chunk_seconds=CHUNK_SECONDS,
    overlap_seconds=CHUNK_OVERLAP_SECONDS,
):
    """Splits into ~chunk_seconds windows by segment start time, with
    a small overlap so a commitment spoken right at a boundary still
    appears whole in at least one chunk."""

    if not transcript_segments:
        return []

    max_time = max(
        float(segment.get("end", 0)) for segment in transcript_segments
    )

    chunks = []
    window_start = 0.0

    while window_start <= max_time:

        window_end = window_start + chunk_seconds

        chunk_segments = [
            segment
            for segment in transcript_segments
            if window_start - overlap_seconds
            <= float(segment.get("start", 0))
            < window_end
        ]

        if chunk_segments:
            chunks.append(chunk_segments)

        window_start = window_end

    return chunks


def format_chunk_for_prompt(chunk_segments, max_chars=MAX_CHUNK_CHARS):

    lines = []
    char_count = 0

    for line_id, segment in enumerate(chunk_segments):

        timestamp = float(segment.get("start", 0))
        minutes, seconds = divmod(int(timestamp), 60)

        line = (
            f"[L{line_id}] {segment.get('speaker', 'UNKNOWN')} "
            f"({minutes:02d}:{seconds:02d}): "
            f"{segment.get('text', '').strip()}"
        )

        if char_count + len(line) > max_chars:
            break

        lines.append(line)
        char_count += len(line)

    return "\n".join(lines)


# ============================================================
# MAP: PROMPT + PER-CHUNK EXTRACTION
# ============================================================

PROMPT_TEMPLATE = """You are extracting ONLY what was explicitly said in this meeting transcript excerpt. Do not infer, guess, summarize loosely, or invent anything not directly supported by the text.

Return a JSON object with exactly these keys: "decisions", "action_items", "deadlines", "open_questions". Each is a list of objects with fields:
- "text": a short, literal restatement of the item
- "owner": the person responsible, ONLY if explicitly named, else null
- "due_date_text": the exact phrase used for timing (e.g. "by Friday"), ONLY if actually stated, else null
- "line_ids": a list of integer IDs copied from the [L<id>] markers of lines that directly support this item

Classification rules:
- "action_items" = a specific person or group commits or is assigned to do a concrete thing.
- "decisions" = something the group explicitly agreed on.
- "deadlines" = a concrete action item or decision with a specifically stated date or timeframe.
- "open_questions" = an unresolved topic, suggestion, or proposal, including anything phrased as "we should discuss/consider", "maybe", or a question not answered here. These are not action items unless someone commits to a concrete task.
- If nothing in this excerpt clearly qualifies for a category, return an empty list for it. Empty lists are expected and fine.
- Cite only line IDs of transcript lines that directly support the item. Do not invent IDs.

Transcript excerpt:
{chunk_text}
"""


def extract_chunk(chunk_segments, model, max_retries=1, allow_split=True):
    """Returns (RawChunkResult, chunk_text, error_or_None)."""

    chunk_text = format_chunk_for_prompt(chunk_segments)

    if not chunk_text.strip():
        return RawChunkResult(), chunk_text, None

    prompt = PROMPT_TEMPLATE.format(chunk_text=chunk_text)

    last_error = None

    for _ in range(max_retries + 1):

        try:

            raw_response = call_ollama(prompt, model)
            data = json.loads(raw_response)

            if not isinstance(data, dict):
                last_error = f"Expected a JSON object, got {type(data).__name__}"
                continue

            # Sanitize before pydantic sees it: the model sometimes
            # puts a non-dict entry (e.g. a bare string) into one of
            # these lists. A single such entry previously invalidated
            # ALL items in that category for the whole chunk -- now
            # it's just dropped and counted, matching the per-item
            # handling parse_chunk_items() already does for
            # dict-shaped-but-wrong-fields items.
            for key in (
                "decisions",
                "action_items",
                "deadlines",
                "open_questions",
            ):

                value = data.get(key, [])

                if isinstance(value, list):
                    data[key] = [
                        entry for entry in value if isinstance(entry, dict)
                    ]
                else:
                    data[key] = []

            return RawChunkResult.model_validate(data), chunk_text, None

        except (json.JSONDecodeError, ValidationError) as error:
            last_error = str(error)
            continue

        except requests.Timeout as error:
            if allow_split and len(chunk_segments) > 1:
                midpoint = len(chunk_segments) // 2
                left, _, left_error = extract_chunk(
                    chunk_segments[:midpoint], model, max_retries,
                    allow_split=False,
                )
                right, _, right_error = extract_chunk(
                    chunk_segments[midpoint:], model, max_retries,
                    allow_split=False,
                )
                for category in (
                    "decisions", "action_items", "deadlines", "open_questions"
                ):
                    for item in getattr(right, category):
                        item["line_ids"] = [
                            line_id + midpoint
                            for line_id in item.get("line_ids", [])
                            if isinstance(line_id, int)
                        ]
                errors = [message for message in (left_error, right_error) if message]
                combined = RawChunkResult(
                    **{
                        category: getattr(left, category) + getattr(right, category)
                        for category in (
                            "decisions", "action_items", "deadlines", "open_questions"
                        )
                    }
                )
                return combined, chunk_text, "; ".join(errors) if errors else None
            return RawChunkResult(), chunk_text, str(error)
        except Exception as error:
            return RawChunkResult(), chunk_text, str(error)

    return RawChunkResult(), chunk_text, last_error


# ============================================================
# VALIDATE: ANTI-HALLUCINATION QUOTE CHECK
# ============================================================

def quote_is_grounded(quote, source_text, threshold=FUZZY_QUOTE_THRESHOLD):

    if not quote or not quote.strip():
        return False

    score = fuzz.partial_ratio(quote.strip(), source_text) / 100.0

    return score >= threshold


def meaningful_tokens(text):

    return {
        token.lower()
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2
    }


def raw_items_to_takeaways(raw_items, chunk_segments, drop_counter, line_id_map=None):

    kept = []

    for raw_item in raw_items:

        valid_line_ids = sorted({
            line_id for line_id in raw_item.line_ids
            if isinstance(line_id, int) and 0 <= line_id < len(chunk_segments)
        })

        if not valid_line_ids:
            drop_counter["dropped"] += 1
            continue

        cited_segments = [chunk_segments[line_id] for line_id in valid_line_ids]
        quote = " ".join(
            str(segment.get("text", "")).strip()
            for segment in cited_segments
            if str(segment.get("text", "")).strip()
        )
        item_tokens = meaningful_tokens(raw_item.text)
        cited_tokens = meaningful_tokens(quote)
        overlap = (
            len(item_tokens & cited_tokens) / len(item_tokens)
            if item_tokens else 0.0
        )
        fuzzy_score = fuzz.partial_ratio(raw_item.text, quote) / 100.0 if quote else 0.0

        if overlap < 0.3 and fuzzy_score < 0.5:
            drop_counter["dropped_sanity"] += 1
            continue

        takeaway = TakeawayItem(
                text=raw_item.text,
                owner=raw_item.owner,
                due_date_text=raw_item.due_date_text,
                evidence=Evidence(
                    speaker=str(cited_segments[0].get("speaker", "UNKNOWN")),
                    timestamp=float(cited_segments[0].get("start", 0)),
                    quote=quote,
                ),
            )
        kept.append(takeaway)
        if line_id_map is not None:
            line_id_map[id(takeaway)] = {
                (
                    round(float(segment.get("start", 0)), 3),
                    str(segment.get("speaker", "UNKNOWN")),
                    str(segment.get("text", "")).strip(),
                )
                for segment in cited_segments
            }
        drop_counter["kept"] += 1

    return kept


# ============================================================
# REDUCE: MERGE + DEDUPE ACROSS CHUNKS
# ============================================================

def dedupe_items(items, threshold=DEDUP_SIMILARITY_THRESHOLD):
    """Keeps the earliest-evidence version among near-duplicates
    (fuzzy text similarity >= threshold)."""

    kept = []

    for item in items:

        match_index = None

        for index, existing in enumerate(kept):

            if fuzz.token_sort_ratio(item.text, existing.text) >= threshold:
                match_index = index
                break

        if match_index is None:
            kept.append(item)

        elif item.evidence.timestamp < kept[match_index].evidence.timestamp:
            kept[match_index] = item

    return sorted(kept, key=lambda item: item.evidence.timestamp)


def normalize_item_text(text):

    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    normalized = re.sub(
        r"^(?:decide\s+on|decide\s+whether|consider|make\s+a\s+decision\s+on|"
        r"make\s+a\s+decision\s+about|discuss)\s+",
        "",
        normalized,
    )
    return normalized


def items_are_duplicates(left, right, line_id_map):

    left_lines = line_id_map.get(id(left), set())
    right_lines = line_id_map.get(id(right), set())
    if left_lines and right_lines:
        shared_fraction = len(left_lines & right_lines) / min(
            len(left_lines), len(right_lines)
        )
        if shared_fraction >= 0.5:
            return True

    left_text = normalize_item_text(left.text)
    right_text = normalize_item_text(right.text)
    return fuzz.token_sort_ratio(left_text, right_text) >= EVIDENCE_TEXT_SIMILARITY_THRESHOLD


def evidence_dedupe(decisions, action_items, open_questions, line_id_map):

    entries = (
        [("decision", item) for item in decisions]
        + [("action_item", item) for item in action_items]
        + [("open_question", item) for item in open_questions]
    )
    parents = list(range(len(entries)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        parents[find(right)] = find(left)

    for left in range(len(entries)):
        for right in range(left + 1, len(entries)):
            if items_are_duplicates(entries[left][1], entries[right][1], line_id_map):
                union(left, right)

    groups = {}
    for index, entry in enumerate(entries):
        groups.setdefault(find(index), []).append(entry)

    priority = {"decision": 0, "action_item": 1, "open_question": 2}
    output = {"decision": [], "action_item": [], "open_question": []}
    for group in groups.values():
        selected_type, selected = min(
            group,
            key=lambda entry: (
                priority[entry[0]],
                -len(meaningful_tokens(entry[1].text)),
                -len(entry[1].text),
            ),
        )
        merged_lines = set()
        for _, item in group:
            merged_lines.update(line_id_map.get(id(item), set()))
            selected.due_date_text = selected.due_date_text or item.due_date_text
            selected.due_date = selected.due_date or item.due_date
        line_id_map[id(selected)] = merged_lines
        output[selected_type].append(selected)

    for items in output.values():
        items.sort(key=lambda item: item.evidence.timestamp)
    return output["decision"], output["action_item"], output["open_question"]


VERIFICATION_PROMPT = """Check whether the candidate is supported by the cited transcript lines. Classify only what those lines establish. Return JSON only with exactly these fields: {{\"supported\": true or false, \"type\": \"decision\"|\"action_item\"|\"discussion\"|\"none\", \"reason\": \"at most 15 words\"}}.

Cited transcript lines:
{evidence}

Candidate ({candidate_type}): {candidate}
"""


def verify_takeaway(item, candidate_type, model):

    prompt = VERIFICATION_PROMPT.format(
        evidence=item.evidence.quote,
        candidate_type=candidate_type,
        candidate=item.text,
    )
    try:
        response = json.loads(call_ollama(prompt, model, num_predict=120))
        supported = response.get("supported") is True
        result_type = response.get("type", "none")
        if result_type not in {"decision", "action_item", "discussion", "none"}:
            result_type = "none"
        return supported, result_type
    except Exception:
        return False, "none"


def verify_items(decisions, action_items, open_questions, model, stats):

    verified = {"decision": [], "action_item": [], "open_question": list(open_questions)}
    for candidate_type, items in (("decision", decisions), ("action_item", action_items)):
        for item in items:
            stats["verification_total"] += 1
            supported, verified_type = verify_takeaway(item, candidate_type, model)
            if supported and verified_type == candidate_type:
                verified[candidate_type].append(item)
                stats["verification_kept"] += 1
            elif supported and verified_type == "discussion":
                verified["open_question"].append(item)
                stats["verification_reclassified"] += 1
            else:
                stats["verification_dropped"] += 1
    return verified["decision"], verified["action_item"], verified["open_question"]


def cap_open_questions(items, line_id_map, limit=8):

    ranked = sorted(
        items,
        key=lambda item: (
            -len(line_id_map.get(id(item), set())),
            item.evidence.timestamp,
        ),
    )
    return ranked[:limit]


def items_share_evidence(left, right, line_id_map=None):

    if line_id_map is not None:
        left_lines = line_id_map.get(id(left), set())
        right_lines = line_id_map.get(id(right), set())
        if left_lines and right_lines:
            if len(left_lines & right_lines) / min(len(left_lines), len(right_lines)) >= 0.5:
                return True

    if abs(left.evidence.timestamp - right.evidence.timestamp) < 0.01:
        return True

    left_tokens = meaningful_tokens(left.evidence.quote)
    right_tokens = meaningful_tokens(right.evidence.quote)
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
    return overlap >= 0.7 or fuzz.token_set_ratio(
        left.evidence.quote, right.evidence.quote
    ) >= 85


def merge_attached_deadlines(action_items, deadlines, line_id_map=None):
    """Attach deadlines with matching evidence to actions and return
    only deadlines that stand alone rather than describing an action."""

    standalone = []
    for deadline in deadlines:
        match = next(
            (
                item for item in action_items
                if items_share_evidence(item, deadline, line_id_map)
                or normalize_item_text(item.text) == normalize_item_text(deadline.text)
            ),
            None,
        )
        if match is None:
            standalone.append(deadline)
            continue
        match.due_date_text = deadline.due_date_text or match.due_date_text
        match.due_date = deadline.due_date or match.due_date
        if line_id_map is not None:
            line_id_map.setdefault(id(match), set()).update(
                line_id_map.get(id(deadline), set())
            )
    return standalone


UNRESOLVED_ITEM_PATTERN = re.compile(
    r"\b(discuss\w*|consider\w*|maybe|whether|could|possibly|suggest(?:ion)?|"
    r"option|proposal|should|decide whether|decide on|if agreed)\b",
    re.IGNORECASE,
)


def route_unresolved_items(decisions, action_items, open_questions):
    """Enforce the prompt's unresolved-topic categories when the model
    labels a suggestion or discussion point as a decision/action."""

    kept_decisions = []
    kept_actions = []
    for item in decisions:
        (open_questions if UNRESOLVED_ITEM_PATTERN.search(item.text)
         else kept_decisions).append(item)
    for item in action_items:
        (open_questions if UNRESOLVED_ITEM_PATTERN.search(item.text)
         else kept_actions).append(item)
    return kept_decisions, kept_actions, open_questions


# ============================================================
# RESOLVE RELATIVE DATES
# ============================================================

def resolve_due_date(due_date_text, meeting_date):

    if not due_date_text or dateparser is None:
        return None

    try:

        reference = datetime.strptime(meeting_date, "%Y-%m-%d")

        parsed = dateparser.parse(
            due_date_text,
            settings={"RELATIVE_BASE": reference},
        )

    except Exception:
        return None

    if parsed is None:
        return None

    return parsed.strftime("%Y-%m-%d")


# ============================================================
# SUMMARY (from existing chunk summaries, not a transcript re-read)
# ============================================================

SUMMARY_PROMPT_TEMPLATE = """Summarize what this meeting was about in 3-5 sentences, based only on the partial summaries below. Be factual and concise. Do not invent details not present below.

Partial summaries:
{joined_summaries}
"""


def generate_overall_summary(chunk_summaries, model):

    if not chunk_summaries:
        return ""

    joined = "\n".join(f"- {text}" for text in chunk_summaries if text)

    if not joined.strip():
        return ""

    prompt = SUMMARY_PROMPT_TEMPLATE.format(joined_summaries=joined)

    try:

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
            "keep_alive": OLLAMA_KEEP_ALIVE,
        }

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        return response.json().get("response", "").strip()

    except Exception:
        # Summary generation failing is non-fatal; the batch
        # T5 summary (already in report.summary) is kept instead.
        return ""


# ============================================================
# EXTRACTOR
# ============================================================

@register_extractor("llm")
class LLMExtractor(TakeawayExtractor):

    name = "llm"

    def __init__(self):
        self.last_run_stats = {}

    def is_available(self):
        return check_ollama_available()

    def extract(self, transcript_segments, meeting_date, chunk_summaries=None):

        ok, resolved_model_or_message = self.is_available()

        if not ok:
            raise RuntimeError(resolved_model_or_message)

        model = resolved_model_or_message

        chunks = chunk_transcript(transcript_segments)

        drop_counter = {"dropped": 0, "dropped_sanity": 0, "kept": 0}
        schema_drop_counter = {"dropped": 0}
        line_id_map = {}
        verification_stats = {
            "verification_total": 0,
            "verification_kept": 0,
            "verification_reclassified": 0,
            "verification_dropped": 0,
        }
        raw_item_count = 0

        all_decisions = []
        all_action_items = []
        all_deadlines = []
        all_open_questions = []

        chunk_errors = []
        failed_chunks = []

        for chunk_index, chunk_segments in enumerate(chunks):

            raw_result, _chunk_text, error = extract_chunk(
                chunk_segments, model
            )

            if error:
                chunk_errors.append(error)
                failed_chunks.append({
                    "chunk_index": chunk_index,
                    "line_count": len(chunk_segments),
                    "error": error,
                })

            for raw_dicts, bucket in (
                (raw_result.decisions, all_decisions),
                (raw_result.action_items, all_action_items),
                (raw_result.deadlines, all_deadlines),
                (raw_result.open_questions, all_open_questions),
            ):

                raw_item_count += len(raw_dicts)

                parsed_items = parse_chunk_items(
                    raw_dicts, schema_drop_counter
                )

                bucket += raw_items_to_takeaways(
                    parsed_items, chunk_segments, drop_counter, line_id_map
                )

        # Resolve due dates on action items and deadlines before dedup
        # (dedup compares .text, not due_date, so order doesn't matter
        # for correctness, but items need their final due_date set).
        for item in all_action_items + all_deadlines:

            if item.due_date_text:
                item.due_date = resolve_due_date(
                    item.due_date_text, meeting_date
                )

        decisions = all_decisions
        action_items = all_action_items
        deadlines = all_deadlines
        deadlines = merge_attached_deadlines(action_items, deadlines, line_id_map)
        deadlines = dedupe_items(deadlines, threshold=EVIDENCE_TEXT_SIMILARITY_THRESHOLD)
        open_questions = all_open_questions
        decisions, action_items, open_questions = route_unresolved_items(
            decisions, action_items, open_questions
        )
        decisions, action_items, open_questions = evidence_dedupe(
            decisions, action_items, open_questions, line_id_map
        )
        decisions, action_items, open_questions = verify_items(
            decisions, action_items, open_questions, model, verification_stats
        )
        decisions, action_items, open_questions = evidence_dedupe(
            decisions, action_items, open_questions, line_id_map
        )
        open_questions = cap_open_questions(open_questions, line_id_map)

        summary = generate_overall_summary(chunk_summaries or [], model)
        if failed_chunks:
            warning = (
                f"WARNING: {len(failed_chunks)} transcript chunk(s) failed; "
                "this report may be incomplete. See extractor_stats.failed_chunks."
            )
            summary = f"{warning}\n\n{summary}".strip()

        self.last_run_stats = {
            "model": model,
            "chunk_count": len(chunks),
            "raw_items": raw_item_count,
            "items_kept_before_dedup": drop_counter["kept"],
            "items_dropped_ungrounded": drop_counter["dropped"],
            "items_dropped_sanity": drop_counter["dropped_sanity"],
            "items_dropped_bad_schema": schema_drop_counter["dropped"],
            "chunk_errors": chunk_errors,
            "failed_chunks": failed_chunks,
            **verification_stats,
        }

        return {
            "decisions": decisions,
            "action_items": action_items,
            "deadlines": deadlines,
            "open_questions": open_questions,
            "summary": summary,
            "stats": self.last_run_stats,
        }
