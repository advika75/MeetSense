"""Run a capped simulated live session and write its latency summary."""

import json
import math
import os
import resource
import sys
import time

import numpy as np
import soundfile as sf

_SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from live_controller import LiveSessionController
from meeting_report import load_report, build_report
from live_capture import CHUNK_SECONDS, STEP_SECONDS


AUDIO_PATH = "data/audio/meeting.wav"
MAX_AUDIO_SECONDS = 600.0
OUTPUT_DIR = "outputs/live"
REPORT_ID = "live_latency_baseline_20261019"


def rss_peak_mb():

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def summarize():

    print("Loading the live controller and models.", flush=True)
    controller = LiveSessionController()
    controller.start(
        mode="simulated",
        simulated_file_path=AUDIO_PATH,
        simulated_max_duration_seconds=MAX_AUDIO_SECONDS,
    )

    capture = controller._capture
    print("Simulated capture running, capped at 600 seconds.", flush=True)
    capture.join()

    # Match the UI Stop action: stop capture, wait for the in-flight
    # pipeline chunk, then unload live models in build_report's hook.
    print("Audio cap reached; invoking controller.stop().", flush=True)
    controller.stop()
    queued_at_stop = controller._chunk_queue.qsize()
    session = controller.session
    print(
        f"Captured={controller.captured_audio_seconds:.1f}s; "
        f"processed={controller.processed_audio_seconds:.1f}s; "
        f"chunks={session.chunk_count}; dropped={controller.dropped_chunk_count}",
        flush=True,
    )

    report_start = time.perf_counter()
    report = build_report(
        REPORT_ID,
        transcript_segments=list(session.aligned_transcript),
        chunk_summaries=list(session.chunk_summaries),
        final_summary=session.whole_summary,
        topics=session.topics_result.get("topics", []),
        source="live",
        meeting_date="2026-10-19",
        extractor_name="llm",
        pre_extract_hook=session.unload_models,
    )
    report_build_seconds = time.perf_counter() - report_start

    try:
        load_report(REPORT_ID)
        report_validated = True
    except Exception as error:
        report_validated = False
        print(f"Report validation error: {error}", flush=True)

    log_path = os.path.join(OUTPUT_DIR, "latency_log.jsonl")
    with open(log_path, encoding="utf-8") as file:
        entries = [json.loads(line) for line in file if line.strip()]

    def percentile_stats(values):
        if not values:
            return 0.0, 0.0, 0.0
        return (
            float(np.mean(values)),
            float(np.median(values)),
            float(np.percentile(values, 90)),
        )

    stage_specs = [
        ("VAD", "vad_ms"),
        ("ASR", "asr_ms"),
        ("Diarization", "diarization_ms"),
        ("Alignment + NLP", "alignment_nlp_ms"),
        ("Total", "total_ms"),
        ("RTF", "real_time_factor"),
    ]

    audio_seconds = min(sf.info(AUDIO_PATH).duration, MAX_AUDIO_SECONDS)
    full_windows = (
        max(0, math.floor((audio_seconds - CHUNK_SECONDS) / STEP_SECONDS) + 1)
        if audio_seconds >= CHUNK_SECONDS else 0
    )
    trailing_audio = audio_seconds - full_windows * STEP_SECONDS
    expected_windows = full_windows + (1 if trailing_audio > 1e-6 else 0)
    not_processed = max(0, expected_windows - len(entries))

    lines = [
        "# Live latency baseline",
        "",
        f"- Audio: `{AUDIO_PATH}`; simulated cap: {MAX_AUDIO_SECONDS:.0f}s.",
        "- Live UI chunk settings: 20s chunks, 3s overlap, 17s step; "
        "default three-chunk processing queue.",
        f"- Processed chunks: {len(entries)}; captured audio: "
        f"{controller.captured_audio_seconds:.1f}s; processed endpoint: "
        f"{controller.processed_audio_seconds:.1f}s; dropped chunks: "
        f"{controller.dropped_chunk_count}.",
        f"- Expected capture windows: {expected_windows}; not processed: "
        f"{not_processed} (queue overflow: {controller.dropped_chunk_count}; "
        f"queued when Stop exited: {queued_at_stop}).",
        "",
        "| Stage | Mean ms | Median ms | P90 ms |",
        "|---|---:|---:|---:|",
    ]

    for label, key in stage_specs:
        mean, median, p90 = percentile_stats(
            [float(entry.get(key, 0.0)) for entry in entries]
        )
        if label == "RTF":
            lines.append(
                f"| RTF | {mean:.3f} | {median:.3f} | {p90:.3f} |"
            )
        else:
            lines.append(
                f"| {label} | {mean:.1f} | {median:.1f} | {p90:.1f} |"
            )

    under_one = sum(
        float(entry.get("real_time_factor", 0.0)) < 1.0
        for entry in entries
    )
    fraction = under_one / len(entries) if entries else 0.0
    max_lag = max(
        (float(entry.get("end_to_end_lag_seconds", 0.0)) for entry in entries),
        default=0.0,
    )
    peak_rss = rss_peak_mb()
    lines.extend([
        "",
        f"- Chunks with RTF < 1.0: {under_one}/{len(entries)} "
        f"({fraction:.1%}).",
        f"- Maximum end-to-end lag (chunk end to result): {max_lag:.2f}s.",
        f"- Stop-to-report build time, including live-model unload: "
        f"{report_build_seconds:.2f}s.",
        f"- Peak process RSS: {peak_rss:.1f} MB.",
        f"- `report.json` validation: {'passed' if report_validated else 'FAILED'} "
        f"(`outputs/reports/{REPORT_ID}/report.json`).",
        f"- Report extractor: `{report.extractor_used}`.",
        f"- RTF < 1.0 for most chunks: {'yes' if fraction > 0.5 else 'no'} "
        f"({under_one}/{len(entries)} chunks).",
    ])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "baseline_report.md"), "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print(f"REPORT_BUILD_SECONDS={report_build_seconds:.2f}", flush=True)
    print(f"PEAK_RSS_MB={peak_rss:.1f}", flush=True)
    print(f"REPORT_VALIDATED={report_validated}", flush=True)
    print(f"LOG_CHUNKS={len(entries)}", flush=True)
    print(f"REPORT_ID={report.meeting_id}", flush=True)
    print(f"BASELINE=outputs/live/baseline_report.md", flush=True)


if __name__ == "__main__":
    summarize()
