"""Run the requested 600s live-mode simulation and compare speaker labels."""
import json
import math
import os
import resource
import sys
import time

import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ["LIVE_FULL_DIARIZATION"] = "0"
os.environ["LIVE_SPEAKER_SIMILARITY_THRESHOLD"] = "0.350"

from live_controller import LiveSessionController
from live_capture import CHUNK_SECONDS, STEP_SECONDS
from meeting_report import build_report, load_report

AUDIO_PATH = "data/audio/meeting.wav"
MAX_AUDIO_SECONDS = 600.0
OUTPUT_DIR = "outputs/live"
REPORT_ID = "live_online_embedding_20261020"
OFFLINE_LABELS_PATH = "outputs/transcripts/diarization.json"


def peak_rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def label_for(row, reference):
    overlaps = [
        (max(0.0, min(float(row["end"]), float(ref["end"])) - max(float(row["start"]), float(ref["start"]))), ref["speaker"])
        for ref in reference
    ]
    return max(overlaps)[1] if overlaps and max(overlaps)[0] > 0 else None


def agreement(transcript, reference):
    pairs = []
    for row in transcript:
        gold = label_for(row, reference)
        pred = row.get("speaker")
        if gold is not None and pred not in (None, "Speaker ?", "UNKNOWN"):
            pairs.append((str(pred), str(gold)))
    if not pairs:
        return "n/a", 0, 0
    pred_names = sorted({p for p, _ in pairs})
    gold_names = sorted({g for _, g in pairs})
    matrix = np.zeros((len(pred_names), len(gold_names)), dtype=np.int64)
    pmap = {name: i for i, name in enumerate(pred_names)}
    gmap = {name: i for i, name in enumerate(gold_names)}
    for pred, gold in pairs:
        matrix[pmap[pred], gmap[gold]] += 1
    rows, cols = linear_sum_assignment(-matrix)
    matched = int(matrix[rows, cols].sum())
    return f"{matched}/{len(pairs)} ({matched/len(pairs):.1%})", len(pred_names), len(pairs)


def proxy_der(transcript, offline_segments):
    try:
        from pyannote.core import Annotation, Segment, Timeline
        from pyannote.metrics.diarization import DiarizationErrorRate
        reference = Annotation()
        hypothesis = Annotation()
        for row in offline_segments:
            start, end = max(0.0, float(row["start"])), min(600.0, float(row["end"]))
            if end > start:
                reference[Segment(start, end)] = str(row["speaker"])
        for row in transcript:
            start, end = max(0.0, float(row["start"])), min(600.0, float(row["end"]))
            speaker = row.get("speaker")
            if end > start and speaker not in (None, "Speaker ?", "UNKNOWN"):
                hypothesis[Segment(start, end)] = str(speaker)
        metric = DiarizationErrorRate()
        value = metric(reference, hypothesis, uem=Timeline([Segment(0.0, 600.0)]))
        return f"{value:.1%} proxy DER"
    except Exception as error:
        return f"unavailable ({type(error).__name__})"


def main():
    print("Loading live models and starting online-embedding simulation.", flush=True)
    controller = LiveSessionController()
    controller.start(
        mode="simulated",
        simulated_file_path=AUDIO_PATH,
        simulated_max_duration_seconds=MAX_AUDIO_SECONDS,
    )
    controller._capture.join()
    stop_started = time.perf_counter()
    controller.stop()
    stop_report_elapsed_start = stop_started
    session = controller.session
    queued_at_stop = controller._chunk_queue.qsize()

    report_start = time.perf_counter()
    report = build_report(
        REPORT_ID,
        transcript_segments=list(session.aligned_transcript),
        chunk_summaries=list(session.chunk_summaries),
        final_summary=session.whole_summary,
        topics=session.topics_result.get("topics", []),
        source="live",
        meeting_date="2026-10-20",
        extractor_name="llm",
        pre_extract_hook=session.unload_models,
    )
    report_build_seconds = time.perf_counter() - report_start
    stop_to_report_seconds = time.perf_counter() - stop_report_elapsed_start
    load_report(REPORT_ID)

    with open(os.path.join(OUTPUT_DIR, "latency_log.jsonl"), encoding="utf-8") as file:
        entries = [json.loads(line) for line in file if line.strip()]
    offline_all = json.load(open(OFFLINE_LABELS_PATH, encoding="utf-8"))
    offline = [
        row for row in offline_all
        if float(row["start"]) < MAX_AUDIO_SECONDS and float(row["end"]) > 0
    ]
    audio_seconds = min(sf.info(AUDIO_PATH).duration, MAX_AUDIO_SECONDS)
    full_windows = max(0, math.floor((audio_seconds - CHUNK_SECONDS) / STEP_SECONDS) + 1)
    expected_windows = full_windows + (1 if audio_seconds - full_windows * STEP_SECONDS > 1e-6 else 0)
    processed_windows = min(expected_windows, len(entries) + controller.merged_chunk_count)
    lag = max((float(row.get("end_to_end_lag_seconds", 0.0)) for row in entries), default=0.0)
    rtf = [float(row.get("real_time_factor", 0.0)) for row in entries]
    mean_rtf = float(np.mean(rtf)) if rtf else 0.0
    median_rtf = float(np.median(rtf)) if rtf else 0.0
    p90_rtf = float(np.percentile(rtf, 90)) if rtf else 0.0
    emb_values = []
    for row in entries:
        duration = float(row.get("chunk_duration_s", 20.0))
        windows = 1 if duration <= 20.01 else max(1, round((duration - 3.0) / 17.0))
        emb_values.append(float(row.get("embedding_clustering_ms", 0.0)) / windows)
    emb_mean = float(np.mean(emb_values)) if emb_values else 0.0
    emb_p90 = float(np.percentile(emb_values, 90)) if emb_values else 0.0

    baseline = load_report("live_latency_baseline_20261019")
    async_report = load_report("live_latency_after_20261020")
    online_transcript = list(session.aligned_transcript)
    baseline_agreement, baseline_spk, _ = agreement(baseline.transcript, offline)
    async_agreement, async_spk, _ = agreement(async_report.transcript, offline)
    online_agreement, online_spk, _ = agreement(online_transcript, offline)
    offline_speakers = len({row["speaker"] for row in offline})
    der_base = proxy_der(baseline.transcript, offline)
    der_async = proxy_der(async_report.transcript, offline)
    der_online = proxy_der(online_transcript, offline)

    rows = [
        ("Windows processed / dropped / merged / queued", "17 / 16 / 0 / 3", "36 / 0 / 21 / 0", f"{processed_windows} / {controller.dropped_chunk_count} / {controller.merged_chunk_count} / {queued_at_stop}"),
        ("RTF mean / median / p90", "2.104 / 0.627 / 5.994", "3.020 / 0.887 / 8.199", f"{mean_rtf:.3f} / {median_rtf:.3f} / {p90_rtf:.3f}"),
        ("Maximum transcript lag", "287.50s", "543.10s", f"{lag:.2f}s"),
        ("Stop-to-report time", "81.04s", "581.00s", f"{stop_to_report_seconds:.2f}s"),
        ("Peak RSS", "942.9 MB", "945.7 MB", f"{peak_rss_mb():.1f} MB"),
        ("Embedding + clustering per window (mean / p90)", "n/a", "n/a", f"{emb_mean:.1f} / {emb_p90:.1f} ms"),
        ("Agreement vs offline after optimal ID mapping", f"{baseline_agreement}", f"{async_agreement}", f"{online_agreement}"),
        ("Speakers found / offline speakers", f"{baseline_spk} / {offline_speakers}", f"{async_spk} / {offline_speakers}", f"{online_spk} / {offline_speakers}"),
        ("Offline-reference DER (proxy)", der_base, der_async, der_online),
    ]
    lines = [
        "# Live online speaker-embedding results", "",
        f"- Simulation: `{AUDIO_PATH}`, capped at 600s; threshold tuned on the first 180s against offline pyannote labels: **0.350 cosine** (25/29 segments, 86.2%, two speakers; selected the calibrated plateau value closest to the configured default).",
        f"- Online run processed {processed_windows}/{expected_windows} source windows; {controller.dropped_chunk_count} dropped, {controller.merged_chunk_count} merged, {queued_at_stop} queued at Stop.",
        f"- Per-window embedding + clustering: mean {emb_mean:.1f} ms, p90 {emb_p90:.1f} ms (batch timing normalized by merged source-window count).",
        f"- Live embedding model: cached pyannote pipeline's internal `_embedding`; no model download. Online centroids are L2-normalized, EMA 0.9, max 8 speakers. Full live re-diarization disabled; recorded/offline pipeline unchanged.",
        "- DER, when available, uses the offline pyannote output as a reference proxy. It is not ground-truth DER.", "",
        "| Metric | Baseline | Async-PyAnnote | Online-Embedding |", "|---|---:|---:|---:|",
    ]
    lines.extend(f"| {name} | {base} | {async_value} | {online} |" for name, base, async_value, online in rows)
    lines.extend([
        "",
        f"- Online DER proxy: {der_online}.",
        f"- Online max end-to-end transcript lag: {lag:.2f}s; dropped windows: {controller.dropped_chunk_count}. Real-time claim: {'yes' if controller.dropped_chunk_count == 0 and lag < 30 else 'no'} (requires zero drops and bounded lag below 30s).",
        f"- Live report validation: passed (`outputs/reports/{REPORT_ID}/report.json`); report build {report_build_seconds:.2f}s.",
        f"- Saved audio for optional offline refinement: `{controller.session_audio_path}`.",
    ])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "online_report.md"), "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    print(f"ONLINE_REPORT=outputs/live/online_report.md", flush=True)
    print(f"WINDOWS={processed_windows}/{expected_windows}; DROPPED={controller.dropped_chunk_count}; MERGED={controller.merged_chunk_count}; LAG={lag:.2f}s", flush=True)


if __name__ == "__main__":
    main()
