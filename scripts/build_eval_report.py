"""
Builds outputs/reports/<meeting_id>_<extractor>/report.json for one
extractor from the saved speaker transcript, and prints wall-clock
runtime plus peak RAM (this process and the Ollama server, sampled
from `ps`). Used to produce the inputs to scripts/eval_takeaways.py.

Usage:
    python scripts/build_eval_report.py hybrid [--suffix _run2]
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import meeting_report  # noqa: E402


def ollama_rss_mb():
    out = subprocess.run(
        ["ps", "-axo", "rss=,comm="], capture_output=True, text=True
    ).stdout
    return sum(
        int(line.split(None, 1)[0])
        for line in out.splitlines()
        if "ollama" in line.lower()
    ) / 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("extractor")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--meeting-id", default="meeting_001")
    parser.add_argument("--date", default="2026-10-19")
    parser.add_argument(
        "--transcript", default="outputs/transcripts/speaker_transcript.json"
    )
    args = parser.parse_args()

    peak = {"ollama": 0.0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak["ollama"] = max(peak["ollama"], ollama_rss_mb())
            time.sleep(0.5)

    threading.Thread(target=sample, daemon=True).start()

    if args.extractor == "llm_verify":
        os.environ["LLM_VERIFY"] = "1"

    start = time.time()
    report = meeting_report.build_report(
        meeting_id=f"{args.meeting_id}_{args.extractor}{args.suffix}",
        meeting_date=args.date,
        extractor_name=args.extractor,
        transcript_path=args.transcript,
    )
    elapsed = time.time() - start
    stop.set()

    own_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

    print(json.dumps({
        "extractor_used": report.extractor_used,
        "runtime_sec": round(elapsed, 1),
        "peak_rss_mb_python": round(own_peak, 1),
        "peak_rss_mb_ollama": round(peak["ollama"], 1),
        "decisions": len(report.decisions),
        "action_items": len(report.action_items),
        "deadlines": len(report.deadlines),
        "open_questions": len(report.open_questions),
        "stats": report.extractor_stats,
    }, indent=2))


if __name__ == "__main__":
    main()
