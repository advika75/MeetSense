"""
Reproducible diarization backend selector.

Runs diarization with either the production PyAnnote pipeline or the
experimental VBx pipeline, then scores DER for whichever was run,
using the same reference file and evaluation window for both.

This script does not implement diarization itself -- it just invokes
the existing, independent scripts as subprocesses (the same pattern
src/run_pipeline.py already uses), so each backend continues to work
exactly as it does when run directly.

Usage:

    python run_diarization_experiment.py --method pyannote
    python run_diarization_experiment.py --method vbx
    python run_diarization_experiment.py --method both

Outputs (never overwriting the other method's files):

    pyannote -> outputs/transcripts/diarization.json
                outputs/evaluations/der_results.json
    vbx      -> outputs/transcripts/diarization_vbx.json
                outputs/evaluations/der_vbx.json
"""

import argparse
import subprocess
import sys


STEPS = {
    "pyannote": [
        ("PyAnnote diarization", "src/diarization.py"),
        ("PyAnnote DER evaluation", "src/der_evaluation.py"),
    ],
    "vbx": [
        ("VBx diarization (experimental)", "src/diarization_vbx.py"),
        ("VBx DER evaluation (experimental)", "src/der_evaluation_vbx.py"),
    ],
}


def run_method(method):

    print("=" * 70)
    print(f"DIARIZATION EXPERIMENT: {method}")
    print("=" * 70)

    for step_name, script in STEPS[method]:

        print()
        print("-" * 70)
        print(step_name)
        print("-" * 70)

        result = subprocess.run([sys.executable, script], check=False)

        if result.returncode != 0:

            print()
            print(f"FAILED: {step_name} (exit code {result.returncode})")
            print(f"Script: {script}")

            sys.exit(result.returncode)

    print()
    print(f"✓ {method} diarization experiment complete.")


def main():

    parser = argparse.ArgumentParser(
        description="Run a diarization backend and score its DER."
    )

    parser.add_argument(
        "--method",
        required=True,
        choices=["pyannote", "vbx", "both"],
        help="Which diarization backend to run.",
    )

    args = parser.parse_args()

    methods = ["pyannote", "vbx"] if args.method == "both" else [args.method]

    for method in methods:
        run_method(method)

    print()
    print("=" * 70)
    print("EXPERIMENT RUNNER COMPLETE")
    print("=" * 70)

    if "pyannote" in methods:
        print("PyAnnote: outputs/transcripts/diarization.json, "
              "outputs/evaluations/der_results.json")

    if "vbx" in methods:
        print("VBx:      outputs/transcripts/diarization_vbx.json, "
              "outputs/evaluations/der_vbx.json")


if __name__ == "__main__":
    main()
