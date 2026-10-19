"""
VBx speaker diarization -- EXPERIMENTAL alternative diarization backend,
added purely for research comparison against the production PyAnnote
pipeline (src/diarization.py, left completely untouched).

VBx is NOT an ASR model. It is a speaker-embedding-based diarization
method: sliding-window x-vector (speaker embedding) extraction over a
pretrained ResNet101, followed by agglomerative clustering (AHC) as an
initialization for Variational Bayes HMM (VBx) inference -- a Bayesian
generative model over the sequence of x-vectors that refines the
cluster (speaker) assignment. See:

  F. Landini, J. Profant, M. Diez, L. Burget, "Bayesian HMM clustering
  of x-vector sequences (VBx) in speaker diarization: theory,
  implementation and analysis on standard tasks," Computer Speech &
  Language, 2022. https://arxiv.org/abs/2012.14952

Implementation source: the official BUTSpeechFIT/VBx repository,
Apache 2.0 licensed, vendored (inference files only, unmodified) under
third_party/VBx/ -- see third_party/VBx/PROVENANCE.md for the exact
pinned commit, what was and wasn't vendored, and why.

--------------------------------------------------------------------
Pipeline (mirrors the official run_example.sh exactly):

  1. Convert MeetSense's existing Silero VAD output
     (outputs/vad/speech_segments.json) into the .lab format VBx's
     x-vector extractor expects. VBx does not do its own VAD -- it
     requires externally supplied speech segments.
  2. VBx/predict.py: sliding-window x-vector extraction over the
     speech regions, using the pretrained ResNet101/16kHz ONNX model.
  3. VBx/vbhmm.py: AHC initialization + VB-HMM refinement over the
     x-vector sequence, producing an RTTM file.
  4. Convert that RTTM into MeetSense's standard diarization JSON
     shape ({"start", "end", "speaker"}), matching what
     src/diarization.py produces, so the existing alignment
     (merge_transcript.py) and evaluation (pyannote.metrics-based DER)
     code can consume it unmodified.

IMPORTANT, stated here explicitly rather than glossed over: because
VBx requires external VAD and PyAnnote's pipeline performs its own
internal segmentation, the two systems compared by this experiment do
NOT receive identical segmentation input. This is a genuine
architectural difference between an end-to-end pipeline (PyAnnote) and
a modular one (VBx), not a bug in this integration -- and it should be
stated as a limitation in any write-up of the comparison, not hidden.

Output: outputs/transcripts/diarization_vbx.json -- a NEW file. The
production outputs/transcripts/diarization.json (PyAnnote) is never
read from or written to by this script.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

AUDIO_DIR = BASE_DIR / "data" / "audio"
AUDIO_FILE = AUDIO_DIR / "meeting.wav"

VAD_SEGMENTS_FILE = BASE_DIR / "outputs" / "vad" / "speech_segments.json"

VBX_ROOT = BASE_DIR / "third_party" / "VBx" / "VBx"
VBX_MODEL_DIR = VBX_ROOT / "models" / "ResNet101_16kHz"

WORK_DIR = BASE_DIR / "outputs" / "vbx"

OUTPUT_FILE = BASE_DIR / "outputs" / "transcripts" / "diarization_vbx.json"

RECORDING_NAME = "meeting"

# Hyperparameters copied verbatim from VBx's own official
# run_example.sh (pinned commit, see PROVENANCE.md) -- the authors'
# general-purpose example recipe. NOT tuned against this recording:
# tuning clustering hyperparameters against the one sample being
# evaluated would not be a defensible research practice.
VBX_PARAMS = {
    "init": "AHC+VB",
    "threshold": "-0.015",
    "lda_dim": "128",
    "Fa": "0.3",
    "Fb": "17",
    "loopP": "0.99",
}


def convert_vad_to_lab():
    """Writes outputs/vad/speech_segments.json out as a VBx-format
    .lab file (whitespace-separated "start end label" lines, the same
    format used by VBx's own example/vad/*.lab files)."""

    with open(VAD_SEGMENTS_FILE, "r", encoding="utf-8") as f:
        vad_segments = json.load(f)

    lab_path = WORK_DIR / f"{RECORDING_NAME}.lab"

    with open(lab_path, "w", encoding="utf-8") as f:
        for segment in vad_segments:
            f.write(
                f"{segment['start']:.3f} {segment['end']:.3f} sp\n"
            )

    return lab_path


def run_step(script_name, args, step_name):
    """Runs a vendored VBx script as a subprocess, exactly like
    run_pipeline.py already does for MeetSense's own pipeline steps.
    cwd is set to VBX_ROOT because predict.py imports
    'from models.resnet import *' -- a path relative to VBx/'s own
    directory, matching how the original run_example.sh invokes it
    ("python VBx/predict.py ..." from the repo root)."""

    print(f"--- {step_name} ---")

    result = subprocess.run(
        [sys.executable, script_name] + args,
        cwd=str(VBX_ROOT),
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:

        print(result.stderr)

        raise RuntimeError(
            f"{step_name} failed (exit code {result.returncode}). "
            f"See stderr above for the exact error."
        )


def extract_xvectors(lab_path):

    file_list_path = WORK_DIR / "list.txt"
    file_list_path.write_text(RECORDING_NAME + "\n", encoding="utf-8")

    run_step(
        "predict.py",
        [
            "--in-file-list", str(file_list_path),
            "--in-lab-dir", str(WORK_DIR),
            "--in-wav-dir", str(AUDIO_DIR),
            "--out-ark-fn", str(WORK_DIR / f"{RECORDING_NAME}.ark"),
            "--out-seg-fn", str(WORK_DIR / f"{RECORDING_NAME}.seg"),
            "--weights", str(VBX_MODEL_DIR / "nnet" / "final.onnx"),
            "--backend", "onnx",
        ],
        "VBx x-vector extraction (predict.py)",
    )


def run_vbhmm():

    run_step(
        "vbhmm.py",
        [
            "--init", VBX_PARAMS["init"],
            "--out-rttm-dir", str(WORK_DIR),
            "--xvec-ark-file", str(WORK_DIR / f"{RECORDING_NAME}.ark"),
            "--segments-file", str(WORK_DIR / f"{RECORDING_NAME}.seg"),
            "--xvec-transform", str(VBX_MODEL_DIR / "transform.h5"),
            "--plda-file", str(VBX_MODEL_DIR / "plda"),
            "--threshold", VBX_PARAMS["threshold"],
            "--lda-dim", VBX_PARAMS["lda_dim"],
            "--Fa", VBX_PARAMS["Fa"],
            "--Fb", VBX_PARAMS["Fb"],
            "--loopP", VBX_PARAMS["loopP"],
        ],
        "VBx AHC+VB-HMM clustering (vbhmm.py)",
    )


def rttm_to_segments():
    """Parses the RTTM vbhmm.py wrote and converts it into MeetSense's
    standard diarization JSON shape. RTTM fields (from vbhmm.py's own
    write_output()):
    SPEAKER <file> 1 <start> <duration> <NA> <NA> <label> <NA> <NA>
    """

    rttm_path = WORK_DIR / f"{RECORDING_NAME}.rttm"

    if not rttm_path.exists():
        raise FileNotFoundError(
            f"Expected VBx output not found: {rttm_path}"
        )

    segments = []

    with open(rttm_path, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.split()

            if not parts or parts[0] != "SPEAKER":
                continue

            start = float(parts[3])
            duration = float(parts[4])
            label = int(parts[7]) - 1  # vbhmm.py writes label + 1

            segments.append({
                "start": round(start, 3),
                "end": round(start + duration, 3),
                "speaker": f"SPEAKER_{label:02d}",
            })

    segments.sort(key=lambda s: s["start"])

    return segments


def main():

    print("=" * 60)
    print("VBx SPEAKER DIARIZATION (experimental)")
    print("=" * 60)

    if not AUDIO_FILE.exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_FILE}")

    if not VAD_SEGMENTS_FILE.exists():
        raise FileNotFoundError(
            f"VAD segments not found: {VAD_SEGMENTS_FILE}. "
            "Run src/vad.py first (VBx requires external VAD input)."
        )

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("Converting Silero VAD segments to VBx .lab format...")
    lab_path = convert_vad_to_lab()
    print(f"  Wrote: {lab_path}")

    print()
    extract_xvectors(lab_path)

    print()
    run_vbhmm()

    print()
    print("Converting RTTM output to MeetSense diarization JSON format...")
    segments = rttm_to_segments()

    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    speakers = sorted(set(s["speaker"] for s in segments))

    print()
    print("=" * 60)
    print("VBx DIARIZATION COMPLETE")
    print("=" * 60)
    print()
    print("Number of speaker segments:", len(segments))
    print("Number of speakers:", len(speakers))
    print("Detected speakers:", ", ".join(speakers))
    print()
    print("Output saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
