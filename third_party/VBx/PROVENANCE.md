# VBx — Provenance and Licensing

This directory contains a **partial vendor copy** of the official VBx
speaker diarization toolkit, used as an experimental alternative to
PyAnnote in MeetSense for research comparison purposes only.

## Source

- Repository: https://github.com/BUTSpeechFIT/VBx
- Authors: Lukáš Burget, Mireia Diez, Federico Landini, Jan Profant
  (Brno University of Technology, Speech@FIT / Phonexia)
- Pinned commit: `57466e6e245d5cdfe2e88ee6503702ace3ffdd03`
  (fetched 2026-09-19; pinned rather than tracking `master` so this
  experiment stays reproducible even if the upstream repo changes)
- Primary publication: F. Landini, J. Profant, M. Diez, L. Burget,
  "Bayesian HMM clustering of x-vector sequences (VBx) in speaker
  diarization: theory, implementation and analysis on standard tasks,"
  Computer Speech & Language, 2022.
  Also on arXiv: https://arxiv.org/abs/2012.14952

## License

Apache License, Version 2.0. Every vendored `.py` file retains its
original license header from the upstream repository, unmodified.
`requirements.txt`, `setup.py`, and `README.md` are also vendored
as-is for reference.

## What was vendored, and why only this subset

Only the files needed for **inference** (not training) with the
already-pretrained ResNet101/16kHz checkpoint were copied:

- `VBx/VBx.py`, `VBx/__init__.py`, `VBx/diarization_lib.py`,
  `VBx/features.py`, `VBx/kaldi_utils.py`, `VBx/predict.py`,
  `VBx/vbhmm.py` -- the core inference pipeline (x-vector extraction
  via `predict.py`, then AHC+VB clustering via `vbhmm.py`).
- `VBx/models/resnet.py` -- the model architecture definition needed
  by `predict.py`.
- `VBx/models/ResNet101_16kHz/nnet/final.onnx` (~58MB) -- the
  pretrained x-vector extractor, ONNX format, 16kHz variant (matches
  MeetSense's audio sample rate exactly, no resampling needed).
- `VBx/models/ResNet101_16kHz/plda` and `.../transform.h5` -- the
  matching pretrained PLDA and LDA/mean-normalization parameters.

**Not vendored** (not needed for inference with the pretrained
checkpoint): `raw_81.pth.zip.part*` (a split PyTorch training
checkpoint, ~126MB combined -- only relevant if retraining the
extractor, which this project does not do), the `dscore` and
`AMI-diarization-setup` git submodules (an alternative RTTM scorer and
a benchmark dataset setup -- MeetSense already has its own DER
evaluation via `pyannote.metrics`, reused for VBx too, and does not
use the AMI benchmark), and the dataset-specific run scripts
(`AMI_run.sh`, `CALLHOME_run.sh`, `DIHARD2_run.sh`).

## Not modified

None of the vendored `.py` files were edited. MeetSense's own
integration code (`src/diarization_vbx.py`) calls these scripts as
subprocesses, exactly as `run_example.sh` in the original repo does,
rather than patching them.

## Hyperparameters used

The clustering hyperparameters used by MeetSense's VBx integration are
taken directly from the official `run_example.sh` in the pinned
commit -- the authors' own general-purpose example recipe, not
independently tuned against MeetSense's recording (tuning against a
single evaluation sample would not be a defensible research practice):

```
--init AHC+VB --threshold -0.015 --lda-dim 128 --Fa 0.3 --Fb 17 --loopP 0.99
```
