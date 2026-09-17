# MeetSense

### Speaker-Aware Meeting Transcription and Summarization System

MeetSense turns a recorded meeting into a speaker-attributed transcript, a
transformer-generated summary, extracted keywords/topics, rule-based
action items and decisions, and a quantitative evaluation of every stage —
all served through an interactive Streamlit interface.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Key Features](#key-features)
5. [System Architecture](#system-architecture)
6. [Technologies Used](#technologies-used)
7. [Deep Learning Components](#deep-learning-components)
8. [NLP Components](#nlp-components)
9. [Complete Pipeline](#complete-pipeline)
10. [Project Structure](#project-structure)
11. [Installation](#installation)
12. [Environment Setup](#environment-setup)
13. [Hugging Face Token Configuration](#hugging-face-token-configuration)
14. [FFmpeg Requirement](#ffmpeg-requirement)
15. [Running the Offline Pipeline](#running-the-offline-pipeline)
16. [Running MeetSense (the App)](#running-meetsense-the-app)
17. [Running Evaluation](#running-evaluation)
18. [Running Experiments](#running-experiments)
19. [Evaluation Metrics](#evaluation-metrics)
20. [Actual Results](#actual-results)
21. [Model Comparison](#model-comparison)
22. [Ablation Study](#ablation-study)
23. [Error Analysis](#error-analysis)
24. [Limitations](#limitations)
25. [Future Work](#future-work)
26. [Academic Contribution](#academic-contribution)
27. [Conclusion](#conclusion)

---

## Project Overview

MeetSense is an end-to-end system that processes a recorded meeting and
produces:

- a speaker-attributed transcript,
- keyword and topic summaries,
- rule-based action items and decisions,
- an abstractive summary (transformer-generated),
- and a quantitative evaluation of transcription accuracy, diarization
  accuracy, summarization quality, and action-item extraction quality.

It integrates several **pretrained** deep learning models (Silero VAD,
Whisper, PyAnnote, T5-small) with **traditional NLP** techniques
(TF-IDF, LDA, NLTK-based tokenization/POS/NER) and **rule-based**
information extraction, and wraps the result in a Streamlit dashboard
(MeetSense).

## Problem Statement

Manually reviewing meeting recordings to produce a transcript, identify who
said what, extract action items, and write a summary is slow and
error-prone. Existing transcription tools frequently omit speaker
attribution, and generic summarizers are not tuned for multi-speaker,
disfluent, conversational audio. This project investigates whether an
end-to-end pipeline built from currently-available pretrained speech and
NLP models can produce a usable speaker-aware transcript, summary, and
action-item list from a real meeting recording — and, just as importantly,
quantifies **where and how much** that pipeline actually fails, rather
than only demonstrating a working demo.

## Objectives

1. Build a working pipeline: audio → speech detection → transcription →
   speaker diarization → speaker-aligned transcript → NLP analysis →
   summarization.
2. Quantitatively evaluate each measurable stage (WER, DER, ROUGE,
   action-item precision/recall/F1) against a manually-verified reference.
3. Compare feasible model/component alternatives already available in the
   environment (Whisper `base` vs `small`; T5-small vs a non-neural
   extractive baseline).
4. Run an ablation study isolating the contribution of VAD and
   diarization.
5. Perform structured, evidence-grounded error analysis per stage rather
   than only reporting aggregate scores.
6. Surface all of the above inside a usable interactive dashboard.

## Key Features

- Speaker-aware transcript with per-segment speaker labels and timestamps.
- TF-IDF keyword extraction (overall and per-speaker).
- LDA topic modeling (5 topics) with per-segment topic assignment.
- Rule-based action-item and decision extraction with deadline detection.
- T5-small abstractive meeting summarization.
- Quantitative evaluation: WER, DER, ROUGE-1/2/L, action-item P/R/F1.
- Model comparison (ASR model size, summarization approach).
- Ablation study across the pipeline's major stages.
- Structured, code-grounded error analysis across five categories.
- Interactive Streamlit workspace (MeetSense) presenting all of the above.
- Standalone microphone-based live-transcription utilities
  (`src/live_meeting.py`, `src/live_transcribe.py`,
  `src/live_speaker_transcribe.py`) that write a JSON file the workspace's
  Live Meeting page can display.

## System Architecture

```
Meeting Audio (data/audio/meeting.wav)
        │
        ▼
Audio Preprocessing        (resample to 16kHz mono)
        │
        ▼
Silero VAD                 (speech/silence segmentation)
        │
        ▼
Whisper ASR                (speech-to-text, timestamped segments)
        │
        ▼
PyAnnote Speaker Diarization (who spoke when)
        │
        ▼
Timestamp Alignment        (merge ASR segments with speaker turns)
        │
        ▼
Speaker-Aware Transcript
        │
        ▼
NLP Preprocessing          (tokenize, POS-tag, lemmatize, NER)
        │
        ▼
TF-IDF Keywords   +   LDA Topic Modeling
        │
        ▼
Action Items / Decisions   (rule-based extraction)
        │
        ▼
T5-small Summarization      (abstractive)
        │
        ▼
Evaluation  (WER, DER, ROUGE, Action-item P/R/F1, Model Comparison,
             Ablation Study, Error Analysis)
        │
        ▼
MeetSense Dashboard (Streamlit)
```

### Purpose of each stage

| Stage | Purpose |
|---|---|
| Audio preprocessing | Normalize the raw recording to 16kHz mono PCM, the sample rate Silero VAD and Whisper expect. |
| Silero VAD | Detect speech vs. silence regions; produces a speech-only audio track and segment timestamps. |
| Whisper ASR | Convert speech to timestamped text segments. In production, this runs on the original raw audio (see [Limitations](#limitations) for why VAD's output isn't currently wired into it). |
| PyAnnote diarization | Assign a speaker label to time segments of the audio, independent of transcription. |
| Timestamp alignment | Merge Whisper's text segments with PyAnnote's speaker turns by temporal overlap, producing one speaker-labeled transcript. |
| NLP preprocessing | Sentence/word tokenization, stopword removal, lemmatization, POS tagging, and named-entity chunking (NLTK) over the transcript. |
| TF-IDF keywords | Rank terms (overall and per speaker) by TF-IDF weight. |
| LDA topic modeling | Cluster the meeting's vocabulary into 5 latent topics and assign each transcript segment a dominant topic. |
| Action items / decisions | Regex/keyword-pattern rules flag segments containing actionable or decision language, with optional deadline extraction. |
| T5-small summarization | Chunk the transcript, summarize each chunk, then summarize the concatenation of chunk summaries into one final abstractive summary. |
| Evaluation | Compare every stage's output against a manually-verified reference (WER, DER, ROUGE, action-item P/R/F1), plus model comparison, ablation, and error-analysis layers built on top. |
| MeetSense dashboard | Streamlit UI presenting the transcript, summary, topics, action items, and every evaluation result. |

## Technologies Used

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Web UI | Streamlit 1.63 |
| Deep learning framework | PyTorch 2.13 |
| ASR | OpenAI Whisper (`openai-whisper` 20250625) |
| Speaker diarization | PyAnnote Audio 4.0.7 (`pyannote/speaker-diarization-community-1`) |
| Voice activity detection | Silero VAD 6.2.1 |
| Summarization | Hugging Face Transformers 5.17.0 (`google-t5/t5-small`) |
| NLP toolkit | NLTK 3.10.3 |
| Keyword/topic modeling | scikit-learn 1.9.0 (`TfidfVectorizer`, `CountVectorizer`, `LatentDirichletAllocation`) |
| Evaluation | `jiwer` 4.0.0 (WER), `pyannote.metrics` 4.1 (DER), `rouge-score` 0.1.2 (ROUGE), `rapidfuzz` 3.14.6 (fuzzy action-item matching) |
| Audio I/O | `soundfile`, `sounddevice`, `scipy` |
| Config | `python-dotenv` (Hugging Face token loading) |
| System dependency | FFmpeg (required by Whisper and by PyAnnote's audio backend) |

## Deep Learning Components

> The system currently uses **pretrained** deep learning models and
> focuses on integration, evaluation, experimentation and system-level
> analysis. No model in this project was trained from scratch.

**Silero VAD**
- Role: binary speech/non-speech classification over short audio frames.
- Input: 16kHz mono waveform. Output: a list of `(start, end)` speech
  timestamps, used to build a speech-only audio track and to report
  speech/silence statistics.
- Why used: a lightweight, pretrained neural VAD is far more robust to
  background noise than energy-threshold VAD, without needing any
  training.

**Whisper (`base`)**
- Role: automatic speech recognition — converts audio into timestamped
  text segments.
- Produces per-segment start/end timestamps directly, which downstream
  alignment with diarization depends on.
- Pretrained by OpenAI on large-scale weakly-supervised multilingual
  audio-text data; used here purely for inference, not fine-tuned.

**PyAnnote (`speaker-diarization-community-1`)**
- Role: speaker diarization — determines *who* was speaking *when*,
  independent of what was said.
- Output: a list of `(start, end, speaker_label)` turns (`SPEAKER_00`,
  `SPEAKER_01`, ...). These are anonymous cluster labels, not real
  identities.
- Pretrained, loaded via `pyannote.audio.Pipeline.from_pretrained(...)`
  with a Hugging Face access token (the model is gated).

**T5-small (`google-t5/t5-small`)**
- Role: abstractive summarization via a pretrained encoder-decoder
  transformer, prompted with the `"summarize: "` prefix.
- Generation: beam search (`num_beams=4`), `no_repeat_ngram_size=3`,
  applied hierarchically (chunk-level summaries, then a summary of the
  chunk summaries) to handle transcripts longer than the model's input
  window.
- Loaded via `transformers.AutoModelForSeq2SeqLM.from_pretrained(...)`;
  used purely for inference.

## NLP Components

| Component | Method | Category |
|---|---|---|
| Sentence/word tokenization | NLTK `sent_tokenize` / `word_tokenize` | Traditional/statistical NLP |
| Stopword removal | NLTK stopword list | Traditional NLP |
| Lemmatization | NLTK `WordNetLemmatizer` | Traditional/rule-based NLP |
| POS tagging | NLTK `pos_tag` (averaged perceptron tagger) | Traditional/statistical NLP |
| Named entity recognition | NLTK `ne_chunk` (maxent chunker) | Traditional/statistical NLP — **not** a neural NER model |
| Keyword extraction | scikit-learn `TfidfVectorizer` | Traditional/statistical NLP |
| Topic modeling | scikit-learn `CountVectorizer` + `LatentDirichletAllocation` | Traditional/statistical NLP |
| Action-item extraction | Regex keyword-pattern matching (`ACTION_PATTERNS`) | Rule-based |
| Decision extraction | Regex keyword-pattern matching (`DECISION_PATTERNS`) | Rule-based |
| Deadline extraction | Regex date/day-of-week patterns | Rule-based |
| Summarization | T5-small transformer | Deep learning / transformer |
| Extractive summary baseline | TF-IDF sentence ranking | Traditional/statistical NLP |

## Complete Pipeline

The production pipeline is orchestrated by `src/run_pipeline.py`, which
runs these steps in order (each a standalone script, invoked as a
subprocess, hard-failing the whole run on any non-zero exit code):

1. `src/audio_preprocessing.py` — resample to 16kHz mono.
2. `src/vad.py` — Silero VAD; speech segments + speech-only audio.
3. `src/transcription.py` — Whisper `base` transcription of the raw
   audio (`data/audio/meeting.wav`).
4. `src/diarization.py` — PyAnnote diarization of the raw audio.
5. `src/merge_transcript.py` — aligns Whisper segments with diarization
   turns by temporal overlap, producing `speaker_transcript.json`.
6. `src/nlp_preprocessing.py` — tokenization, POS tagging, lemmatization,
   NER.
7. `src/tfidf_keywords.py` — TF-IDF keyword extraction.
8. `src/topic_modeling.py` — LDA topic modeling.
9. `src/action_item_extraction.py` — rule-based action items/decisions.
10. `src/summarization.py` — T5-small summarization.

Evaluation (`src/wer_evaluation.py`, `src/der_evaluation.py`,
`src/rouge_evaluation.py`, `src/action_item_evaluation.py`) and the
experimentation layer (`src/model_comparison.py`, `src/ablation_study.py`,
`src/error_analysis.py`) are run separately via `src/run_experiments.py`
(see [Running Evaluation](#running-evaluation) and
[Running Experiments](#running-experiments)) — they read the pipeline's
existing outputs rather than being part of `run_pipeline.py` itself.

## Project Structure

```
NLP/
├── app.py                       # MeetSense landing page (Streamlit entrypoint)
├── app_workspace.py             # MeetSense workspace (exec'd by app.py after Start)
├── requirements.txt
├── .env                         # HF_TOKEN (not committed)
├── assets/                      # Landing page images
├── data/
│   └── audio/meeting.wav        # The single development recording
├── outputs/
│   ├── preprocessed/            # 16kHz mono audio + metadata
│   ├── vad/                     # Speech segments, speech-only audio, VAD stats
│   ├── transcripts/             # Whisper transcript, diarization, speaker-aware transcript
│   ├── nlp/                     # NLP-processed transcript, TF-IDF, topics, action items
│   ├── summaries/               # T5-small summary, extractive baseline summary
│   └── evaluations/             # WER/DER/ROUGE/action-item results, model comparison,
│                                 # ablation results, error analysis, manually-verified references
└── src/
    ├── run_pipeline.py          # Orchestrates the 10 production pipeline steps
    ├── audio_preprocessing.py
    ├── vad.py
    ├── transcription.py
    ├── diarization.py
    ├── merge_transcript.py
    ├── nlp_preprocessing.py
    ├── tfidf_keywords.py
    ├── topic_modeling.py
    ├── action_item_extraction.py
    ├── summarization.py
    ├── wer_evaluation.py        # Standalone evaluation scripts
    ├── der_evaluation.py
    ├── rouge_evaluation.py
    ├── action_item_evaluation.py
    ├── create_der_reference.py  # Builds the manually-verified DER reference
    ├── extractive_summary_baseline.py  # TF-IDF summarization baseline
    ├── model_comparison.py      # Experimentation layer
    ├── ablation_study.py
    ├── error_analysis.py
    ├── run_experiments.py       # Runs all evaluation + experimentation scripts, with .bak backups
    ├── live_meeting.py          # Standalone microphone-based live transcription utilities
    ├── live_transcribe.py
    ├── live_speaker_transcribe.py
    └── test_diarization.py
```

## Installation

```bash
cd /Users/advikamishra/Desktop/NLP
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Setup

The app must always be launched through the project's virtual environment,
**not** the bare `streamlit` command, which can resolve to a different
(e.g. Anaconda) Streamlit installation on this machine:

```bash
cd /Users/advikamishra/Desktop/NLP
source .venv/bin/activate
python -m streamlit run app.py
```

## Hugging Face Token Configuration

PyAnnote's `speaker-diarization-community-1` pipeline is a gated model on
Hugging Face. Create a `.env` file in the project root containing:

```
HF_TOKEN=<your Hugging Face access token>
```

`src/diarization.py` loads it via `python-dotenv`
(`load_dotenv(dotenv_path=".../NLP/.env")`) and raises immediately if
`HF_TOKEN` is not set. You must also accept the model's terms of use on
Hugging Face before the token will work.

## FFmpeg Requirement

Whisper and PyAnnote's audio backend both require the FFmpeg binary to be
installed on the system (this is a system dependency, not a Python
package):

```bash
brew install ffmpeg   # macOS
```

`src/diarization.py` additionally patches `DYLD_LIBRARY_PATH` to point at
Homebrew's FFmpeg lib directory on macOS before importing `torch`/
`pyannote`.

## Running the Offline Pipeline

```bash
cd /Users/advikamishra/Desktop/NLP
source .venv/bin/activate
python src/run_pipeline.py
```

This requires `data/audio/meeting.wav` to exist and `HF_TOKEN` to be
configured. It runs all 10 production steps in order and writes their
outputs under `outputs/`.

## Running MeetSense (the App)

```bash
cd /Users/advikamishra/Desktop/NLP
source .venv/bin/activate
python -m streamlit run app.py
```

Open the printed local URL, click **Start**, and the workspace loads the
existing `outputs/` files (it does not re-run the pipeline).

## Running Evaluation

Each evaluation script reads existing pipeline outputs and a
manually-verified reference file, and writes one JSON result file:

```bash
cd /Users/advikamishra/Desktop/NLP
source .venv/bin/activate
python src/wer_evaluation.py
python src/der_evaluation.py
python src/rouge_evaluation.py
python src/action_item_evaluation.py
```

## Running Experiments

`src/run_experiments.py` runs the four evaluation scripts above, then the
model-comparison, ablation-study, and error-analysis scripts, in
dependency order. Any existing output JSON is backed up to `<file>.bak`
before being overwritten:

```bash
cd /Users/advikamishra/Desktop/NLP
source .venv/bin/activate
python src/run_experiments.py
```

This takes roughly 8–9 minutes, dominated by the model-comparison stage,
which transcribes the full ~30-minute recording with both Whisper `base`
and `small`.

## Evaluation Metrics

| Metric | Measures | Computed by |
|---|---|---|
| WER (Word Error Rate) | Substitutions + insertions + deletions, divided by reference word count — lower is better. | `src/wer_evaluation.py`, `jiwer.wer` |
| DER (Diarization Error Rate) | Fraction of reference speech time incorrectly attributed (confusion + missed speech + false alarm) — lower is better. | `src/der_evaluation.py`, `pyannote.metrics.diarization.DiarizationErrorRate` |
| ROUGE-1 / ROUGE-2 / ROUGE-L | Unigram / bigram / longest-common-subsequence overlap between the generated summary and a reference text — higher is better. | `src/rouge_evaluation.py`, `rouge_score` |
| Action-item Precision / Recall / F1 | Precision/recall/F1 of extracted action items against manually-written reference items, matched via fuzzy string similarity (threshold 50). | `src/action_item_evaluation.py`, `rapidfuzz` |

## Actual Results

**All values below were read directly from the JSON files under
`outputs/evaluations/` — none are estimated.**

> **Scope**: every number below comes from **one manually-verified
> development recording** (`data/audio/meeting.wav`, ≈29 min 41 s), with
> ground truth manually annotated for only the first ~75–98 seconds (the
> exact span of the reference files). This is a single development
> sample, not a benchmark dataset, and these results should not be read
> as general claims about Whisper, PyAnnote, or T5-small performance
> beyond this one recording.

| Metric | Value | Source file |
|---|---|---|
| WER | **0.71%** (0.0070921986) | `outputs/evaluations/wer_results.json` |
| DER | **2.22%** (0.0222192948); confusion 1.518s, missed 0s, false alarm 0s, correct 66.801s over a 68.319s reference window | `outputs/evaluations/der_results.json` |
| ROUGE-1 F1 | **0.3353** (precision 0.5833, recall 0.2353) | `outputs/evaluations/rouge_results.json` |
| ROUGE-2 F1 | **0.2061** (precision 0.3617, recall 0.1441) | `outputs/evaluations/rouge_results.json` |
| ROUGE-L F1 | **0.2635** (precision 0.4583, recall 0.1849) | `outputs/evaluations/rouge_results.json` |
| Action items | Reference 4, Predicted 1, TP 1, FP 0, FN 3, Precision **1.0**, Recall **0.25**, F1 **0.4** | `outputs/evaluations/action_item_results.json` |

Note that `reference_summary.txt` (used for ROUGE) is an extractive
excerpt of the meeting's opening, not a human-written abstractive
summary — ROUGE here measures overlap with the source text, not
summary quality in the fullest sense.

## Model Comparison

From `outputs/evaluations/model_comparison.json` (`src/model_comparison.py`):

**ASR — Whisper `base` (production) vs `small`** (both transcribed the
full recording; WER scored over the same reference window as above):

| Model | WER | Transcribe time |
|---|---|---|
| `base` | 0.71% | ≈90s |
| `small` | 1.77% | ≈239s |

`small` scored a *higher* WER than `base` on this recording — a genuine,
single-sample result, not necessarily representative beyond this one
meeting.

**Summarization — T5-small (production) vs TF-IDF extractive baseline**
(both scored against the same ROUGE reference):

| Model | ROUGE-1 F1 | Summary length |
|---|---|---|
| T5-small | 0.3353 | 45 words |
| TF-IDF extractive baseline | 0.2045 | 53 words |

**Diarization**: no model-vs-model comparison was run — only one
diarization pipeline (`pyannote/speaker-diarization-community-1`) is
installed in this environment, so a second model isn't available to
compare against. This is stated explicitly in the JSON output rather than
fabricated or silently skipped.

## Ablation Study

From `outputs/evaluations/ablation_results.json`
(`src/ablation_study.py`), structured as Whisper only → +VAD →
+diarization → full pipeline. Metrics that don't apply to a given stage
are `null`, with a stated reason, rather than a fabricated value:

| Stage | WER | DER | Action F1 | ROUGE-1 F1 |
|---|---|---|---|---|
| `whisper_only` | 0.71% | n/a | n/a | n/a |
| `whisper_plus_vad` | **3.90%** | n/a | n/a | n/a |
| `whisper_vad_diarization` | 0.71%¹ | 2.22% | n/a | n/a |
| `full_pipeline` | 0.71%¹ | 2.22%¹ | 0.4000 | 0.3353 |

¹ inherited from an earlier stage (documented in the JSON), not
independently re-measured.

A significant finding, discovered and documented rather than smoothed
over: **VAD filtering increased WER on this recording** (0.71% → 3.90%),
the opposite of the usual expectation. The ablation JSON also documents
an existing architectural fact found during this work:
`src/transcription.py` and `src/diarization.py` both read
`data/audio/meeting.wav` directly, so in the current production pipeline
VAD's output is not actually wired into ASR or diarization — the
`whisper_plus_vad` stage measures VAD's effect on ASR in isolation, not
as currently deployed.

## Error Analysis

From `outputs/evaluations/error_analysis.json`
(`src/error_analysis.py`), grounded in the actual outputs and in the
regex pattern lists used by `src/action_item_extraction.py`:

- **ASR**: 2 substitutions, 0 insertions, 0 deletions (280/282 word
  hits). Punctuation contributes no additional error beyond those 2
  substitutions on this sample.
- **Diarization**: confusion 1.518s (from `der_results.json`). 152
  overlapping adjacent-turn pairs found across the full 637-turn
  predicted diarization output. 11.39% of the 404 speaker-transcript
  segments have <50% speaker-overlap confidence (a plausible, not
  confirmed, boundary-error indicator).
- **NLP**: 2/25 (8%) named entities flagged as likely false positives by
  a curated discourse-marker lexicon (e.g. "Okay" tagged `GPE`). 7/30
  (23%) top TF-IDF keywords are discourse noise — traced to
  `TfidfVectorizer`/`CountVectorizer` in `tfidf_keywords.py`/
  `topic_modeling.py` setting no `stop_words` parameter at all. Average
  pairwise topic-word Jaccard similarity: 0.10.
- **Action items**: 34 of 35 extracted items (full meeting) have never
  been checked against any reference — reported as unknown, not assumed
  correct. Of the 3 reference items missed, 1 is a genuine pattern-
  coverage gap (`ACTION_PATTERNS`/`DECISION_PATTERNS` don't match its
  wording at all); the other 2 do match the action-pattern list in their
  reference wording, so their miss is a paraphrase/threshold issue.
- **Summarization**: 89:1 compression ratio (4004 source words → 45
  summary words). 3 of 5 checked key facts from the meeting's opening
  (three-week timeline, November 2nd deadline, printer/fax equipment)
  are absent from the generated summary. Zero unsupported (non-source)
  proper nouns/numbers detected.

All heuristic checks above (discourse-marker lexicon, key-fact presence,
unsupported-token grounding) are lexical approximations documented as
such in the JSON — not a human-reviewed annotation of every item.

## Limitations

- **Single development sample, not a benchmark dataset**: every
  quantitative result comes from one ~30-minute recording, with ground
  truth manually verified for only its first ~75–98 seconds. No claim
  here generalizes beyond this recording.
- **Pretrained models only**: Whisper, PyAnnote, and T5-small are used
  as-is; none is fine-tuned for this meeting domain.
- **Rule-based action-item/decision extraction**: a fixed keyword-pattern
  regex list, not a learned classifier — has a demonstrated
  pattern-coverage gap (see Error Analysis) and 34 of 35 full-meeting
  extractions have never been checked against ground truth.
- **NER false positives**: NLTK's classical `ne_chunk` (not a neural
  NER model) mistags common discourse words as entities.
- **TF-IDF/LDA filler-word noise**: neither `TfidfVectorizer` nor
  `CountVectorizer` in the keyword/topic-modeling scripts is configured
  with stopword removal, so discourse fillers compete with content words.
- **Speaker confusion and overlapping speech**: DER confusion is nonzero
  (1.518s in the verified window), and 152 overlapping-turn pairs were
  found in the predicted diarization across the full meeting.
- **VAD is not wired into ASR/diarization in production**: `vad.py`'s
  output is currently used only for statistics/experimentation, not fed
  into `transcription.py` or `diarization.py`.
- **Live transcription is a separate, standalone path**: the workspace's
  Live Meeting page only displays a pre-existing
  `outputs/transcripts/live_meeting.json`; it does not itself capture
  microphone audio from the browser — that requires running
  `src/live_meeting.py` (or the other `live_*.py` scripts) separately.
- **Computational requirements**: Whisper `small` took ~4 minutes to
  transcribe the ~30-minute recording on this machine's CPU (no CUDA
  GPU used); PyAnnote diarization and T5 summarization are also
  CPU/MPS-bound without a CUDA GPU.
- **No domain-specific fine-tuning**: all models are general-purpose,
  pretrained checkpoints.

## Future Work

- Evaluate on a larger, more diverse meeting dataset with full
  ground-truth coverage (not just the first ~75–98 seconds).
- Domain-specific fine-tuning of ASR/summarization for meeting-style
  speech.
- Improved, persistent speaker identification (beyond anonymous
  per-recording cluster labels).
- Overlap-aware diarization to better handle simultaneous speech.
- A learned (rather than purely rule-based) action-item/decision
  classifier.
- Stronger summarization models (e.g. larger T5/BART/PEGASUS variants),
  evaluated against a genuine human-written abstractive reference.
- Multilingual evaluation.
- Browser-native live audio capture integrated directly into the
  Streamlit workspace, rather than a separate standalone script.
- Scalable deployment (containerization, job queue for long recordings).
- User authentication and persistent, multi-meeting history.

## Academic Contribution

This project contributes:

- An end-to-end, reproducible speaker-aware meeting-intelligence pipeline
  integrating a pretrained neural VAD, ASR, and speaker-diarization model.
- Timestamp-based alignment between independently-produced ASR and
  diarization outputs into one speaker-labeled transcript.
- A traditional-NLP analysis layer (TF-IDF keywords, LDA topics, rule-
  based action/decision extraction) alongside a transformer-based
  summarizer.
- Quantitative, reference-grounded evaluation (WER, DER, ROUGE,
  action-item P/R/F1) rather than only qualitative demonstration.
- A controlled model comparison (ASR model size; neural vs. non-neural
  summarization) and an ablation study isolating VAD's and diarization's
  measured contribution.
- Structured, code- and data-grounded error analysis across five pipeline
  stages, including identification of concrete, fixable issues (e.g. the
  missing stopword configuration behind TF-IDF/LDA noise, and the
  regex-coverage gap behind two missed action items).
- An interactive dashboard (MeetSense) surfacing all of the above.

This is presented as a **systems integration, evaluation, and analysis**
contribution built on pretrained components — it does not claim to
introduce a novel model architecture or training method.

## Conclusion

MeetSense demonstrates that a working, speaker-aware meeting
transcription and summarization system can be assembled from currently
available pretrained deep learning models and traditional NLP
techniques, and — more importantly for this project's academic goals —
that the resulting system can be rigorously measured: where it performs
well (WER 0.71%, DER 2.22% on the verified window), where it does not
(ROUGE and action-item recall are comparatively low; VAD unexpectedly
increased WER; TF-IDF/LDA carry discourse noise), and why, traced back to
specific configuration and methodology choices in the code rather than
asserted from intuition. All results are reproducible via
`src/run_pipeline.py` and `src/run_experiments.py` against the single
development recording included with this project.
