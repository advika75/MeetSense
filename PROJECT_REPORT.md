# Project Report — MeetSense

### Speaker-Aware Meeting Transcription and Summarization System

---

## 1. Abstract

MeetSense is an end-to-end system that converts a recorded meeting into a
speaker-attributed transcript, keyword and topic summaries, rule-based
action items and decisions, and a transformer-generated abstractive
summary, presented through an interactive Streamlit dashboard. The system
integrates four pretrained deep learning models — Silero VAD, Whisper,
PyAnnote, and T5-small — with traditional statistical NLP (TF-IDF, LDA,
NLTK-based tokenization/POS/NER) and rule-based information extraction.
Beyond building a working pipeline, this project's central focus is
rigorous, reference-grounded evaluation: Word Error Rate, Diarization
Error Rate, ROUGE, and action-item precision/recall/F1 are measured
against a manually-verified reference; a controlled model comparison
(ASR model size; neural vs. non-neural summarization) and an ablation
study (Whisper only → +VAD → +diarization → full pipeline) are run; and a
structured, code- and data-grounded error analysis is performed across
five pipeline stages. All results in this report are drawn from a single
manually-verified development recording and are reported as such, not
generalized beyond it.

## 2. Introduction

Meetings generate large volumes of unstructured audio that is expensive
to review manually. Automatic transcription addresses part of this
problem, but plain transcription without speaker attribution, structured
summaries, and action-item extraction leaves most of the manual-review
burden in place. This project investigates whether an end-to-end system
built from currently-available pretrained speech and language models can
close that gap, and — as the project's primary academic contribution —
whether the resulting system's actual accuracy, failure modes, and
component-level tradeoffs can be measured and explained rather than only
demonstrated.

## 3. Problem Statement

Given a raw meeting audio recording, produce: (a) an accurate,
speaker-attributed transcript; (b) a concise abstractive summary; (c) a
list of extracted action items and decisions; (d) keyword and topic
summaries of what was discussed; and (e) a quantitative account of how
accurate each of these outputs actually is, using currently available
pretrained models without domain-specific fine-tuning or new training
data collection.

## 4. Objectives

1. Assemble a working pipeline: preprocessing → VAD → ASR → diarization →
   alignment → NLP analysis → summarization.
2. Evaluate each measurable stage against a manually-verified reference
   (WER, DER, ROUGE, action-item P/R/F1).
3. Compare feasible model/component alternatives already available in
   the environment.
4. Run an ablation study isolating VAD's and diarization's measured
   contribution.
5. Perform structured, evidence-grounded error analysis per stage.
6. Present all outputs and evaluation results in an interactive
   dashboard (MeetSense).

## 5. Literature / Technology Background

- **Whisper** (Radford et al., OpenAI) — a Transformer encoder-decoder
  trained on large-scale weakly-supervised multilingual speech-text data
  for robust automatic speech recognition, used here via the
  `openai-whisper` package (`base` model).
- **PyAnnote** — an open-source speaker diarization toolkit built on
  pretrained neural speaker-embedding and segmentation models, used here
  via the gated `pyannote/speaker-diarization-community-1` pipeline.
- **Silero VAD** — a lightweight pretrained neural voice-activity
  detector.
- **T5** (Raffel et al., Google) — a text-to-text Transformer pretrained
  with a unified "text-to-text" objective; the `small` variant is used
  here for abstractive summarization via prompt-prefixed generation.
- **TF-IDF and LDA** — classical statistical methods (predating deep
  learning) for keyword weighting and topic modeling, still practical and
  interpretable baselines for unsupervised text analysis.
- **WER, DER, ROUGE** — standard evaluation metrics for ASR, diarization,
  and summarization respectively, each computed here with widely-used
  open-source libraries (`jiwer`, `pyannote.metrics`, `rouge_score`)
  rather than custom implementations.

## 6. Methodology

The system is built as a sequence of independent, file-based pipeline
stages (each a standalone script reading and writing JSON/audio files
under `outputs/`), orchestrated by `src/run_pipeline.py`. This
file-based design means every intermediate artifact — VAD segments, raw
ASR output, raw diarization output, the merged speaker-aware transcript,
NLP-processed transcript, keywords, topics, action items, and the summary
— is independently inspectable and independently evaluable, which is what
enables the ablation study and error analysis described in Sections 21–23.
Evaluation and experimentation are deliberately kept separate from the
production pipeline (`src/wer_evaluation.py`,
`src/der_evaluation.py`, `src/rouge_evaluation.py`,
`src/action_item_evaluation.py`, `src/model_comparison.py`,
`src/ablation_study.py`, `src/error_analysis.py`, orchestrated by
`src/run_experiments.py`), so that re-running evaluation never requires
re-running the (comparatively expensive, HF-token-gated) production
pipeline.

## 7. System Architecture

```
Meeting Audio
   → Audio Preprocessing (16kHz mono resample)
   → Silero VAD (speech/silence segmentation)
   → Whisper ASR (timestamped transcription)
   → PyAnnote Speaker Diarization (speaker turns)
   → Timestamp Alignment (merge by temporal overlap)
   → Speaker-Aware Transcript
   → NLP Preprocessing (tokenize, POS-tag, lemmatize, NER)
   → TF-IDF Keywords + LDA Topic Modeling
   → Action Items / Decisions (rule-based)
   → T5-small Summarization
   → Evaluation (WER, DER, ROUGE, Action-item P/R/F1,
                 Model Comparison, Ablation Study, Error Analysis)
   → MeetSense Dashboard (Streamlit)
```

Architecturally significant fact, discovered during this project's error
analysis rather than assumed: Whisper transcription and PyAnnote
diarization both operate on the **raw** input audio
(`data/audio/meeting.wav`) directly; Silero VAD's output is not currently
consumed by either step. VAD's contribution is measured independently in
the ablation study (Section 22) rather than assumed to already be part of
the production ASR/diarization path.

## 8. Audio Preprocessing

`src/audio_preprocessing.py` resamples the raw recording to 16kHz mono
PCM (`outputs/preprocessed/meeting_16khz_mono.wav`), the format Silero
VAD and Whisper expect, and records basic metadata.

## 9. Voice Activity Detection

`src/vad.py` runs Silero VAD over the preprocessed audio
(`load_silero_vad`, `get_speech_timestamps`, threshold 0.5, minimum
speech duration 250ms, minimum silence duration 300ms, 100ms speech
padding), producing a list of speech segments
(`outputs/vad/speech_segments.json`), a concatenated speech-only audio
track (`outputs/vad/speech_only.wav`), and summary statistics
(`outputs/vad/vad_statistics.json`): on this recording, 73.29% speech /
26.71% silence out of 1780.65 total seconds, across 377 detected speech
segments.

## 10. Automatic Speech Recognition

`src/transcription.py` loads Whisper `base`
(`whisper.load_model("base")`, CPU unless CUDA is available) and
transcribes the raw audio, producing timestamped segments
(`outputs/transcripts/meeting.json`). Whisper's own per-segment
timestamps are what make downstream alignment with diarization possible.

## 11. Speaker Diarization

`src/diarization.py` loads the gated pretrained pipeline
`pyannote/speaker-diarization-community-1` via a Hugging Face access
token (`HF_TOKEN`, loaded from `.env`) and runs it on the raw audio,
producing a list of `(start, end, speaker_label)` turns
(`outputs/transcripts/diarization.json`) — 637 turns across the
recording in the current run.

## 12. Timestamp Alignment

`src/merge_transcript.py` merges Whisper's segments with PyAnnote's turns
by computing, for each Whisper segment, the diarization turn with maximum
temporal overlap, assigning that turn's speaker label to the segment and
recording the overlap duration as `speaker_overlap` (used later, in error
analysis, as an approximate confidence signal for the assignment). Output:
`outputs/transcripts/speaker_transcript.json`, 404 speaker-labeled
segments.

## 13. NLP Processing

`src/nlp_preprocessing.py` applies, per transcript segment: sentence and
word tokenization, stopword removal, lemmatization (NLTK
`WordNetLemmatizer`), POS tagging (NLTK averaged perceptron tagger), and
named entity chunking (NLTK `ne_chunk`, a classical statistical chunker,
not a neural NER model) — producing
`outputs/nlp/nlp_processed_transcript.json` with 404 processed segments
and 25 extracted named entities.

## 14. Keyword Extraction

`src/tfidf_keywords.py` builds a `TfidfVectorizer` (unigrams and
bigrams, `min_df=2`, `max_df=0.95`, **no stopword filtering configured**)
over the transcript's sentences, producing overall and per-speaker top
keywords (`outputs/nlp/tfidf_keywords.json`) — vocabulary size 335 over
393 documents.

## 15. Topic Modeling

`src/topic_modeling.py` builds a `CountVectorizer` (same `min_df`/
`max_df`, also with no stopword filtering configured) and fits a
5-topic `LatentDirichletAllocation` model, assigning each transcript
segment a dominant topic and topic-probability distribution
(`outputs/nlp/topic_modeling.json`) — vocabulary size 228.

## 16. Action Item and Decision Extraction

`src/action_item_extraction.py` scans each transcript segment's text
against three fixed regex pattern lists — `ACTION_PATTERNS` (modal/
intent phrases like "need to", "will", "should"), `DECISION_PATTERNS`
("we decided", "agreed to", ...), and `DEADLINE_PATTERNS` (day-of-week
and date phrases) — flagging matching segments as action items or
decisions, with deadline text extracted where present
(`outputs/nlp/action_items_decisions.json`): 35 action items and 1
decision extracted from the full meeting.

## 17. Summarization

`src/summarization.py` loads `google-t5/t5-small` and, because a full
meeting transcript exceeds T5's input window, applies summarization
hierarchically: the speaker-aware transcript is split into
token-bounded chunks (max 450 input tokens per chunk, prompted with
`"summarize: "`), each chunk is summarized independently (beam search,
`num_beams=4`, `no_repeat_ngram_size=3`, max 120 output tokens), and the
concatenation of chunk summaries is summarized once more into the final
45-word summary (`outputs/summaries/meeting_summary.json`).
`src/extractive_summary_baseline.py` additionally provides a non-neural
alternative: sentences from the NLP-processed transcript are ranked by
summed TF-IDF weight and greedily selected up to a word budget matched to
T5's output length, then re-ordered chronologically.

## 18. Evaluation Methodology

Four standalone scripts each compare one pipeline output against a
manually-verified reference file and write a self-describing JSON result:

- `src/wer_evaluation.py` — Whisper's first 17 segments (0.0–97.72s) vs.
  `reference_transcript.txt`, via `jiwer.wer` after lowercasing and
  punctuation stripping.
- `src/der_evaluation.py` — PyAnnote's diarization (restricted to the
  same 6.393–75.0s window) vs. `reference_diarization.json` (7 manually
  verified turns), via `pyannote.metrics.diarization.DiarizationErrorRate`.
- `src/rouge_evaluation.py` — T5's `final_summary` vs.
  `reference_summary.txt`, via `rouge_score.RougeScorer` (stemmed).
- `src/action_item_evaluation.py` — extracted action items in the
  0–75s window vs. `reference_action_items.txt` (4 items), matched via
  RapidFuzz fuzzy similarity (`0.7·token_set_ratio + 0.3·ratio`,
  threshold 50), yielding precision/recall/F1.

All four reference files were manually annotated against only the
recording's first ~75–98 seconds; no ground truth exists for the
remainder of the ~30-minute meeting.

## 19. Experimental Setup

All experiments run on the single available development recording
(`data/audio/meeting.wav`, 16kHz mono, ≈1780.65s / ≈29 min 41s), on CPU
(no CUDA GPU available in this environment; Whisper explicitly avoids MPS
in `src/transcription.py`). Whisper model weights for `base` and `small`
were both already cached locally, so the model comparison (Section 21)
required no new downloads. `src/run_experiments.py` orchestrates the full
evaluation + experimentation layer in dependency order, backing up any
existing output JSON to `<file>.bak` before overwriting it.

## 20. Results

All values below are read directly from `outputs/evaluations/*.json`:

| Metric | Value |
|---|---|
| WER | 0.71% (2 substitutions, 0 insertions, 0 deletions / 282 reference words) |
| DER | 2.22% (confusion 1.518s, missed 0s, false alarm 0s / 68.319s reference) |
| ROUGE-1 F1 | 0.3353 (precision 0.5833, recall 0.2353) |
| ROUGE-2 F1 | 0.2061 (precision 0.3617, recall 0.1441) |
| ROUGE-L F1 | 0.2635 (precision 0.4583, recall 0.1849) |
| Action items | Reference 4, Predicted 1, TP 1, FP 0, FN 3, P 1.0, R 0.25, F1 0.4 |

These results describe one manually-verified recording, not a benchmark
dataset, and are not generalized beyond it anywhere in this report.

## 21. Model Comparison

`src/model_comparison.py` compared the only model variants actually
available in this environment:

**ASR — Whisper `base` vs `small`** (both transcribed the full
recording; WER scored over the same reference window):

| Model | WER | Transcribe time |
|---|---|---|
| `base` | 0.71% | ≈90s |
| `small` | 1.77% | ≈239s |

**Summarization — T5-small vs. a TF-IDF extractive baseline** (both
scored against the same ROUGE reference):

| Model | ROUGE-1 F1 | Length |
|---|---|---|
| T5-small | 0.3353 | 45 words |
| TF-IDF extractive baseline | 0.2045 | 53 words |

**Diarization**: not compared — only one diarization pipeline is
installed in this environment, so no second model exists to compare
against; this is stated explicitly in the output rather than omitted or
fabricated.

An unexpected but genuine finding: `small` produced a *higher* WER than
`base` on this recording. This is reported as a single-sample
observation, not a general claim about Whisper model-size scaling.

## 22. Ablation Study

`src/ablation_study.py` measured four configurations — Whisper only,
+VAD, +diarization, full pipeline — reporting `null` with a stated reason
for any metric not meaningful at a given stage:

| Stage | WER | DER | Action F1 | ROUGE-1 F1 |
|---|---|---|---|---|
| Whisper only | 0.71% | n/a | n/a | n/a |
| + VAD | 3.90% | n/a | n/a | n/a |
| + Diarization | 0.71%* | 2.22% | n/a | n/a |
| Full pipeline | 0.71%* | 2.22%* | 0.4000 | 0.3353 |

\* inherited from an earlier stage, not independently re-measured.

The central finding is that VAD filtering **increased** WER (0.71% →
3.90%) on this recording — contrary to the usual expectation that
silence removal helps ASR. The study also surfaced, and documents
explicitly in its output, that the production pipeline's
`transcription.py` and `diarization.py` do not actually consume VAD's
output — so `whisper_plus_vad` isolates VAD's effect on ASR rather than
describing the deployed pipeline's actual current behavior.

## 23. Error Analysis

`src/error_analysis.py` performed five category-specific, data-grounded
analyses (full detail in `outputs/evaluations/error_analysis.json`):

- **ASR**: 2 word substitutions, 0 insertions, 0 deletions; punctuation
  formatting contributes zero additional error on this sample beyond
  those two word-level substitutions.
- **Diarization**: 1.518s of confusion in the verified window; 152
  overlapping adjacent-turn pairs (different speakers, intersecting time
  ranges) found across the full 637-turn predicted output; 11.39% of the
  404 transcript segments show <50% speaker-overlap confidence, a
  plausible boundary-error indicator.
- **NLP**: 2/25 (8%) named entities flagged as likely false positives
  (e.g. "Okay" tagged `GPE`) by a curated discourse-marker lexicon; 7/30
  (23%) top TF-IDF keywords are discourse noise, traced directly to the
  `TfidfVectorizer`/`CountVectorizer` configurations in
  `tfidf_keywords.py`/`topic_modeling.py` having no `stop_words`
  parameter set; average pairwise topic-word Jaccard similarity 0.10.
- **Action items**: 34 of 35 full-meeting extractions have never been
  checked against a reference (reported as unknown, not assumed
  correct); of the 3 reference items missed, 1 is a genuine
  pattern-coverage gap (matches neither `ACTION_PATTERNS` nor
  `DECISION_PATTERNS`), the other 2 do match `ACTION_PATTERNS` in their
  own wording, indicating a paraphrase/matching-threshold issue rather
  than a missing trigger word.
- **Summarization**: 89:1 compression ratio (4004 → 45 words); 3 of 5
  checked key facts from the meeting's opening (three-week timeline,
  November 2nd deadline, printer/fax equipment) are absent from the
  generated summary; zero unsupported (non-source) proper nouns/numbers
  detected in the summary text.

Every heuristic check above (the discourse-marker lexicon, the key-fact
presence check, the unsupported-token grounding check) is explicitly
documented in the JSON output as a lexical approximation, not a
human-verified annotation.

## 24. Limitations

- All quantitative results come from a single manually-verified
  development recording; ground truth covers only its first ~75–98
  seconds. This is a case study, not a benchmark evaluation.
- All four deep learning components are pretrained, general-purpose
  checkpoints, not fine-tuned for meeting-domain audio or language.
- Action-item/decision extraction is a fixed rule-based regex system
  with a demonstrated pattern-coverage gap, and most of its full-meeting
  output has never been checked against ground truth.
- Named entity recognition uses a classical statistical chunker (NLTK
  `ne_chunk`), not a neural NER model, and produces measurable false
  positives on discourse words.
- TF-IDF keyword extraction and LDA topic modeling have no stopword
  filtering configured, producing measurable discourse-word noise.
- Diarization shows nonzero speaker confusion and a substantial number
  of overlapping-turn pairs across the full meeting.
- VAD's output is not currently consumed by ASR or diarization in the
  production pipeline; its measured effect (a WER increase) applies to
  VAD used in isolation, not as currently deployed.
- The workspace's Live Meeting page only displays a pre-existing JSON
  file; live microphone capture requires running a separate standalone
  script.
- No CUDA GPU is available in this environment. Whisper transcription
  explicitly runs on CPU regardless (`src/transcription.py` only checks
  for CUDA, not MPS), so larger Whisper variants are noticeably slower
  (Whisper `small` took roughly 2.6× longer than `base` on the full
  recording).

## 25. Future Work

Evaluate on a larger, fully ground-truth-annotated meeting dataset;
domain-specific fine-tuning of ASR and summarization; improved,
persistent speaker identification; overlap-aware diarization; a learned
(rather than purely rule-based) action-item/decision classifier; a
stronger summarization model evaluated against a genuine human-written
abstractive reference; multilingual evaluation; browser-native live
audio capture integrated directly into the Streamlit workspace; scalable
deployment; user authentication; and persistent multi-meeting history.

## 26. Conclusion

MeetSense demonstrates that pretrained deep learning models for voice
activity detection, speech recognition, speaker diarization, and
summarization can be integrated, alongside traditional statistical NLP
and rule-based extraction, into a working, speaker-aware meeting
intelligence pipeline with an interactive dashboard. Equally central to
this project is that the resulting system was rigorously, quantitatively
evaluated rather than only demonstrated: WER and DER are low on the
verified window (0.71% and 2.22% respectively), while ROUGE and
action-item recall are comparatively low, VAD unexpectedly increased WER
in the ablation study, and TF-IDF/LDA noise and NER false positives were
traced to specific, fixable configuration gaps in the code. Every result
in this report is reproducible from the included recording via
`src/run_pipeline.py` and `src/run_experiments.py`, and every limitation
stated here reflects an actual, verified property of the implementation
rather than a general disclaimer.
