# Viva Preparation — MeetSense

Concise, implementation-accurate answers for oral defense. Every number
here matches the JSON files under `outputs/evaluations/` at the time of
writing (see `README.md` → Actual Results for full detail and source
files).

---

**1. What is the problem statement?**

Manually producing a speaker-attributed transcript, summary, and
action-item list from a meeting recording is slow and error-prone.
Generic transcription/summarization tools typically lack speaker
attribution and aren't tuned for multi-speaker, disfluent conversational
audio. This project builds and rigorously evaluates a pipeline that
addresses this using currently-available pretrained models.

**2. What is the objective?**

To build a working, reproducible, speaker-aware meeting-intelligence
pipeline (audio → transcript → speaker labels → NLP analysis → summary),
and to quantitatively evaluate it — via WER, DER, ROUGE, action-item
precision/recall/F1, a model comparison, an ablation study, and a
structured error analysis — rather than presenting only a demo.

**3. Why is the project "speaker-aware"?**

Because the transcript is not just plain text — every segment is
attributed to a speaker label (`SPEAKER_00`, `SPEAKER_01`, ...) obtained
by aligning Whisper's transcribed segments with PyAnnote's independently
computed diarization turns via temporal overlap
(`src/merge_transcript.py`).

**4. Why Whisper?**

Whisper is a strong, freely available pretrained ASR model that outputs
timestamped segments directly, which is required for aligning
transcribed text with diarization turns. The `base` size was chosen for
the production pipeline as a practical CPU-runnable speed/accuracy
tradeoff; `small` was also compared (see Q25).

**5. Why PyAnnote?**

`pyannote/speaker-diarization-community-1` is a state-of-the-art
pretrained diarization pipeline, directly usable via
`pyannote.audio.Pipeline.from_pretrained(...)`, and integrates with
`pyannote.metrics` for standard DER evaluation — avoiding the need to
implement or train a diarization model from scratch.

**6. Why Silero VAD?**

Silero VAD is a small, fast, pretrained neural voice-activity detector.
It's used to segment speech vs. silence and produce speech statistics
(73.29% speech / 26.71% silence on the project's recording), and its
speech-only audio is used in the ablation study to isolate VAD's effect
on ASR.

**7. Why T5?**

`google-t5/t5-small` is a small, fast, pretrained sequence-to-sequence
transformer well suited to a CPU-only development environment, prompted
with the standard `"summarize: "` prefix. Because a single meeting
transcript exceeds T5's input window, summarization is applied
hierarchically: each chunk is summarized, then the concatenation of
chunk summaries is summarized again into the final summary.

**8. Where is Deep Learning used?**

Four pretrained neural components: Silero VAD (speech detection),
Whisper (ASR), PyAnnote (speaker diarization), and T5-small
(summarization). All four are used purely for inference — none is
trained or fine-tuned in this project.

**9. Where is NLP used?**

Tokenization, stopword removal, lemmatization, POS tagging, and named
entity recognition (all NLTK, `src/nlp_preprocessing.py`); TF-IDF keyword
extraction and LDA topic modeling (scikit-learn); rule-based action-item,
decision, and deadline extraction (regex pattern matching,
`src/action_item_extraction.py`).

**10. Why do we need VAD?**

To separate speech from silence — in principle reducing wasted
computation on ASR/diarization and removing non-speech noise. In this
project's ablation study, VAD's effect was measured in isolation and
found to *increase* WER on this recording (0.71% → 3.90%) — an honest,
reported result, not the expected outcome, and discussed further in
Q26.

**11. What is speaker diarization?**

The task of partitioning an audio stream into segments labeled by
speaker identity ("who spoke when"), without necessarily knowing who
the speakers actually are — PyAnnote outputs anonymous cluster labels
(`SPEAKER_00`, `SPEAKER_01`, ...), not real names.

**12. What is timestamp alignment?**

The process (`src/merge_transcript.py`) of matching each Whisper
transcript segment to the diarization turn with the greatest temporal
overlap, assigning that turn's speaker label to the segment, and
recording the overlap duration (`speaker_overlap`) as a rough confidence
signal for that assignment.

**13. How are Whisper and PyAnnote combined?**

They run independently on the same raw audio file and produce two
separate outputs — Whisper's timestamped text segments
(`meeting.json`) and PyAnnote's timestamped speaker turns
(`diarization.json`). `merge_transcript.py` then combines them
after the fact by temporal overlap; there is no joint/end-to-end model.

**14. What is WER?**

Word Error Rate: `(substitutions + insertions + deletions) / reference
word count`, computed by `jiwer`. Measured on this project: **0.71%**
(2 substitutions, 0 insertions, 0 deletions out of 282 reference words),
over the manually-verified reference window.

**15. What is DER?**

Diarization Error Rate: the fraction of reference speech time
incorrectly attributed, combining missed-speech, false-alarm, and
speaker-confusion time, computed by `pyannote.metrics`. Measured:
**2.22%** (confusion 1.518s, missed 0s, false alarm 0s, over a 68.319s
manually-verified window).

**16. What is ROUGE?**

A family of recall-oriented n-gram/subsequence overlap metrics between a
generated summary and a reference text, computed here via `rouge_score`.

**17. What does ROUGE-1 measure?**

Unigram (single-word) overlap between the generated and reference
summaries. Measured: F1 **0.3353** (precision 0.5833, recall 0.2353).

**18. What does ROUGE-2 measure?**

Bigram (two-consecutive-word) overlap. Measured: F1 **0.2061**
(precision 0.3617, recall 0.1441).

**19. What does ROUGE-L measure?**

Longest common subsequence overlap, sensitive to sentence-level word
order without requiring exact contiguous n-grams. Measured: F1
**0.2635** (precision 0.4583, recall 0.1849).

**20. What is TF-IDF?**

Term Frequency–Inverse Document Frequency: a statistical weighting that
scores a term higher when it's frequent in a given document (here, a
transcript segment) but rare across the whole corpus, used to rank
keywords (`src/tfidf_keywords.py`, `sklearn.TfidfVectorizer`).

**21. What is LDA?**

Latent Dirichlet Allocation: a generative probabilistic topic model that
represents each document as a mixture of latent topics and each topic as
a distribution over words. Used here (`src/topic_modeling.py`,
`sklearn.LatentDirichletAllocation`) with 5 fixed topics.

**22. Why use rule-based action extraction?**

It's transparent, fast, requires no training data or labels, and its
exact failure modes are directly inspectable in the code (a fixed
`ACTION_PATTERNS`/`DECISION_PATTERNS` regex list). The tradeoff, found
during error analysis, is a genuine coverage gap: reference items phrased
without any of the listed trigger phrases (e.g. "Make the... decisions
by November 2nd") can never be extracted regardless of context.

**23. What are the limitations?**

See `README.md` → Limitations for the full list; headline points: results
come from a single manually-verified recording, not a benchmark dataset;
all models are pretrained and not fine-tuned for this domain; action-item
extraction is rule-based with a demonstrated coverage gap; NER (NLTK
`ne_chunk`) produces false positives on discourse words; TF-IDF/LDA have
no stopword filtering configured; diarization shows nonzero confusion and
152 overlapping-turn pairs across the full meeting; VAD is not currently
wired into ASR/diarization in production.

**24. What are the false positives?**

Concrete, measured examples from `outputs/evaluations/error_analysis.json`:
NER tags "Okay" as a `GPE` entity (2 of 25 total entities flagged by a
discourse-marker lexicon check); 7 of the top 30 TF-IDF keywords are
discourse filler words (e.g. "yeah") rather than content words, traced to
`TfidfVectorizer`/`CountVectorizer` being configured with no `stop_words`
parameter at all.

**25. What is the ablation study?**

A staged evaluation (`src/ablation_study.py`) comparing: (1) Whisper
only, (2) Whisper + VAD-filtered audio, (3) + PyAnnote diarization, (4)
the full pipeline — reporting only metrics that are scientifically
meaningful for each stage (e.g. DER is `null` with a stated reason for
stages without diarization, not a fabricated number).

**26. What did the ablation demonstrate?**

That VAD filtering *increased* WER on this recording (0.71% → 3.90%),
contrary to the usual expectation — a genuine single-sample finding. It
also surfaced and documented an existing architectural fact: in the
current production pipeline, `src/transcription.py` and
`src/diarization.py` both read the raw audio file directly, so VAD's
output isn't actually wired into either step; the ablation measures
VAD's effect on ASR *in isolation*, separately from how the pipeline is
currently deployed.

**27. What models are pretrained?**

All four deep learning components — Silero VAD, Whisper (`base`),
PyAnnote (`speaker-diarization-community-1`), and T5-small
(`google-t5/t5-small`) — are pretrained checkpoints loaded and used
purely for inference.

**28. Did you train any model from scratch?**

No. Every deep learning model used is a pretrained, publicly released
checkpoint. This project's contribution is integration, alignment,
evaluation, comparison, ablation, and error analysis — not model
training.

**29. What would you improve in future?**

See `README.md` → Future Work; headline items: evaluate on a larger,
fully-annotated dataset; fine-tune ASR/summarization for meeting-style
speech; overlap-aware diarization; a learned action-item/decision
classifier; a stronger summarizer scored against a genuine human-written
abstractive reference; browser-native live audio in the workspace itself.

**30. What is the main contribution of the project?**

An end-to-end, reproducible integration of pretrained neural VAD, ASR,
and diarization models with a traditional-NLP analysis layer and a
transformer summarizer, combined with a rigorous, reference-grounded
evaluation methodology (WER/DER/ROUGE/action-item P-R-F1), a controlled
model comparison, an ablation study, and structured, code-grounded error
analysis — all surfaced in an interactive dashboard. The contribution is
systems integration and evaluation rigor, not a novel model architecture.
