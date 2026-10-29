"""
Incremental live-meeting pipeline (Stage 3).

LiveMeetingSession owns all growing state for one live meeting and
exposes a single entry point, ingest_chunk(), that a capture source
(MicrophoneCapture or SimulatedFileCapture from live_capture.py) calls
once per completed rolling-window chunk.

Every stage below reuses the *same* functions the batch pipeline uses
(vad.detect_speech, transcription.transcribe_audio, diarization
.diarize_audio, merge_transcript.align_segments,
nlp_preprocessing.process_segment_text/extract_entities,
tfidf_keywords.compute_tfidf_keywords, topic_modeling.compute_topics,
action_item_extraction's per-segment helpers, and
summarization.summarize_text) -- none of that logic is duplicated
here. This module only adds the *incremental orchestration* around
those functions: overlap trimming, a capped rolling diarization
buffer, periodic recompute, and per-stage latency logging.

Known, deliberate limitation (see PROJECT_REPORT / paper writeup):
pyannote's community-1 pipeline, as used here, has no incremental/
streaming clustering API -- it globally re-clusters everything handed
to it in one call. Rather than diarizing each small chunk in
isolation (which would give inconsistent speaker labels across
chunks), this session re-runs the same, unmodified diarization call
on the *entire accumulated speech buffer so far*, capped at
DIARIZATION_BUFFER_CAP_SECONDS. This keeps labels internally
consistent for everything within the buffer, at the cost of: (a)
diarization cost that grows with buffer length until the cap is hit,
and (b) a theoretical risk that once older audio falls out of the
capped buffer, re-clustering could relabel a speaker inconsistently
with segments already shown to the user. Stage 5 measures whether
this is fast enough to be "live" in practice.
"""

import os
import sys
import json
import tempfile
import threading
import time
import resource
import subprocess

import numpy as np
import soundfile as sf

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from live_capture import SAMPLE_RATE, OVERLAP_SECONDS

from vad import load_silero_vad, detect_speech

from transcription import load_whisper_model

from diarization import load_diarization_pipeline, diarize_audio, embed_speech_segments

from merge_transcript import align_segments

from nlp_preprocessing import (
    ensure_nltk_resources,
    process_segment_text,
    extract_entities,
)

from tfidf_keywords import (
    build_documents as build_tfidf_documents,
    compute_tfidf_keywords,
)

from topic_modeling import (
    build_documents as build_topic_documents,
    compute_topics,
)

from action_item_extraction import (
    get_text,
    get_speaker,
    get_start_time,
    contains_pattern,
    extract_deadline,
    clean_action_text,
    ACTION_PATTERNS,
    DECISION_PATTERNS,
)

from summarization import load_summarizer, create_chunks, summarize_text


def apply_auto_gain(audio):
    """Boost-only gain control for live mic chunks (see
    LIVE_WHISPER_LANGUAGE comment above for why this exists). Leaves
    audio untouched if it is already loud enough, or if it is quiet
    enough to be noise/silence rather than soft speech."""

    if len(audio) == 0:
        return audio

    rms = float(np.sqrt(np.mean(np.square(audio))))

    if rms < AUTO_GAIN_NOISE_FLOOR_RMS or rms <= 0.0:
        return audio

    gain = min(AUTO_GAIN_MAX_FACTOR, AUTO_GAIN_TARGET_RMS / rms)

    if gain <= 1.0:
        return audio

    boosted = audio * gain

    peak = float(np.max(np.abs(boosted)))

    if peak > AUTO_GAIN_MAX_PEAK:
        boosted = boosted * (AUTO_GAIN_MAX_PEAK / peak)

    return boosted.astype(np.float32)


def current_process_rss_mb():

    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if output:
            return int(output) / 1024.0
    except Exception:
        pass

    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss / (1024 * 1024 if sys.platform == "darwin" else 1024)


# ============================================================
# CONFIGURATION
# ============================================================

WHISPER_MODEL_NAME = "base"

# Real-microphone testing (see PROJECT_REPORT) found live chunk audio
# sitting far quieter than the batch pipeline's normalized input
# (observed RMS ~0.02 vs. batch's peak-normalized-to-1.0 audio), which
# under-triggers Silero VAD's 0.5 energy-shaped threshold and starves
# Whisper of signal. Forcing a language also matters here: Whisper's
# per-chunk auto-detect on 17-20s of quiet audio was observed
# mis-detecting English speech as Nepali/Hindi/Marathi, producing
# garbled transcripts -- the batch pipeline avoids this only because
# it auto-detects once from a much longer, louder recording. Empty
# string disables the override and restores auto-detect per chunk.
LIVE_WHISPER_LANGUAGE = os.getenv("LIVE_WHISPER_LANGUAGE", "en").strip() or None

# Auto-gain target/limits for the above: boost-only (never attenuate),
# capped so a near-silent chunk's noise floor isn't amplified into
# apparent speech, and capped in peak so boosted audio never clips.
AUTO_GAIN_TARGET_RMS = 0.08
AUTO_GAIN_MAX_FACTOR = 12.0
AUTO_GAIN_NOISE_FLOOR_RMS = 0.0015
AUTO_GAIN_MAX_PEAK = 0.98

# NOTE: an early measured run (see PROJECT_REPORT / Stage 5 writeup)
# showed pyannote community-1's diarization cost growing superlinearly
# with buffer length on this hardware (M1, 8GB): ~4s at a 20s buffer,
# ~12.4s at 37s, ~35.6s at 57s, ~95.8s at 74s. A 180s cap would make
# individual diarization calls take minutes. 60s is a hardware-driven
# compromise, not a value chosen for accuracy reasons; DIARIZE_EVERY_
# N_CHUNKS further amortizes the cost by not re-running it every chunk.
DIARIZATION_BUFFER_CAP_SECONDS = float(os.getenv("DIARIZATION_BUFFER_CAP_SECONDS", "90"))

DIARIZE_EVERY_N_CHUNKS = 2

RECOMPUTE_EVERY_N_CHUNKS = 3

MIN_BUFFER_SECONDS_FOR_DIARIZATION = 2.0

DEFAULT_OUTPUT_DIR = "outputs/live"


class LiveMeetingSession:

    def __init__(
        self,
        whisper_model_name=WHISPER_MODEL_NAME,
        diarization_buffer_cap_seconds=DIARIZATION_BUFFER_CAP_SECONDS,
        diarize_every_n_chunks=DIARIZE_EVERY_N_CHUNKS,
        recompute_every_n_chunks=RECOMPUTE_EVERY_N_CHUNKS,
        output_dir=DEFAULT_OUTPUT_DIR,
        load_models=True,
        diarization_device=None,
        live_full_diarization=False,
        online_similarity_threshold=0.35,
    ):

        self._lock = threading.RLock()

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.latency_log_path = os.path.join(
            output_dir, "latency_log.jsonl"
        )

        self.state_snapshot_path = os.path.join(
            output_dir, "live_meeting_state.json"
        )

        # Fresh log per session -- one JSONL file per live meeting run.
        open(self.latency_log_path, "w", encoding="utf-8").close()

        self.diarization_buffer_cap_seconds = (
            diarization_buffer_cap_seconds
        )

        self.diarize_every_n_chunks = diarize_every_n_chunks

        self.recompute_every_n_chunks = recompute_every_n_chunks
        self.live_full_diarization = bool(live_full_diarization) or os.getenv(
            "LIVE_FULL_DIARIZATION", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.online_similarity_threshold = float(os.getenv(
            "LIVE_SPEAKER_SIMILARITY_THRESHOLD", str(online_similarity_threshold)
        ))

        self._vad_model = None
        self._whisper_model = None
        self._diarization_pipeline = None
        self._summarizer_tokenizer = None
        self._summarizer_model = None
        self._summarizer_device = None

        # Set later via LiveSessionController.start(expected_speaker_count=
        # ...), once the user has supplied it -- not a constructor param
        # since the controller builds this session before the Start form
        # is even shown. None falls back to the original unbounded-up-
        # to-8 behavior. When set, this replaces the hardcoded cap in
        # _online_speaker_segments(): without a known count, a voice that
        # doesn't match any existing centroid closely enough always gets
        # a brand-new "Speaker N" label, which fragments one real person
        # into several labels over a session (the most common complaint
        # about live diarization quality). Knowing the true count turns
        # "not similar enough -> new speaker" into "not similar enough ->
        # still the closest existing speaker, once we're at full count".
        self.expected_speaker_count = None

        if load_models:
            self.load_models(
                whisper_model_name,
                diarization_device=diarization_device,
            )

        self._init_state()

    # ------------------------------------------------------------
    # GROWING STATE (init + reset share this; models are untouched)
    # ------------------------------------------------------------

    def _init_state(self):

        self.asr_segments = []
        # Growing list of {"start","end","text"} on the GLOBAL
        # meeting timeline, already trimmed for chunk overlap.

        self._diarization_buffer = np.zeros(0, dtype=np.float32)
        self._diarization_buffer_start_time = 0.0

        self.diarization_segments = []
        # Latest full diarization result, GLOBAL timeline.

        self.aligned_transcript = []
        self._processed_cursor = 0
        self.online_speaker_centroids = {}
        self.online_speaker_updates = {}
        self._last_online_speaker = None
        self.embedding_clustering_ms = []
        self._diarization_thread = None
        self._pending_diarization = None
        self._diarization_running = False
        self._diarization_last_error = None
        self._diarization_started_at = None
        self._diarization_refinement_seconds = []
        self.stop_diarization_timed_out = False

        self.nlp_processed_segments = []
        self.named_entities = []
        self._seen_entity_keys = set()

        self.action_items = []
        self.decisions = []

        self.chunk_summaries = []
        self.whole_summary = ""

        self.tfidf_result = {
            "document_count": 0,
            "vocabulary_size": 0,
            "top_keywords": [],
            "speaker_keywords": {},
        }

        self.topics_result = {
            "document_count": 0,
            "vocabulary_size": 0,
            "number_of_topics": 0,
            "topics": [],
            "topic_counts": {},
            "speaker_topic_distribution": {},
            "segment_topic_assignments": [],
        }

        self.chunk_count = 0

        self.latency_log = []

        self.session_start_wall_time = time.time()
        self.capture_start_wall_time = self.session_start_wall_time

    def reset_state(self):
        """Clear all growing meeting state (transcript, summaries,
        action items, keywords, latency log) without reloading any
        model. Added for the Streamlit "Reset" control -- on an 8GB
        machine, reloading Whisper/pyannote/T5 just to clear state
        is not acceptable, and there was previously no way to clear
        state independently of __init__."""

        with self._lock:

            self._init_state()

            # Fresh log file for the new in-process session.
            open(
                self.latency_log_path, "w", encoding="utf-8"
            ).close()

            if os.path.exists(self.state_snapshot_path):
                os.remove(self.state_snapshot_path)

    # ------------------------------------------------------------
    # MODEL LOADING (once per session, reused every chunk)
    # ------------------------------------------------------------

    def load_models(
        self,
        whisper_model_name=WHISPER_MODEL_NAME,
        diarization_device=None,
    ):

        print("[live_pipeline] Loading VAD model...")
        self._vad_model = load_silero_vad()

        print(
            f"[live_pipeline] Loading Whisper model "
            f"'{whisper_model_name}'..."
        )
        self._whisper_model = load_whisper_model(whisper_model_name)

        print(
            f"[live_pipeline] Loading diarization pipeline "
            f"(device={diarization_device or 'auto'})..."
        )
        self._diarization_pipeline = load_diarization_pipeline(
            diarization_device
        )

        print("[live_pipeline] Loading summarization model...")

        (
            self._summarizer_tokenizer,
            self._summarizer_model,
            self._summarizer_device,
        ) = load_summarizer()

        ensure_nltk_resources()

        print("[live_pipeline] All models loaded.")

    # ------------------------------------------------------------
    # UNLOAD (frees VAD/Whisper/diarization/T5 before e.g. the local
    # LLM takeaway extractor runs -- on an 8GB machine these cannot
    # all be resident at once). Growing meeting state is untouched;
    # only the model references are dropped. Models are NOT
    # reloaded automatically after this -- a session that calls
    # unload_models() is done needing the pipeline for its lifetime.
    # ------------------------------------------------------------

    def unload_models(self):

        with self._lock:

            self._vad_model = None
            self._whisper_model = None
            self._diarization_pipeline = None
            self._summarizer_tokenizer = None
            self._summarizer_model = None
            self._summarizer_device = None

        import gc

        gc.collect()

        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        except Exception:
            pass

        print("[live_pipeline] Models unloaded.")

    # ------------------------------------------------------------
    # MAIN ENTRY POINT: called once per completed rolling-window chunk
    # ------------------------------------------------------------

    def ingest_chunk(self, chunk_audio, chunk_index, chunk_start_time):
        """NOTE on locking: self._lock protects only the shared STATE
        that snapshot()/reset_state() read -- not the slow model
        calls themselves. ingest_chunk() is only ever called from one
        thread at a time (the controller's single processing worker),
        so there is no risk of two ingest_chunk calls racing each
        other regardless of locking; the lock exists only so a
        concurrent snapshot() (polled by the UI every ~2s) never
        observes state mid-mutation. An earlier version held the lock
        for this whole method, which meant a slow diarization call
        (up to ~160s, measured in Stage 3 testing) blocked every
        snapshot() call for that entire duration -- silently freezing
        the live UI it was supposed to keep responsive. Fixed once
        Stage 4 (the UI, which polls snapshot() concurrently with
        ingest_chunk running) exposed this."""

        timings = {}

        t0 = time.perf_counter()

        chunk_audio = apply_auto_gain(chunk_audio)

        # -------------------- 1. VAD --------------------

        t_stage = time.perf_counter()

        try:
            speech_timestamps = detect_speech(
                chunk_audio, SAMPLE_RATE, model=self._vad_model
            )
        except Exception as error:
            speech_timestamps = []
            timings["vad_error"] = str(error)

        timings["vad_ms"] = (time.perf_counter() - t_stage) * 1000

        has_speech = len(speech_timestamps) > 0

        # -------------------- 2. ASR --------------------

        t_stage = time.perf_counter()

        new_asr_segments = []

        if has_speech:

            try:

                result = self._whisper_model.transcribe(
                    chunk_audio,
                    fp16=False,
                    task="transcribe",
                    language=LIVE_WHISPER_LANGUAGE,
                    condition_on_previous_text=False,
                    verbose=False,
                )

                for segment in result.get("segments", []):

                    text = segment.get("text", "").strip()

                    if not text:
                        continue

                    new_asr_segments.append(
                        {
                            "start": chunk_start_time + segment["start"],
                            "end": chunk_start_time + segment["end"],
                            "text": text,
                        }
                    )

            except Exception as error:
                timings["asr_error"] = str(error)

        # Overlap trim: for every chunk after the first, drop
        # segments starting inside the region already covered by
        # the previous chunk's tail.
        if chunk_index > 0:

            boundary = chunk_start_time + OVERLAP_SECONDS

            new_asr_segments = [
                segment
                for segment in new_asr_segments
                if segment["start"] >= boundary
            ]

        with self._lock:
            self.asr_segments.extend(new_asr_segments)

        timings["asr_ms"] = (time.perf_counter() - t_stage) * 1000

        # -------------------- 3. Online embeddings / optional full diarization --------------------

        t_stage = time.perf_counter()

        ran_diarization = False

        if self.live_full_diarization and chunk_index == 0:

            new_audio_for_buffer = chunk_audio
            new_audio_global_start = chunk_start_time

        elif self.live_full_diarization:

            boundary_sample = int(round(OVERLAP_SECONDS * SAMPLE_RATE))

            new_audio_for_buffer = chunk_audio[boundary_sample:]
            new_audio_global_start = chunk_start_time + OVERLAP_SECONDS

        else:
            new_audio_for_buffer = np.zeros(0, dtype=np.float32)
            new_audio_global_start = chunk_start_time

        if self.live_full_diarization:
            with self._lock:

                if self._diarization_buffer.size == 0:

                    self._diarization_buffer = new_audio_for_buffer.copy()
                    self._diarization_buffer_start_time = (
                        new_audio_global_start
                    )

                else:

                    self._diarization_buffer = np.concatenate(
                        [self._diarization_buffer, new_audio_for_buffer]
                    )

                max_samples = int(
                    round(
                        self.diarization_buffer_cap_seconds * SAMPLE_RATE
                    )
                )

                if len(self._diarization_buffer) > max_samples:

                    trimmed = (
                        len(self._diarization_buffer) - max_samples
                    )

                    self._diarization_buffer = self._diarization_buffer[
                        trimmed:
                    ]

                    self._diarization_buffer_start_time += (
                        trimmed / SAMPLE_RATE
                    )

            # Snapshot the buffer under the lock so the slow
            # diarization call below (which can run for a very long
            # time, unlocked) runs against a stable copy rather than
            # one reset_state() could mutate concurrently.
                diarization_buffer_copy = self._diarization_buffer.copy()
                diarization_buffer_start_time = (
                    self._diarization_buffer_start_time
                )

            buffer_seconds = len(diarization_buffer_copy) / SAMPLE_RATE

            should_diarize_this_chunk = (
                chunk_index == 0
                or chunk_index % self.diarize_every_n_chunks == 0
            )

            if (
                has_speech
                and should_diarize_this_chunk
                and buffer_seconds >= MIN_BUFFER_SECONDS_FOR_DIARIZATION
            ):
                self._request_diarization(
                    diarization_buffer_copy, diarization_buffer_start_time
                )
                ran_diarization = True
        else:
            t_embed = time.perf_counter()
            online_segments = self._online_speaker_segments(
                chunk_audio, speech_timestamps, chunk_index, chunk_start_time
            )
            timings["embedding_clustering_ms"] = (time.perf_counter() - t_embed) * 1000
            self.embedding_clustering_ms.append(timings["embedding_clustering_ms"])
            with self._lock:
                self.diarization_segments.extend(online_segments)
            buffer_seconds = 0.0

        timings["diarization_ms"] = (
            time.perf_counter() - t_stage
        ) * 1000

        # -------------------- 4. Alignment --------------------

        t_stage = time.perf_counter()

        with self._lock:
            asr_segments_for_alignment = list(self.asr_segments)
            diarization_segments_for_alignment = list(
                self.diarization_segments
            )

        if diarization_segments_for_alignment:

            new_aligned_transcript = align_segments(
                asr_segments_for_alignment,
                diarization_segments_for_alignment,
            )

        else:

            new_aligned_transcript = [
                {
                    "start": round(segment["start"], 3),
                    "end": round(segment["end"], 3),
                    "speaker": self._provisional_speaker(segment),
                    "text": segment["text"],
                    "speaker_overlap": 0.0,
                    "speakers_refined": False,
                }
                for segment in asr_segments_for_alignment
            ]

        # Label uncovered rows provisionally from the nearest known
        # diarized turn; background completion flips the row marker.
        for row in new_aligned_transcript:
            mid = (row["start"] + row["end"]) / 2
            covered = any(s["start"] <= mid <= s["end"] for s in diarization_segments_for_alignment)
            if not covered:
                row["speaker"] = self._provisional_speaker(row)
                row["speakers_refined"] = False
            else:
                row["speakers_refined"] = True

        with self._lock:

            self.aligned_transcript = new_aligned_transcript

            new_segments = self.aligned_transcript[
                self._processed_cursor:
            ]

            self._processed_cursor = len(self.aligned_transcript)

        timings["alignment_ms"] = (
            time.perf_counter() - t_stage
        ) * 1000

        # -------------------- 5. Action items / decisions --------------------
        # Applied only to newly-arrived aligned segments.

        t_stage = time.perf_counter()

        new_action_items = []
        new_decisions = []

        for segment in new_segments:

            text = get_text(segment)

            if not text:
                continue

            speaker = get_speaker(segment)
            start_time_value = get_start_time(segment)
            deadline = extract_deadline(text)

            if contains_pattern(text, ACTION_PATTERNS):

                new_action_items.append(
                    {
                        "speaker": speaker,
                        "start_time": start_time_value,
                        "action": clean_action_text(text),
                        "deadline": deadline,
                        "extraction_method": "keyword_pattern",
                    }
                )

            if contains_pattern(text, DECISION_PATTERNS):

                new_decisions.append(
                    {
                        "speaker": speaker,
                        "start_time": start_time_value,
                        "decision": clean_action_text(text),
                        "extraction_method": "keyword_pattern",
                    }
                )

        with self._lock:

            for item in new_action_items:
                item["id"] = len(self.action_items) + 1
                self.action_items.append(item)

            for item in new_decisions:
                item["id"] = len(self.decisions) + 1
                self.decisions.append(item)

        timings["action_item_ms"] = (
            time.perf_counter() - t_stage
        ) * 1000

        # -------------------- 6. NLP preprocessing --------------------
        # Applied only to newly-arrived aligned segments.

        t_stage = time.perf_counter()

        new_nlp_segments = []
        new_texts = []

        for segment in new_segments:

            text = segment.get("text", "").strip()

            if not text:
                continue

            nlp_result = process_segment_text(text)

            new_nlp_segments.append(
                {
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "speaker": segment.get("speaker"),
                    "original_text": text,
                    "sentences": nlp_result["sentences"],
                    "tokens": nlp_result["tokens"],
                    "filtered_tokens": nlp_result["filtered_tokens"],
                    "pos_tags": nlp_result["pos_tags"],
                    "lemmatized_tokens": nlp_result[
                        "lemmatized_tokens"
                    ],
                }
            )

            new_texts.append(text)

        new_entities = (
            extract_entities(" ".join(new_texts)) if new_texts else []
        )

        with self._lock:

            self.nlp_processed_segments.extend(new_nlp_segments)

            for entity in new_entities:

                key = (entity["entity"], entity["type"])

                if key not in self._seen_entity_keys:

                    self._seen_entity_keys.add(key)
                    self.named_entities.append(entity)

            nlp_processed_segments_copy = list(
                self.nlp_processed_segments
            )

        timings["nlp_ms"] = (time.perf_counter() - t_stage) * 1000

        # -------------------- 7. Chunk summary (map step) --------------------

        t_stage = time.perf_counter()

        if new_segments:

            chunk_text_parts = [
                f"{get_speaker(segment)}: {get_text(segment)}"
                for segment in new_segments
                if get_text(segment)
            ]

            chunk_text = " ".join(chunk_text_parts).strip()

            if chunk_text:

                try:

                    sub_chunks = create_chunks(
                        chunk_text, self._summarizer_tokenizer
                    )

                    new_chunk_summaries = []

                    for sub_chunk in sub_chunks:

                        summary = summarize_text(
                            sub_chunk,
                            self._summarizer_tokenizer,
                            self._summarizer_model,
                            self._summarizer_device,
                        )

                        new_chunk_summaries.append(summary)

                    with self._lock:
                        self.chunk_summaries.extend(new_chunk_summaries)

                except Exception as error:
                    timings["chunk_summary_error"] = str(error)

        timings["chunk_summary_ms"] = (
            time.perf_counter() - t_stage
        ) * 1000

        # -------------------- 8. Periodic recompute --------------------
        # Whole-meeting summary (reduce step), TF-IDF keywords,
        # and LDA topics: all refit/regenerated over everything
        # accumulated so far, every N chunks -- not per chunk.

        with self._lock:
            self.chunk_count += 1
            chunk_count = self.chunk_count
            chunk_summaries_copy = list(self.chunk_summaries)

        did_recompute = (
            chunk_count % self.recompute_every_n_chunks == 0
        )

        t_stage = time.perf_counter()

        if did_recompute and chunk_summaries_copy:

            try:

                combined_summary = " ".join(chunk_summaries_copy)

                whole_summary = summarize_text(
                    combined_summary,
                    self._summarizer_tokenizer,
                    self._summarizer_model,
                    self._summarizer_device,
                )

                with self._lock:
                    self.whole_summary = whole_summary

            except Exception as error:
                timings["whole_summary_error"] = str(error)

        timings["whole_summary_ms"] = (
            time.perf_counter() - t_stage
        ) * 1000

        t_stage = time.perf_counter()

        if did_recompute and nlp_processed_segments_copy:

            documents, metadata = build_tfidf_documents(
                nlp_processed_segments_copy
            )

            tfidf_result = compute_tfidf_keywords(documents, metadata)

            with self._lock:
                self.tfidf_result = tfidf_result

        timings["tfidf_ms"] = (time.perf_counter() - t_stage) * 1000

        t_stage = time.perf_counter()

        if did_recompute and nlp_processed_segments_copy:

            documents, metadata = build_topic_documents(
                nlp_processed_segments_copy
            )

            topics_result = compute_topics(documents, metadata)

            with self._lock:
                self.topics_result = topics_result

        timings["topics_ms"] = (time.perf_counter() - t_stage) * 1000

        # -------------------- Latency accounting --------------------

        total_ms = (time.perf_counter() - t0) * 1000

        chunk_duration_s = len(chunk_audio) / SAMPLE_RATE

        real_time_factor = (
            total_ms / (chunk_duration_s * 1000)
            if chunk_duration_s > 0
            else 0.0
        )

        result_wall_clock = time.time()
        process_rss_mb = current_process_rss_mb()
        chunk_end_time = chunk_start_time + chunk_duration_s
        end_to_end_lag_seconds = max(
            0.0,
            result_wall_clock - self.capture_start_wall_time - chunk_end_time,
        )

        log_entry = {
            "chunk_index": chunk_index,
            "chunk_start_time": round(chunk_start_time, 3),
            "chunk_duration_s": round(chunk_duration_s, 3),
            "chunk_seconds": round(chunk_duration_s, 3),
            "wall_clock": time.time(),
            "had_speech": has_speech,
            "ran_diarization": ran_diarization,
            "diarization_buffer_s": round(buffer_seconds, 2),
            "did_periodic_recompute": did_recompute,
            "new_segment_count": len(new_segments),
            "total_ms": round(total_ms, 2),
            "real_time_factor": round(real_time_factor, 3),
            "alignment_nlp_ms": round(
                timings.get("alignment_ms", 0.0) + timings.get("nlp_ms", 0.0),
                2,
            ),
            "process_rss_mb": round(process_rss_mb, 2),
            "end_to_end_lag_seconds": round(end_to_end_lag_seconds, 3),
        }

        for key, value in timings.items():

            if isinstance(value, float):
                log_entry[key] = round(value, 2)
            else:
                log_entry[key] = value

        with self._lock:
            self.latency_log.append(log_entry)

        self._append_latency_log(log_entry)
        self._dump_state_snapshot()

        return log_entry

    # ------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------

    def _online_speaker_segments(self, chunk_audio, speech_timestamps, chunk_index, chunk_start_time):
        candidates = []
        for item in speech_timestamps:
            start_sample = int(item["start"])
            end_sample = int(item["end"])
            if chunk_index > 0:
                start_sample = max(start_sample, int(OVERLAP_SECONDS * SAMPLE_RATE))
            if end_sample <= start_sample:
                continue
            start = start_sample / SAMPLE_RATE
            end = end_sample / SAMPLE_RATE
            candidates.append({"start": start, "end": end, "speaker": None})

        long_indices = [i for i, row in enumerate(candidates) if row["end"] - row["start"] >= 1.0]
        vectors = embed_speech_segments(
            self._diarization_pipeline,
            chunk_audio,
            [(candidates[i]["start"], candidates[i]["end"]) for i in long_indices],
            SAMPLE_RATE,
        ) if long_indices else np.zeros((0, 0), dtype=np.float32)

        max_speakers = self.expected_speaker_count or 8

        for row_index, vector in zip(long_indices, vectors):
            if self.online_speaker_centroids:
                names = list(self.online_speaker_centroids)
                similarities = [float(np.dot(vector, self.online_speaker_centroids[name])) for name in names]
                nearest_index = int(np.argmax(similarities))
                if similarities[nearest_index] >= self.online_similarity_threshold or len(names) >= max_speakers:
                    speaker = names[nearest_index]
                else:
                    speaker = f"Speaker {len(names) + 1}"
            else:
                speaker = "Speaker 1"
            if speaker not in self.online_speaker_centroids:
                self.online_speaker_centroids[speaker] = vector.copy()
                self.online_speaker_updates[speaker] = 1
            else:
                centroid = 0.9 * self.online_speaker_centroids[speaker] + 0.1 * vector
                self.online_speaker_centroids[speaker] = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
                self.online_speaker_updates[speaker] += 1
            candidates[row_index]["speaker"] = speaker

        # Short VAD regions inherit the nearest long region in this
        # window; if none exists, they inherit the last online speaker.
        long_rows = [row for row in candidates if row["speaker"]]
        for row in candidates:
            if row["speaker"] is not None:
                continue
            if long_rows:
                midpoint = (row["start"] + row["end"]) / 2
                row["speaker"] = min(long_rows, key=lambda other: abs((other["start"] + other["end"]) / 2 - midpoint))["speaker"]
            else:
                row["speaker"] = self._last_online_speaker or "Speaker ?"
        if candidates:
            self._last_online_speaker = candidates[-1]["speaker"]
        return [
            {"start": round(chunk_start_time + row["start"], 3),
             "end": round(chunk_start_time + row["end"], 3),
             "speaker": row["speaker"]}
            for row in candidates
        ]

    def _provisional_speaker(self, segment):
        """Use the nearest known diarized turn as a provisional proxy."""
        with self._lock:
            if not self.diarization_segments:
                return "Speaker ?"
            midpoint = (segment["start"] + segment["end"]) / 2
            nearest = min(self.diarization_segments, key=lambda row: abs((row["start"] + row["end"]) / 2 - midpoint))
            return nearest["speaker"] if abs((nearest["start"] + nearest["end"]) / 2 - midpoint) <= 30 else "Speaker ?"

    def _request_diarization(self, audio, start_time):
        request = (audio.copy(), start_time, time.time())
        with self._lock:
            if self._diarization_running:
                self._pending_diarization = request
                return
            self._diarization_running = True
            self._diarization_started_at = request[2]
            self._diarization_thread = threading.Thread(target=self._diarization_worker, args=(request,), daemon=True)
            self._diarization_thread.start()

    def _diarization_worker(self, request):
        while request is not None:
            audio, start_time, requested_at = request
            temp_path = self._write_temp_wav(audio)
            try:
                local = diarize_audio(self._diarization_pipeline, temp_path)
                refined = [{"start": s["start"] + start_time, "end": s["end"] + start_time, "speaker": s["speaker"]} for s in local]
                with self._lock:
                    self.diarization_segments = refined
                    covered_end = start_time + len(audio) / SAMPLE_RATE
                    covered = [s for s in self.asr_segments if start_time <= (s["start"] + s["end"]) / 2 <= covered_end]
                    aligned = align_segments(covered, refined)
                    by_key = {(s["start"], s["end"], s["text"]): s for s in aligned}
                    for row in self.aligned_transcript:
                        key = (row["start"], row["end"], row["text"])
                        if key in by_key:
                            row["speaker"] = by_key[key]["speaker"]
                            row["speaker_overlap"] = by_key[key]["speaker_overlap"]
                            row["speakers_refined"] = True
                    self._diarization_refinement_seconds.append(max(0.0, time.time() - requested_at))
                    self._diarization_last_error = None
            except Exception as error:
                with self._lock:
                    self._diarization_last_error = str(error)
            finally:
                if os.path.exists(temp_path): os.remove(temp_path)
            with self._lock:
                request = self._pending_diarization
                self._pending_diarization = None
                if request is None:
                    self._diarization_running = False
                    self._diarization_thread = None


    def _write_temp_wav(self, audio):

        fd, path = tempfile.mkstemp(
            suffix=".wav", dir=self.output_dir
        )

        os.close(fd)

        sf.write(path, audio, SAMPLE_RATE)

        return path

    def _append_latency_log(self, entry):

        with open(
            self.latency_log_path, "a", encoding="utf-8"
        ) as file:

            file.write(json.dumps(entry) + "\n")

    def _dump_state_snapshot(self):

        with open(
            self.state_snapshot_path, "w", encoding="utf-8"
        ) as file:

            json.dump(
                self.snapshot(),
                file,
                indent=2,
                ensure_ascii=False,
            )

    def snapshot(self):
        """Thread-safe read of the current session state, for display
        in a UI thread that is not the thread ingest_chunk() runs on."""

        with self._lock:

            return {
                "chunk_count": self.chunk_count,
                "elapsed_wall_seconds": round(
                    time.time() - self.session_start_wall_time, 1
                ),
                "transcript": list(self.aligned_transcript),
                "chunk_summaries": list(self.chunk_summaries),
                "whole_summary": self.whole_summary,
                "action_items": list(self.action_items),
                "decisions": list(self.decisions),
                "keywords": self.tfidf_result,
                "topics": self.topics_result,
                "named_entities": list(self.named_entities),
                "latest_latency": (
                    self.latency_log[-1]
                    if self.latency_log
                    else None
                ),
                "diarization_running": self._diarization_running,
                "diarization_last_error": self._diarization_last_error,
                "stop_diarization_timed_out": self.stop_diarization_timed_out,
                "diarization_refinement_seconds": list(self._diarization_refinement_seconds),
                "online_speaker_count": len(self.online_speaker_centroids),
                "online_similarity_threshold": self.online_similarity_threshold,
                "expected_speaker_count": self.expected_speaker_count,
                "embedding_clustering_ms": list(self.embedding_clustering_ms),
            }


# ============================================================
# STANDALONE SELF-TEST (simulated file playback)
# ============================================================
#
# Proves the incremental pipeline produces growing output chunk by
# chunk on a real recording, without a live microphone. This is a
# functional self-test for Stage 3, not the full Stage 5 evaluation
# (which measures a complete run and writes up the honest findings).

if __name__ == "__main__":

    import argparse

    from live_capture import SimulatedFileCapture

    parser = argparse.ArgumentParser(
        description="Live pipeline self-test (simulated file playback)"
    )

    parser.add_argument(
        "--file", default="data/audio/meeting.wav"
    )

    parser.add_argument(
        "--speed-factor", type=float, default=1.0
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Stop after this many chunks (for quick smoke testing)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("LIVE PIPELINE SELF-TEST (simulated file playback)")
    print("=" * 70)

    session = LiveMeetingSession()

    stop_event = threading.Event()

    def on_chunk(chunk_audio, chunk_index, chunk_start_time):

        if (
            args.max_chunks is not None
            and chunk_index >= args.max_chunks
        ):
            stop_event.set()
            return

        log_entry = session.ingest_chunk(
            chunk_audio, chunk_index, chunk_start_time
        )

        print(
            f"\nchunk {log_entry['chunk_index']:3d} "
            f"| t={log_entry['chunk_start_time']:7.1f}s "
            f"| speech={log_entry['had_speech']} "
            f"| diarized={log_entry['ran_diarization']} "
            f"| new_segments={log_entry['new_segment_count']} "
            f"| total={log_entry['total_ms']:8.1f}ms "
            f"| RTF={log_entry['real_time_factor']:.2f}"
        )

        print(
            f"    vad={log_entry.get('vad_ms', 0):.1f}ms "
            f"asr={log_entry.get('asr_ms', 0):.1f}ms "
            f"diar={log_entry.get('diarization_ms', 0):.1f}ms "
            f"align={log_entry.get('alignment_ms', 0):.1f}ms "
            f"nlp={log_entry.get('nlp_ms', 0):.1f}ms "
            f"chunk_sum={log_entry.get('chunk_summary_ms', 0):.1f}ms"
        )

        if session.aligned_transcript:

            latest = session.aligned_transcript[-1]

            print(
                f"    latest: [{latest['speaker']}] "
                f"{latest['text'][:80]}"
            )

        if log_entry["did_periodic_recompute"]:

            print(
                f"    -- recomputed whole summary / keywords / "
                f"topics at chunk {log_entry['chunk_index']} --"
            )

            print(f"    whole_summary: {session.whole_summary[:150]}")

        if stop_event.is_set():
            raise KeyboardInterrupt

    capture = SimulatedFileCapture(
        args.file, on_chunk, speed_factor=args.speed_factor
    )

    try:

        capture.start()
        capture.join()

    except KeyboardInterrupt:

        capture.stop()

    print()
    print("=" * 70)
    print(f"Chunks processed: {session.chunk_count}")
    print(f"Transcript segments: {len(session.aligned_transcript)}")
    print(f"Action items: {len(session.action_items)}")
    print(f"Decisions: {len(session.decisions)}")
    print(f"Chunk summaries: {len(session.chunk_summaries)}")
    print(f"Latency log: {session.latency_log_path}")
    print(f"State snapshot: {session.state_snapshot_path}")
    print("=" * 70)
