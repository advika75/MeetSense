"""
Stage 4 orchestration glue for the Streamlit "Live Meeting" tab.

This module does not duplicate or modify the capture logic in
live_capture.py or the incremental-pipeline logic in live_pipeline.py
(aside from the one additive reset_state() method added there) -- it
only wires the two together with a bounded queue and a dedicated
worker thread, so the Streamlit UI thread never runs VAD/ASR/
diarization itself and never blocks on it.

Why a queue sits between capture and processing: both
MicrophoneCapture's drainer thread and SimulatedFileCapture's
playback thread call their on_chunk callback *synchronously*. If that
callback were LiveMeetingSession.ingest_chunk directly -- which took
up to ~160s in Stage 3 testing during a diarization spike -- it would
block the capture/drainer thread itself: incoming mic frames would
queue up unboundedly, and simulated playback couldn't pace itself at
all. So the capture-side callback here only enqueues a completed
chunk; a separate worker thread drains that queue and is the only
thread that calls into the (slow, sometimes very slow) pipeline.
"""

import json
import os
import queue
import threading
import time
import numpy as np
import soundfile as sf

from live_capture import MicrophoneCapture, SimulatedFileCapture, SAMPLE_RATE, OVERLAP_SECONDS
from live_pipeline import LiveMeetingSession


DEFAULT_CHUNK_QUEUE_MAXSIZE = 3

DEFAULT_OUTPUT_DIR = "outputs/live"


class LiveSessionController:
    """Owns one LiveMeetingSession (models loaded exactly once, in
    __init__) and at most one active capture source. The Streamlit UI
    talks to this class only through start()/stop()/reset()/
    save_session()/get_status() -- never touches the pipeline or
    capture modules directly."""

    def __init__(self, chunk_queue_maxsize=DEFAULT_CHUNK_QUEUE_MAXSIZE):

        self._status_lock = threading.Lock()

        # The one and only place models are loaded for this process.
        self.session = LiveMeetingSession()

        self._chunk_queue = queue.Queue(maxsize=chunk_queue_maxsize)

        self._capture = None
        self._worker_thread = None
        self._stop_event = threading.Event()

        self.mode = None
        self.running = False
        self.stopping = False

        self.captured_audio_seconds = 0.0
        self.processed_audio_seconds = 0.0
        self.dropped_chunk_count = 0
        self.merged_chunk_count = 0
        self._recorded_audio_parts = []
        self.session_audio_path = os.path.join(DEFAULT_OUTPUT_DIR, "live_session_audio.wav")
        self.last_error = None
        self.last_chunk_log = None

    # ------------------------------------------------------------
    # CAPTURE-SIDE CALLBACK -- must stay cheap. Runs on the capture
    # source's own thread (PortAudio drainer, or simulated-file
    # playback), which must keep up with real-time pacing.
    # ------------------------------------------------------------

    def _on_chunk_captured(self, chunk_audio, chunk_index, chunk_start_time):

        chunk_end_time = chunk_start_time + len(chunk_audio) / SAMPLE_RATE

        with self._status_lock:

            self.captured_audio_seconds = max(
                self.captured_audio_seconds, chunk_end_time
            )

            if chunk_index == 0:
                self._recorded_audio_parts = [np.asarray(chunk_audio, dtype=np.float32).copy()]
            else:
                overlap_samples = int(round(OVERLAP_SECONDS * SAMPLE_RATE))
                self._recorded_audio_parts.append(np.asarray(chunk_audio[overlap_samples:], dtype=np.float32).copy())

        try:
            self._chunk_queue.put_nowait((chunk_audio, chunk_index, chunk_start_time))
        except queue.Full:
            # Fold the entire queued backlog and this window into one
            # continuous clip, removing duplicated overlap samples.
            queued = []
            while True:
                try: queued.append(self._chunk_queue.get_nowait())
                except queue.Empty: break
            queued.append((chunk_audio, chunk_index, chunk_start_time))
            queued.sort(key=lambda item: item[2])
            first_audio, first_index, first_start = queued[0]
            pieces = [first_audio]
            last_end = first_start + len(first_audio) / SAMPLE_RATE
            for audio, _, start in queued[1:]:
                offset = max(0, int(round((last_end - start) * SAMPLE_RATE)))
                if offset < len(audio): pieces.append(audio[offset:])
                last_end = max(last_end, start + len(audio) / SAMPLE_RATE)
            merged = np.concatenate(pieces)
            self._chunk_queue.put_nowait((merged, first_index, first_start))
            with self._status_lock:
                self.merged_chunk_count += len(queued) - 1
            print(f"[live_controller] merged {len(queued)} queued windows into one ({len(merged)/SAMPLE_RATE:.1f}s)", flush=True)

    # ------------------------------------------------------------
    # PROCESSING WORKER -- the only thread that calls into the heavy
    # pipeline. Runs independently of capture/playback pacing.
    # ------------------------------------------------------------

    def _processing_loop(self):

        while not self._stop_event.is_set() or not self._chunk_queue.empty():

            try:
                chunk_audio, chunk_index, chunk_start_time = (
                    self._chunk_queue.get(timeout=0.2)
                )
            except queue.Empty:
                continue

            try:

                log_entry = self.session.ingest_chunk(
                    chunk_audio, chunk_index, chunk_start_time
                )

                chunk_end_time = (
                    chunk_start_time + len(chunk_audio) / SAMPLE_RATE
                )

                with self._status_lock:

                    self.processed_audio_seconds = max(
                        self.processed_audio_seconds, chunk_end_time
                    )

                    self.last_chunk_log = log_entry
                    self.last_error = None

            except Exception as error:

                with self._status_lock:
                    self.last_error = str(error)

    # ------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------

    def start(
        self, mode, simulated_file_path=None, device=None,
        simulated_max_duration_seconds=None,
    ):

        if self.running:
            return

        self._stop_event.clear()

        self.mode = mode

        with self._status_lock:
            self.last_error = None

        try:

            if mode == "microphone":

                self._capture = MicrophoneCapture(
                    self._on_chunk_captured, device=device
                )

            elif mode == "simulated":

                if not simulated_file_path:
                    raise ValueError(
                        "simulated_file_path is required for "
                        "simulated mode"
                    )

                self._capture = SimulatedFileCapture(
                    simulated_file_path,
                    self._on_chunk_captured,
                    speed_factor=1.0,
                    max_duration_seconds=simulated_max_duration_seconds,
                )

            else:
                raise ValueError(f"Unknown capture mode: {mode}")

            self._worker_thread = threading.Thread(
                target=self._processing_loop, daemon=True
            )
            self._worker_thread.start()

            self.session.capture_start_wall_time = time.time()
            self._capture.start()

            self.running = True

        except Exception as error:

            # Roll back whatever partially started, so a failed
            # start() leaves the controller in a clean, retriable
            # state rather than a half-started one.
            self._stop_event.set()

            if self._worker_thread is not None:
                self._worker_thread.join(timeout=2.0)

            self._worker_thread = None
            self._capture = None
            self.running = False

            with self._status_lock:
                self.last_error = str(error)

            raise

    def stop(self):
        """Stops capture immediately, then waits for the processing
        worker to finish its current item.

        NOTE: if a chunk is mid-flight inside the pipeline (a live
        Whisper/pyannote/T5 call), there is no safe way to interrupt
        a running model inference mid-call -- this blocks until it
        returns. Stage 3 testing saw diarization calls take up to
        ~160s in the worst case. Callers (the UI) must show a
        "stopping" state and not assume this returns quickly."""

        if not self.running:
            return

        self.stopping = True

        if self._capture is not None:
            self._capture.stop()

        self._stop_event.set()

        if self._worker_thread is not None:
            self._worker_thread.join()

        # Stop waits for the one in-flight diarization and its coalesced
        # latest request, but report construction remains bounded.
        diar_thread = self.session._diarization_thread
        if diar_thread is not None:
            diar_thread.join(timeout=120.0)
        self.session.stop_diarization_timed_out = bool(diar_thread and diar_thread.is_alive())

        self._capture = None
        self._worker_thread = None

        if self._recorded_audio_parts:
            os.makedirs(os.path.dirname(self.session_audio_path), exist_ok=True)
            sf.write(self.session_audio_path, np.concatenate(self._recorded_audio_parts), SAMPLE_RATE)

        self.running = False
        self.stopping = False

    def reset(self):
        """Clears all accumulated meeting state. Requires the
        session to be stopped first -- resetting while a capture
        thread might still be enqueuing chunks would race with the
        session's own internal state."""

        if self.running:
            raise RuntimeError(
                "Stop the session before resetting it"
            )

        self.session.reset_state()

        with self._status_lock:

            self.captured_audio_seconds = 0.0
            self.processed_audio_seconds = 0.0
            self.dropped_chunk_count = 0
            self.merged_chunk_count = 0
            self._recorded_audio_parts = []
            self.last_error = None
            self.last_chunk_log = None

    # ------------------------------------------------------------
    # SAVE SESSION
    # ------------------------------------------------------------

    def save_session(self, output_dir=DEFAULT_OUTPUT_DIR):
        """Writes the current session's transcript/summary/action-
        items/keywords/topics in the same JSON shapes the batch
        pipeline produces, under outputs/live/session_*.json -- a
        DIFFERENT path from the batch pipeline's own
        outputs/transcripts/, outputs/summaries/, outputs/nlp/ files,
        so this can never overwrite real batch results.

        Note on scope: the batch "Action Items" results PAGE in the
        app actually reads evidence_linking.py's enriched output, not
        action_item_extraction.py's raw shape directly. This writes
        the raw rule-based-extraction shape (matching
        action_items_decisions.json), which is what the paper's
        extraction-method tables use -- it will not, by itself,
        populate that specific results page, since evidence-linking
        is a separate downstream batch-only step out of this stage's
        scope."""

        os.makedirs(output_dir, exist_ok=True)

        snapshot = self.session.snapshot()

        paths = {}

        paths["transcript"] = os.path.join(
            output_dir, "session_speaker_transcript.json"
        )

        with open(paths["transcript"], "w", encoding="utf-8") as file:

            json.dump(
                snapshot["transcript"],
                file,
                indent=2,
                ensure_ascii=False,
            )

        paths["summary"] = os.path.join(
            output_dir, "session_meeting_summary.json"
        )

        with open(paths["summary"], "w", encoding="utf-8") as file:

            json.dump(
                {
                    "model": "google-t5/t5-small",
                    "source_file": "live_session",
                    "transcript_segments": len(
                        snapshot["transcript"]
                    ),
                    "number_of_chunks": len(
                        snapshot["chunk_summaries"]
                    ),
                    "chunk_summaries": snapshot["chunk_summaries"],
                    "final_summary": snapshot["whole_summary"],
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

        paths["action_items"] = os.path.join(
            output_dir, "session_action_items_decisions.json"
        )

        with open(
            paths["action_items"], "w", encoding="utf-8"
        ) as file:

            json.dump(
                {
                    "method": "Rule-based Information Extraction",
                    "total_segments": len(snapshot["transcript"]),
                    "number_of_action_items": len(
                        snapshot["action_items"]
                    ),
                    "number_of_decisions": len(
                        snapshot["decisions"]
                    ),
                    "action_items": snapshot["action_items"],
                    "decisions": snapshot["decisions"],
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

        paths["keywords"] = os.path.join(
            output_dir, "session_tfidf_keywords.json"
        )

        with open(paths["keywords"], "w", encoding="utf-8") as file:

            json.dump(
                {
                    "source_file": "live_session",
                    **snapshot["keywords"],
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

        paths["topics"] = os.path.join(
            output_dir, "session_topic_modeling.json"
        )

        with open(paths["topics"], "w", encoding="utf-8") as file:

            json.dump(
                {
                    "source_file": "live_session",
                    **snapshot["topics"],
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

        return paths

    # ------------------------------------------------------------
    # STATUS -- read by the Streamlit fragment roughly every 2s
    # ------------------------------------------------------------

    def get_status(self):

        with self._status_lock:

            lag_seconds = (
                self.captured_audio_seconds
                - self.processed_audio_seconds
            )

            status = {
                "mode": self.mode,
                "running": self.running,
                "stopping": self.stopping,
                "captured_audio_seconds": round(
                    self.captured_audio_seconds, 1
                ),
                "processed_audio_seconds": round(
                    self.processed_audio_seconds, 1
                ),
                "lag_seconds": round(lag_seconds, 1),
                "dropped_chunk_count": self.dropped_chunk_count,
                "queued_chunk_count": self._chunk_queue.qsize(),
                "last_error": self.last_error,
                "last_chunk_log": self.last_chunk_log,
            }

        status.update(self.session.snapshot())

        return status
