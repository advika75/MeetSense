"""
Live audio capture for MeetSense's live meeting mode.

This is the Stage 2 capture layer: it turns a continuous audio source
(the local microphone, via sounddevice -- or, for reproducible testing,
a pre-recorded file played back at real-time speed) into a stream of
overlapping rolling-window chunks. It does no ASR, diarization, or NLP
itself -- it only produces chunks and hands them to a callback. Stage 3
attaches the incremental pipeline to that callback.

Two capture sources share the same RollingChunker/on_chunk interface:

- MicrophoneCapture: real microphone input via sounddevice.InputStream,
  for the actual live demo.
- SimulatedFileCapture: replays a saved .wav file with the same pacing
  a live microphone would produce, so the incremental pipeline (and its
  latency measurements) can be exercised deterministically without a
  live mic -- used for Stage 5 verification.

This module is additive: it does not modify or replace the existing
sounddevice-based scripts (live_transcribe.py, live_meeting.py,
live_speaker_transcribe.py) or the browser-based live_web.py. Those are
left untouched as prior art / reference.
"""

import os
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from audio_preprocessing import convert_to_mono, resample_audio


# ============================================================
# CONFIGURATION
# ============================================================
#
# 20s chunks with 3s overlap: long enough to give Whisper/diarization
# useful context, short enough to keep "live" latency reasonable. The
# 3s overlap exists so words/speaker turns that straddle a chunk
# boundary are not clipped -- the pipeline stage that consumes chunks
# is responsible for trimming the duplicated overlap region back out
# of the growing transcript.

SAMPLE_RATE = 16000

CHUNK_SECONDS = 20.0

OVERLAP_SECONDS = 3.0

STEP_SECONDS = CHUNK_SECONDS - OVERLAP_SECONDS

CHUNK_SAMPLES = int(round(CHUNK_SECONDS * SAMPLE_RATE))

STEP_SAMPLES = int(round(STEP_SECONDS * SAMPLE_RATE))


# ============================================================
# ROLLING CHUNKER
# ============================================================

class RollingChunker:
    """Assembles a continuous stream of small audio frames into fixed-
    length, overlapping chunks, calling on_chunk(audio, chunk_index,
    chunk_start_time) each time a chunk is ready.

    Transport-agnostic: callers push raw float32 mono frames at
    SAMPLE_RATE via push(). Correct under a single producer thread;
    callers with multiple producer threads must serialize their own
    calls to push().
    """

    def __init__(
        self,
        on_chunk,
        chunk_seconds=CHUNK_SECONDS,
        overlap_seconds=OVERLAP_SECONDS,
        sample_rate=SAMPLE_RATE,
    ):
        if overlap_seconds >= chunk_seconds:
            raise ValueError(
                "overlap_seconds must be smaller than chunk_seconds"
            )

        self._on_chunk = on_chunk

        self._sample_rate = sample_rate

        self._chunk_samples = int(
            round(chunk_seconds * sample_rate)
        )

        self._step_samples = int(
            round((chunk_seconds - overlap_seconds) * sample_rate)
        )

        self._step_seconds = chunk_seconds - overlap_seconds

        self._frames = []
        self._frames_total = 0

        self._chunk_index = 0

        self._lock = threading.Lock()

    def push(self, audio):
        """Feed newly-arrived audio samples (float32, mono, at
        self._sample_rate). May trigger zero or more on_chunk calls,
        made after the internal lock is released."""

        ready_chunks = []

        with self._lock:

            self._frames.append(np.asarray(audio, dtype=np.float32))
            self._frames_total += len(audio)

            while self._frames_total >= self._chunk_samples:

                combined = np.concatenate(self._frames)

                chunk = combined[: self._chunk_samples].copy()

                remainder = combined[self._step_samples:]

                self._frames = [remainder] if len(remainder) else []
                self._frames_total = len(remainder)

                chunk_start_time = self._chunk_index * self._step_seconds

                ready_chunks.append(
                    (chunk, self._chunk_index, chunk_start_time)
                )

                self._chunk_index += 1

        for chunk, chunk_index, chunk_start_time in ready_chunks:
            self._on_chunk(chunk, chunk_index, chunk_start_time)

    def flush(self):
        """Emit whatever partial audio remains as a final, shorter
        chunk. Call once when the audio source ends (mic stopped, or
        end of a simulated file)."""

        with self._lock:

            if not self._frames or self._frames_total == 0:
                return

            combined = np.concatenate(self._frames)

            chunk_start_time = self._chunk_index * self._step_seconds

            chunk_index = self._chunk_index

            self._frames = []
            self._frames_total = 0
            self._chunk_index += 1

        self._on_chunk(combined, chunk_index, chunk_start_time)


# ============================================================
# MICROPHONE CAPTURE (real live demo)
# ============================================================

class MicrophoneCapture:
    """Captures from the local microphone via sounddevice and feeds a
    RollingChunker. The sounddevice callback (which runs on
    PortAudio's own real-time thread) only enqueues raw frames -- all
    chunk assembly and the on_chunk callback happen on a separate
    drainer thread, so a slow on_chunk handler cannot cause audio
    dropouts in the callback itself.
    """

    def __init__(
        self,
        on_chunk,
        device=None,
        chunk_seconds=CHUNK_SECONDS,
        overlap_seconds=OVERLAP_SECONDS,
        sample_rate=SAMPLE_RATE,
        blocksize=4000,
    ):
        self._sample_rate = sample_rate

        self._device = device

        self._blocksize = blocksize

        self._chunker = RollingChunker(
            on_chunk=on_chunk,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            sample_rate=sample_rate,
        )

        self._frame_queue = queue.Queue()

        self._stream = None

        self._drainer_thread = None

        self._stop_event = threading.Event()

    def _audio_callback(self, indata, frames, time_info, status):

        if status:
            print(f"\n[live_capture] audio status: {status}")

        # Cheap: copy and enqueue only. No chunk assembly here.
        self._frame_queue.put(indata[:, 0].copy())

    def _drain_loop(self):

        while not self._stop_event.is_set() or not self._frame_queue.empty():

            try:
                frame = self._frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self._chunker.push(frame)

    def start(self):

        if self._stream is not None:
            raise RuntimeError("MicrophoneCapture already started")

        self._stop_event.clear()

        self._drainer_thread = threading.Thread(
            target=self._drain_loop,
            daemon=True,
        )
        self._drainer_thread.start()

        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._blocksize,
            callback=self._audio_callback,
        )

        self._stream.start()

    def stop(self):

        if self._stream is None:
            return

        self._stream.stop()
        self._stream.close()
        self._stream = None

        self._stop_event.set()

        if self._drainer_thread is not None:
            self._drainer_thread.join(timeout=5.0)
            self._drainer_thread = None

        # Any audio shorter than one full chunk still gets processed.
        self._chunker.flush()


# ============================================================
# SIMULATED FILE CAPTURE (reproducible testing / Stage 5 evaluation)
# ============================================================

class SimulatedFileCapture:
    """Replays a saved .wav file as if it were a live microphone feed,
    pacing pushes to the RollingChunker in real time (or faster, via
    speed_factor, for quick iteration). Runs on a background thread so
    callers can start()/stop() it the same way as MicrophoneCapture.

    This exists so the incremental pipeline -- and the latency numbers
    measured from it -- can be exercised deterministically on a known
    recording, without depending on a live mic being available.
    """

    def __init__(
        self,
        file_path,
        on_chunk,
        chunk_seconds=CHUNK_SECONDS,
        overlap_seconds=OVERLAP_SECONDS,
        sample_rate=SAMPLE_RATE,
        speed_factor=1.0,
        push_seconds=0.5,
        max_duration_seconds=None,
    ):
        self._file_path = file_path

        self._sample_rate = sample_rate

        self._speed_factor = speed_factor
        self._max_duration_seconds = max_duration_seconds

        self._push_samples = int(round(push_seconds * sample_rate))

        self._chunker = RollingChunker(
            on_chunk=on_chunk,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            sample_rate=sample_rate,
        )

        self._thread = None

        self._stop_event = threading.Event()

    def _load_audio(self):

        import soundfile as sf

        audio, file_sample_rate = sf.read(self._file_path)

        audio = convert_to_mono(audio)

        audio = resample_audio(
            audio, file_sample_rate, self._sample_rate
        )

        return audio.astype(np.float32)

    def _run(self):

        audio = self._load_audio()

        total_samples = len(audio)
        if self._max_duration_seconds is not None:
            total_samples = min(
                total_samples,
                int(round(self._max_duration_seconds * self._sample_rate)),
            )

        position = 0

        push_interval_seconds = (
            self._push_samples / self._sample_rate
        ) / self._speed_factor

        while position < total_samples and not self._stop_event.is_set():

            end = min(position + self._push_samples, total_samples)

            self._chunker.push(audio[position:end])

            position = end

            time.sleep(push_interval_seconds)

        if not self._stop_event.is_set():
            self._chunker.flush()

    def start(self):

        if self._thread is not None:
            raise RuntimeError("SimulatedFileCapture already started")

        self._stop_event.clear()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def join(self):
        """Block until playback of the file finishes on its own."""

        if self._thread is not None:
            self._thread.join()


# ============================================================
# DEVICE LISTING HELPER
# ============================================================

def list_input_devices():

    devices = sd.query_devices()

    return [
        {"index": index, "name": device["name"]}
        for index, device in enumerate(devices)
        if device.get("max_input_channels", 0) > 0
    ]


# ============================================================
# STANDALONE SMOKE TEST
# ============================================================
#
# Proves the rolling-window chunk assembly is correct (right chunk
# count, right start times, right overlap) without loading Whisper,
# pyannote, or touching a live microphone. Stage 3 wires a real
# pipeline into on_chunk; this only reports chunk metadata.

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Live capture smoke test (simulated file playback)"
    )

    parser.add_argument(
        "--file",
        default="data/audio/meeting.wav",
        help="WAV file to replay as a simulated live feed",
    )

    parser.add_argument(
        "--speed-factor",
        type=float,
        default=20.0,
        help="Playback speed multiplier (1.0 = real time)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("LIVE CAPTURE SMOKE TEST (simulated file playback)")
    print("=" * 70)
    print(f"File: {args.file}")
    print(f"Chunk seconds: {CHUNK_SECONDS}")
    print(f"Overlap seconds: {OVERLAP_SECONDS}")
    print(f"Step seconds: {STEP_SECONDS}")
    print(f"Speed factor: {args.speed_factor}x")
    print()

    chunk_log = []

    def on_chunk(chunk, chunk_index, chunk_start_time):

        duration = len(chunk) / SAMPLE_RATE

        rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0

        chunk_log.append(
            (chunk_index, chunk_start_time, duration, rms)
        )

        print(
            f"chunk {chunk_index:3d} | "
            f"start={chunk_start_time:7.2f}s | "
            f"duration={duration:6.2f}s | "
            f"rms={rms:.4f}"
        )

    capture = SimulatedFileCapture(
        args.file,
        on_chunk,
        speed_factor=args.speed_factor,
    )

    capture.start()
    capture.join()

    print()
    print("=" * 70)
    print(f"Chunks emitted: {len(chunk_log)}")

    if len(chunk_log) >= 2:

        gaps = [
            round(chunk_log[i + 1][1] - chunk_log[i][1], 3)
            for i in range(len(chunk_log) - 1)
        ]

        print(f"Start-time deltas between chunks: {gaps}")
        print(f"Expected step seconds: {STEP_SECONDS}")

    print("=" * 70)

    if list_input_devices():

        print()
        print("Available microphone input devices:")

        for device in list_input_devices():
            print(f"  [{device['index']}] {device['name']}")
