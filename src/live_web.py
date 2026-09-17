"""
Browser-based live microphone capture for MeetSense (Path B).

Stage D scope only: get audio frames from the browser microphone into
Python via streamlit-webrtc and prove they arrive. Whisper/PyAnnote
processing is added in later stages -- this module intentionally does
not transcribe or diarize anything yet.

This is a new, additive module. It does not modify or replace the
existing local-microphone (sounddevice) live scripts
(live_transcribe.py, live_meeting.py, live_speaker_transcribe.py in
this same directory), which remain the local development/testing path
(Path A). This module is Path B: the browser/WebRTC path used by the
deployed MeetSense workspace.
"""

import threading

import av
import numpy as np
import streamlit as st
import whisper

from streamlit_webrtc import AudioProcessorBase


# Matches the production ASR model (src/transcription.py uses "base";
# the local live path's live_speaker_transcribe.py uses "small" for a
# faster rolling-window turnaround -- "base" is kept here to stay
# consistent with the offline pipeline's own default).
WHISPER_MODEL_SIZE = "base"

# Seconds of audio accumulated before each Whisper call. Matches the
# CHUNK_SECONDS already used by the existing local live scripts
# (live_transcribe.py, live_speaker_transcribe.py).
LIVE_CHUNK_SECONDS = 10

LIVE_SAMPLE_RATE = 16000


@st.cache_resource(show_spinner="Loading Whisper for live transcription...")
def load_live_whisper_model():
    """Load Whisper once per Streamlit process and reuse it across
    reruns and audio callbacks, instead of reloading it per chunk."""

    return whisper.load_model(WHISPER_MODEL_SIZE)


class LiveAudioReceiver(AudioProcessorBase):
    """Receives browser microphone audio frames via WebRTC, buffers
    them into fixed-size chunks, and transcribes each chunk with
    Whisper once a model is attached.

    Stage D counted frames only. Stage E/F add: resampling incoming
    frames to a canonical 16kHz mono format (via av.AudioResampler,
    since WebRTC frames can arrive in varying sample rates/channel
    layouts depending on the browser), buffering LIVE_CHUNK_SECONDS of
    audio, and running Whisper on each completed chunk.

    No PyAnnote/diarization yet -- that is a later stage. Segments are
    stored without a speaker label at this point.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Frame-level stats (kept from Stage D for the status display).
        self.frame_count = 0
        self.total_samples = 0
        self.last_sample_rate = None
        self.last_channels = None

        # Resamples arbitrary incoming WebRTC audio to 16kHz mono
        # 16-bit PCM, the format Whisper expects.
        self._resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=LIVE_SAMPLE_RATE,
        )

        self._pending_samples = []  # list of int16 numpy arrays at 16kHz mono
        self._pending_sample_count = 0

        self._model = None
        self._elapsed_seconds = 0.0

        self.transcript_segments = []  # [{start, end, text}]
        self.last_error = None

    def attach_model(self, model) -> None:
        """Called once from the Streamlit main thread after the
        webrtc component has been created, so the (cached, already
        loaded) Whisper model is available to the audio callback
        thread without reloading it."""

        with self._lock:
            self._model = model

    def recv(self, frame):

        with self._lock:
            self.frame_count += 1
            self.total_samples += frame.samples
            self.last_sample_rate = frame.sample_rate
            self.last_channels = len(frame.layout.channels)

        try:
            resampled_frames = self._resampler.resample(frame)
        except Exception as error:
            with self._lock:
                self.last_error = f"Resampling failed: {error}"
            return frame

        ready_chunk = None

        with self._lock:

            for resampled in resampled_frames:

                samples = resampled.to_ndarray()
                self._pending_samples.append(samples)
                self._pending_sample_count += samples.shape[-1]

            if (
                self._model is not None
                and self._pending_sample_count
                >= LIVE_CHUNK_SECONDS * LIVE_SAMPLE_RATE
            ):

                ready_chunk = np.concatenate(
                    self._pending_samples, axis=-1
                )

                self._pending_samples = []
                self._pending_sample_count = 0

        if ready_chunk is not None:
            self._transcribe_chunk(ready_chunk)

        # Passed through unchanged. In SENDONLY mode nothing is sent
        # back to the browser, but the callback must still return a
        # valid frame.
        return frame

    def _transcribe_chunk(self, int16_samples) -> None:

        # int16 PCM -> float32 in [-1, 1], the format Whisper expects
        # when given a numpy array directly.
        audio_float32 = (
            int16_samples.astype(np.float32).flatten() / 32768.0
        )

        chunk_start = self._elapsed_seconds
        chunk_end = chunk_start + (
            len(audio_float32) / LIVE_SAMPLE_RATE
        )

        text = ""

        try:

            result = self._model.transcribe(
                audio_float32,
                language=None,
                task="transcribe",
                fp16=False,
                verbose=False,
                condition_on_previous_text=False,
            )

            text = result.get("text", "").strip()

        except Exception as error:

            with self._lock:
                self.last_error = f"Transcription failed: {error}"

        with self._lock:

            self._elapsed_seconds = chunk_end

            if text:

                self.transcript_segments.append({
                    "start": round(chunk_start, 2),
                    "end": round(chunk_end, 2),
                    "text": text,
                })

    def snapshot(self):
        """Thread-safe read of the current counters and transcript,
        for display in the Streamlit UI thread (which is not the same
        thread recv() runs on)."""

        with self._lock:

            seconds_received = (
                round(self.total_samples / self.last_sample_rate, 2)
                if self.last_sample_rate
                else 0.0
            )

            return {
                "frame_count": self.frame_count,
                "total_samples": self.total_samples,
                "sample_rate": self.last_sample_rate,
                "channels": self.last_channels,
                "seconds_received": seconds_received,
                "transcript_segments": list(self.transcript_segments),
                "last_error": self.last_error,
            }
