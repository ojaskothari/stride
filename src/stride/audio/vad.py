"""Energy VAD and utterance segmentation. Pure (numpy only): frames in, utterances out.

Frames are 20 ms of mono float32 at 16 kHz, each stamped with the clock time its last sample
arrived. The live loop feeds microphone frames; tests feed synthetic ones.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt

SAMPLE_RATE = 16_000
FRAME = 320  # 20 ms
FRAME_S = FRAME / SAMPLE_RATE
SILENCE_DB = -120.0  # stands in for log(0)

Audio = npt.NDArray[np.float32]


class MicPermissionError(RuntimeError):
    """The input stream delivers pure digital silence: macOS denied mic access silently."""


def frame_db(frame: Audio) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
    return 20 * np.log10(rms) if rms > 0 else SILENCE_DB


def check_signal(frames: list[Audio], discard_s: float = 0.2) -> None:
    """Raise if the mic delivers only zeros. The first ``discard_s`` are startup silence even when
    permission is granted (measured at -173 dBFS, docs/spikes.md §3), so they don't count."""
    skip = round(discard_s / FRAME_S)
    rest = frames[skip:]
    if rest and not any(np.any(f) for f in rest):
        raise MicPermissionError(
            "the microphone is returning silence. Grant access to your terminal app in "
            "System Settings > Privacy & Security > Microphone, then restart the terminal."
        )


class NoiseFloor:
    """Rolling low percentile of recent frame levels. Updated only while idle, so the user's own
    speech never raises it, but it follows the environment (a crowd arriving, wind picking up)."""

    def __init__(self, window_s: float = 2.0, percentile: float = 20.0) -> None:
        self._levels: deque[float] = deque(maxlen=round(window_s / FRAME_S))
        self._pct = percentile

    def update(self, db: float) -> None:
        self._levels.append(db)

    @property
    def ready(self) -> bool:
        return len(self._levels) >= 10  # 200 ms of idle audio

    @property
    def db(self) -> float:
        if not self._levels:
            return SILENCE_DB
        return float(np.percentile(np.fromiter(self._levels, dtype=np.float64), self._pct))


class EndReason(StrEnum):
    SILENCE = "silence"  # speech, then silence_end_ms of quiet
    MAX_LENGTH = "max_length"
    NO_SPEECH = "no_speech"  # nothing loud enough within start_timeout_s
    TAP = "tap"  # the user tapped again to end the utterance


@dataclass(frozen=True)
class Utterance:
    audio: Audio
    reason: EndReason
    t_listen: float  # when listening started (the tap)
    t_done: float  # when the segmenter decided the utterance was over
    t_speech_start: float | None
    t_speech_end: float | None  # end of the last loud frame
    floor_db: float
    peak_db: float

    @property
    def hangover_s(self) -> float | None:
        """End of speech -> end detected: the silence wait, the biggest latency cost."""
        return None if self.t_speech_end is None else self.t_done - self.t_speech_end


@dataclass
class Segmenter:
    """Feed frames after a tap; returns an :class:`Utterance` once it ends."""

    t_listen: float
    floor_db: float
    threshold_db: float = 10.0
    silence_end_s: float = 0.6
    max_utterance_s: float = 15.0
    start_timeout_s: float = 3.0
    min_speech_frames: int = 3  # 60 ms: a tap or click is one or two frames
    tail_s: float = 0.2  # silence kept after speech for the recognizer
    lead_s: float = 0.3  # audio kept before the onset (soft first consonants fall below the gate)

    _frames: list[Audio] = field(default_factory=list)
    _times: list[float] = field(default_factory=list)
    _run: int = 0
    _start_idx: int | None = None
    _last_loud_idx: int | None = None
    _peak: float = SILENCE_DB

    @property
    def gate_db(self) -> float:
        return self.floor_db + self.threshold_db

    def push(self, frame: Audio, t_end: float) -> Utterance | None:
        self._frames.append(frame)
        self._times.append(t_end)
        i = len(self._frames) - 1
        db = frame_db(frame)
        loud = db > self.gate_db

        if loud:
            self._run += 1
            self._last_loud_idx = i
            if self._start_idx is None and self._run >= self.min_speech_frames:
                self._start_idx = i - self._run + 1
        else:
            self._run = 0
        if self._start_idx is not None:
            self._peak = max(self._peak, db)

        if self._start_idx is None:
            if t_end - self.t_listen >= self.start_timeout_s:
                return self._finish(EndReason.NO_SPEECH, t_end)
            return None
        assert self._last_loud_idx is not None
        if t_end - self._times[self._last_loud_idx] >= self.silence_end_s:
            return self._finish(EndReason.SILENCE, t_end)
        if t_end - self._frame_start(self._start_idx) >= self.max_utterance_s:
            return self._finish(EndReason.MAX_LENGTH, t_end)
        return None

    def finish(self, now: float) -> Utterance:
        """End now (a second tap). Skips the silence wait: the fastest way to get a reply."""
        if self._start_idx is None:
            return self._finish(EndReason.NO_SPEECH, now)
        return self._finish(EndReason.TAP, now)

    def _frame_start(self, i: int) -> float:
        return self._times[i] - FRAME_S

    def _finish(self, reason: EndReason, t_done: float) -> Utterance:
        if self._start_idx is None:
            audio: Audio = np.zeros(0, dtype=np.float32)
            start = end = None
        else:
            assert self._last_loud_idx is not None
            first = max(0, self._start_idx - round(self.lead_s / FRAME_S))
            last = self._last_loud_idx + 1 + round(self.tail_s / FRAME_S)
            audio = np.concatenate(self._frames[first:last]).astype(np.float32, copy=False)
            start = self._frame_start(self._start_idx)
            end = self._times[self._last_loud_idx]
        return Utterance(
            audio=audio,
            reason=reason,
            t_listen=self.t_listen,
            t_done=t_done,
            t_speech_start=start,
            t_speech_end=end,
            floor_db=self.floor_db,
            peak_db=self._peak,
        )
