"""Plays queued speech on a background thread through one persistent output stream.

Needs the ``[audio]`` extra (numpy, sounddevice). Never blocks the caller: ``say()`` enqueues.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from stride.tts.cache import SAMPLE_RATE, ClipCache
from stride.tts.catalog import resolve
from stride.tts.queue import SpeechQueue
from stride.workout import Speech

FRAGMENT_GAP_S = 0.04


class AudioPlayer:
    def __init__(
        self,
        cache: ClipCache,
        on_spoken: Callable[[Speech, float], None] | None = None,
    ) -> None:
        """``on_spoken(speech, started_at)`` is called as each speech starts (for latency logs)."""
        self._cache = cache
        self._on_spoken = on_spoken
        self.queue = SpeechQueue()
        self._lock = threading.Condition()
        self._stop = False
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", latency="low"
        )
        self._stream.start()
        self._gap = np.zeros(int(FRAGMENT_GAP_S * SAMPLE_RATE), dtype=np.int16)
        self._thread = threading.Thread(target=self._run, name="stride-tts", daemon=True)
        self._thread.start()

    @property
    def output_latency(self) -> float:
        """Seconds from writing audio to hearing it (as the driver reports it)."""
        return float(self._stream.latency)

    def say(self, speech: Speech) -> None:
        with self._lock:
            if speech.mark is None:
                self.queue.push_reply(speech)
            else:
                self.queue.push(speech)
            self._lock.notify()

    def hold(self) -> None:
        with self._lock:
            self.queue.hold()

    def release(self) -> None:
        with self._lock:
            self.queue.release()
            self._lock.notify()

    def close(self) -> None:
        with self._lock:
            self._stop = True
            self._lock.notify()
        self._thread.join(timeout=2)
        self._stream.stop()
        self._stream.close()

    def _pop(self) -> Speech | None:
        # Judge staleness at the moment the audio will be *heard*, not when it's written.
        return self.queue.pop(time.monotonic() + self.output_latency)

    def dropped(self) -> list[Speech]:
        with self._lock:
            out, self.queue.dropped = self.queue.dropped, []
        return out

    def _run(self) -> None:
        while True:
            with self._lock:
                speech = None
                while not self._stop and (speech := self._pop()) is None:
                    self._lock.wait(timeout=0.02)
                if self._stop:
                    return
            assert speech is not None
            clips = [self._cache.get(t) for t in resolve(speech.parts, self._cache.has)]
            chunks: list[np.ndarray] = []
            for k, c in enumerate(clips):
                if k:
                    chunks.append(self._gap)
                chunks.append(np.frombuffer(c.samples.tobytes(), dtype=np.int16))
            if self._on_spoken:
                self._on_spoken(speech, time.monotonic())
            self._stream.write(np.concatenate(chunks))


class SilentPlayer:
    """``tts.backend = "none"``: no audio (headless runs). The run loop prints replies."""

    output_latency = 0.0

    def __init__(self, on_spoken: Callable[[Speech, float], None] | None = None) -> None:
        self._on_spoken = on_spoken

    def say(self, speech: Speech) -> None:
        if self._on_spoken:
            self._on_spoken(speech, time.monotonic())

    def hold(self) -> None: ...

    def release(self) -> None: ...

    def close(self) -> None: ...
