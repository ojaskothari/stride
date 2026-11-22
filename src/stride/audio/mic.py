"""Always-open microphone stream. Needs the ``[audio]`` extra.

The stream stays open for the whole session so there's no device-open delay on a tap and the
pre-roll buffer always holds the last half second. The callback only copies frames onto a queue;
all processing happens in the consumer.
"""

from __future__ import annotations

import queue
import time

import numpy as np
import sounddevice as sd

from stride.audio.vad import FRAME, SAMPLE_RATE, Audio


class Mic:
    def __init__(self, device: int | str | None = None) -> None:
        self._q: queue.Queue[tuple[Audio, float]] = queue.Queue()
        self.overflows = 0
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME,
            channels=1,
            dtype="float32",
            latency="low",
            device=device,
            callback=self._callback,
        )

    def _callback(
        self, indata: np.ndarray, frames: int, _t: object, status: sd.CallbackFlags
    ) -> None:
        if status.input_overflow:
            self.overflows += 1
        self._q.put((indata[:, 0].copy(), time.monotonic()))

    @property
    def name(self) -> str:
        return str(sd.query_devices(self._stream.device)["name"])

    def start(self) -> None:
        self._stream.start()

    def read(self, timeout: float) -> list[tuple[Audio, float]]:
        """Every frame available now, waiting up to ``timeout`` for the first one."""
        out: list[tuple[Audio, float]] = []
        try:
            out.append(self._q.get(timeout=timeout))
            while True:
                out.append(self._q.get_nowait())
        except queue.Empty:
            pass
        return out

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
