"""Session log: one JSON object per line. What `stride score` and the M6 validation read.

Raw audio is written only when ``save_audio`` is on (evaluation recording). Thread-safe: replies
are finalized from the audio thread when they start playing.
"""

from __future__ import annotations

import json
import re
import threading
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from stride.audio.vad import SAMPLE_RATE, Audio

SCHEMA_VERSION = 1


class SessionLog:
    def __init__(self, root: Path, env: str | None, save_audio: bool, now: datetime) -> None:
        stamp = now.strftime("%Y%m%d-%H%M%S")
        safe_env = re.sub(r"[^a-z0-9_-]+", "-", env.lower()) if env else ""
        self.stem = f"{stamp}-{safe_env}" if safe_env else stamp
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{self.stem}.jsonl"
        self.audio_dir = root / self.stem if save_audio else None
        if self.audio_dir:
            self.audio_dir.mkdir(exist_ok=True)
        self._f = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps({"v": SCHEMA_VERSION, **record}, default=_jsonable)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()  # a crash mid-workout must not lose the session

    def save_audio(self, utt_id: int, audio: Audio) -> str | None:
        """Write the utterance as 16-bit WAV if evaluation recording is on; return its name."""
        if self.audio_dir is None or len(audio) == 0:
            return None
        name = f"utt-{utt_id:03d}.wav"
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
        with wave.open(str(self.audio_dir / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
        return f"{self.stem}/{name}"

    def close(self) -> None:
        with self._lock:
            self._f.close()


def _jsonable(x: object) -> object:
    if isinstance(x, float | np.floating):
        return round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    raise TypeError(f"not JSON serializable: {type(x).__name__}")


def read_log(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
