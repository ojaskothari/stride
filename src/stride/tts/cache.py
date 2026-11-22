"""Render cue clips once with macOS ``say -o``, trim silence, cache them on disk.

Stdlib only (no numpy), so it is testable in CI with a fake renderer. Spawning ``say`` per cue costs
~1.2 s (docs/spikes.md §1); playing a cached clip starts in ~35 ms.
"""

from __future__ import annotations

import array
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
import wave
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

SAMPLE_RATE = 22_050
_VERSION = "2"  # bump to invalidate every cached clip (e.g. after changing trimming)
TRIM_THRESHOLD = 300  # int16 amplitude (~ -40 dBFS) that counts as sound
PAD_S = 0.02  # keep a little silence around the trimmed clip so onsets aren't clipped
BATCH_GAP_MS = 500  # silence between phrases in a batched render


@dataclass(frozen=True)
class Clip:
    samples: array.array[int]  # mono int16
    sample_rate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.sample_rate


class Renderer(Protocol):
    def __call__(self, text: str, voice: str, out: Path, sample_rate: int) -> None: ...


def say_render(text: str, voice: str, out: Path, sample_rate: int) -> None:
    cmd = ["say", "-o", str(out), "--file-format=WAVE", f"--data-format=LEI16@{sample_rate}"]
    if voice:
        cmd += ["-v", voice]
    subprocess.run([*cmd, "--", text], check=True, capture_output=True)


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "stride" / "tts"


def trim(samples: array.array[int], sample_rate: int) -> array.array[int]:
    loud = [i for i, v in enumerate(samples) if abs(v) > TRIM_THRESHOLD]
    if not loud:
        return samples
    pad = int(PAD_S * sample_rate)
    return samples[max(0, loud[0] - pad) : loud[-1] + 1 + pad]


def split_on_silence(
    samples: array.array[int], sample_rate: int, min_gap_s: float
) -> list[array.array[int]]:
    """Split into sounding segments separated by at least ``min_gap_s`` of silence (padded)."""
    min_gap = int(min_gap_s * sample_rate)
    pad = int(PAD_S * sample_rate)
    bounds: list[tuple[int, int]] = []
    start = last = -1
    for i, v in enumerate(samples):
        if abs(v) > TRIM_THRESHOLD:
            if start < 0:
                start = i
            elif i - last > min_gap:
                bounds.append((start, last))
                start = i
            last = i
    if start >= 0:
        bounds.append((start, last))
    return [samples[max(0, a - pad) : b + 1 + pad] for a, b in bounds]


def read_wav(path: Path) -> Clip:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono 16-bit audio")
        samples = array.array("h", w.readframes(w.getnframes()))
        rate = w.getframerate()
    if sys.byteorder == "big":  # pragma: no cover
        samples.byteswap()
    return Clip(samples, rate)


def write_wav(path: Path, clip: Clip) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(clip.sample_rate)
        w.writeframes(clip.samples.tobytes())


class ClipCache:
    def __init__(
        self,
        root: Path,
        voice: str = "",
        render: Renderer = say_render,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.root = root
        self.voice = voice
        self._render = render
        self._rate = sample_rate
        self._mem: dict[str, Clip] = {}
        root.mkdir(parents=True, exist_ok=True)

    def path(self, text: str) -> Path:
        key = hashlib.sha1(
            f"{_VERSION}\0{self.voice}\0{self._rate}\0{text}".encode(), usedforsecurity=False
        ).hexdigest()
        return self.root / f"{key}.wav"

    def has(self, text: str) -> bool:
        return text in self._mem or self.path(text).exists()

    def get(self, text: str) -> Clip:
        """The clip for ``text``, rendering it now if it isn't cached (slow: ~1 s)."""
        if (clip := self._mem.get(text)) is not None:
            return clip
        path = self.path(text)
        if not path.exists():
            self._render_to(text, path)
        clip = read_wav(path)
        self._mem[text] = clip
        return clip

    def prerender(
        self,
        texts: Iterable[str],
        workers: int = 4,
        progress: Callable[[int, int], None] | None = None,
    ) -> int:
        """Render every missing clip. Returns how many were rendered.

        Phrases are rendered in batches, one ``say`` call per batch with long silences between
        phrases, then split on those silences: 315 clips take ~53 s instead of ~162 s one call
        each (docs/spikes.md §4). If a batch doesn't split into exactly one segment per phrase,
        that batch falls back to one render per phrase, so a bad split can never mislabel a clip.
        """
        missing = sorted(t for t in set(texts) if not self.path(t).exists())
        if not missing:
            return 0
        chunks = [missing[i::workers] for i in range(workers) if missing[i::workers]]
        done = 0
        lock = threading.Lock()

        def run(chunk: list[str]) -> None:
            nonlocal done
            self._render_batch(chunk)
            with lock:
                done += len(chunk)
                if progress:
                    progress(done, len(missing))

        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            list(pool.map(run, chunks))
        return len(missing)

    def _render_batch(self, chunk: list[str]) -> None:
        if len(chunk) == 1:
            self._render_to(chunk[0], self.path(chunk[0]))
            return
        script = f" [[slnc {BATCH_GAP_MS}]] ".join(f"{t}." for t in chunk)
        fd, tmp_name = tempfile.mkstemp(suffix=".wav", dir=self.root)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            self._render(script, self.voice, tmp, self._rate)
            raw = read_wav(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        segments = split_on_silence(raw.samples, raw.sample_rate, BATCH_GAP_MS / 1000 * 0.6)
        if len(segments) != len(chunk):
            for t in chunk:
                self._render_to(t, self.path(t))
            return
        for text, seg in zip(chunk, segments, strict=True):
            self._write_atomic(self.path(text), Clip(seg, raw.sample_rate))

    def _write_atomic(self, path: Path, clip: Clip) -> None:
        fd, tmp_name = tempfile.mkstemp(suffix=".wav", dir=self.root)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            write_wav(tmp, clip)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _render_to(self, text: str, path: Path) -> None:
        # Render to a temp file and rename, so a crash or a parallel render never leaves a
        # half-written clip in the cache.
        fd, tmp_name = tempfile.mkstemp(suffix=".wav", dir=self.root)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            self._render(f"{text}.", self.voice, tmp, self._rate)
            raw = read_wav(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        self._write_atomic(path, Clip(trim(raw.samples, raw.sample_rate), raw.sample_rate))
