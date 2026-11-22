"""Spike: pre-render cue clips with `say -o`, trim silence, play with sounddevice.

Checks (1) trimmed clip lengths (countdown digits must fit inside 1 s) and
(2) playback start latency: call -> first output callback, plus the device's reported latency.
Usage: python spikes/prerendered_cues.py [voice]
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

SR = 22_050
CLIPS = ["5", "4", "3", "2", "1", "Go", "10 seconds", "30 seconds", "1 minute left", "Resting"]


def render(text: str, voice: str | None, tmp: Path) -> np.ndarray:
    out = tmp / "c.wav"
    cmd = ["say", "-o", str(out), "--file-format=WAVE", f"--data-format=LEI16@{SR}", text]
    if voice:
        cmd[1:1] = ["-v", voice]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    render_ms = (time.perf_counter() - t0) * 1000
    raw = out.read_bytes()
    pcm = np.frombuffer(raw[raw.index(b"data") + 8 :], dtype="<i2").astype(np.float32) / 32768
    print(f"  rendered {text!r} in {render_ms:.0f} ms")
    return pcm


def trim(x: np.ndarray, thresh: float = 0.01) -> np.ndarray:
    idx = np.flatnonzero(np.abs(x) > thresh)
    return x[idx[0] : idx[-1] + 1] if idx.size else x


def start_latency_ms(clip: np.ndarray) -> float:
    """Time from requesting playback to the first audio callback."""
    started = threading.Event()
    pos = 0

    def cb(out: np.ndarray, frames: int, _t: object, _s: sd.CallbackFlags) -> None:
        nonlocal pos
        started.set()
        chunk = clip[pos : pos + frames]
        out[: len(chunk), 0] = chunk
        out[len(chunk) :, 0] = 0
        pos += frames
        if pos >= len(clip):
            raise sd.CallbackStop

    stream = sd.OutputStream(samplerate=SR, channels=1, callback=cb, latency="low")
    t0 = time.perf_counter()
    stream.start()
    started.wait(2)
    ms = (time.perf_counter() - t0) * 1000
    time.sleep(len(clip) / SR + 0.1)
    stream.close()
    return ms


def main() -> None:
    voice = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"voice: {voice or 'system default'}")
    with tempfile.TemporaryDirectory() as d:
        clips = {t: render(t, voice, Path(d)) for t in CLIPS}
    print(f"\n{'clip':<16}{'raw':>8}{'trimmed':>9}")
    for t, x in clips.items():
        print(f"{t!r:<16}{len(x) / SR * 1000:>6.0f}ms{len(trim(x)) / SR * 1000:>7.0f}ms")

    dev = sd.query_devices(kind="output")
    print(
        f"\noutput device: {dev['name']}, reported low latency "
        f"{dev['default_low_output_latency'] * 1000:.1f} ms"
    )
    lat = [start_latency_ms(trim(clips[t])) for t in ["5", "4", "3", "2", "1", "Go"]]
    print(f"call -> first callback: median {statistics.median(lat):.1f} ms, max {max(lat):.1f} ms")


if __name__ == "__main__":
    main()
