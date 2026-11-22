"""Spike: faster-whisper latency per model on this machine.

Synthetic `say` clips at 16 kHz: valid for timing, says nothing about accuracy (that's M4).
Usage: python spikes/stt_latency.py [reps]
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

COMMANDS = ["start my three minute rest", "add thirty seconds", "how much time left", "stop"]
PROMPT = "start my rest, rest 90, 3 minute rest, add 30 seconds, another minute, time left, stop."
MODELS = ["tiny.en", "base.en", "small.en"]


def clip(text: str, tmp: Path) -> np.ndarray:
    out = tmp / "c.wav"
    subprocess.run(
        ["say", "-o", str(out), "--file-format=WAVE", "--data-format=LEI16@16000", text], check=True
    )
    raw = out.read_bytes()
    return np.frombuffer(raw[raw.index(b"data") + 8 :], dtype="<i2").astype(np.float32) / 32768


def transcribe(model: WhisperModel, audio: np.ndarray, beam: int) -> tuple[float, str]:
    t0 = time.perf_counter()
    segs, _ = model.transcribe(
        audio,
        language="en",
        beam_size=beam,
        initial_prompt=PROMPT,
        without_timestamps=True,
        condition_on_previous_text=False,
    )
    text = " ".join(s.text.strip() for s in segs)  # generator: consuming it does the work
    return (time.perf_counter() - t0) * 1000, text


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"{platform.machine()} {platform.mac_ver()[0]}, {os.cpu_count()} cores\n")
    with tempfile.TemporaryDirectory() as d:
        clips = {c: clip(c, Path(d)) for c in COMMANDS}
    dur = statistics.mean(len(a) / 16_000 for a in clips.values())
    print(f"mean clip length {dur:.2f} s\n")
    print(f"{'model':<10}{'load':>8}{'beam':>6}{'median':>9}{'p90':>8}  transcripts")
    for name in MODELS:
        t0 = time.perf_counter()
        model = WhisperModel(name, device="cpu", compute_type="int8")
        load = time.perf_counter() - t0
        transcribe(model, clips[COMMANDS[0]], 1)  # warm-up
        for beam in (1, 5):
            times = [transcribe(model, a, beam)[0] for _ in range(reps) for a in clips.values()]
            texts = [transcribe(model, a, beam)[1] for a in clips.values()]
            p90 = statistics.quantiles(times, n=10)[-1]
            print(
                f"{name:<10}{load:>7.1f}s{beam:>6}{statistics.median(times):>7.0f}ms"
                f"{p90:>6.0f}ms  {texts}"
            )


if __name__ == "__main__":
    main()
