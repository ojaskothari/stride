"""Spike: how much overhead does macOS `say` add on top of the audio itself?

overhead = wall time of `say <text>` - duration of the same text rendered to a file.
That overhead (mostly startup) is how late every spoken cue starts.
Usage: python spikes/say_latency.py [reps]
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

PHRASES = ["5", "Go", "10 seconds", "1 minute left", "Resting 1 30", "5, 4, 3, 2, 1, go"]


def audio_seconds(text: str, tmp: Path) -> float:
    out = tmp / "x.wav"
    subprocess.run(
        ["say", "-o", str(out), "--file-format=WAVE", "--data-format=LEI16@22050", text],
        check=True,
    )
    with wave.open(str(out)) as w:
        return w.getnframes() / w.getframerate()


def wall_seconds(text: str) -> float:
    t0 = time.perf_counter()
    subprocess.run(["say", text], check=True)
    return time.perf_counter() - t0


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cold = wall_seconds("ready") - audio_seconds("ready", tmp)
        print(f"cold first call overhead: {cold * 1000:.0f} ms\n")
        print(f"{'phrase':<22}{'audio':>8}{'overhead med':>14}{'max':>8}")
        for text in PHRASES:
            dur = audio_seconds(text, tmp)
            over = [wall_seconds(text) - dur for _ in range(reps)]
            print(
                f"{text!r:<22}{dur * 1000:>6.0f}ms"
                f"{statistics.median(over) * 1000:>12.0f}ms{max(over) * 1000:>6.0f}ms"
            )


if __name__ == "__main__":
    main()
