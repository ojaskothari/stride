"""Spike: can this process read the mic? macOS returns silence (not an error) when denied.

Records a few seconds and reports the input device, the level, and whether it's all zeros.
Usage: python spikes/mic_check.py [seconds]
"""

from __future__ import annotations

import sys

import numpy as np
import sounddevice as sd

SR = 16_000


def dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20 * np.log10(rms) if rms > 0 else float("-inf")


def main() -> None:
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    dev = sd.query_devices(kind="input")
    print(
        f"input device: {dev['name']} ({dev['max_input_channels']} ch, "
        f"default {dev['default_samplerate']:.0f} Hz)"
    )
    audio = sd.rec(int(secs * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    x = audio[:, 0]
    frames = x[: len(x) // 320 * 320].reshape(-1, 320)  # 20 ms frames
    levels = [dbfs(f) for f in frames]
    print(f"overall {dbfs(x):.1f} dBFS, peak {20 * np.log10(np.abs(x).max() + 1e-12):.1f} dBFS")
    print(
        f"20 ms frames: min {min(levels):.1f}, median {np.median(levels):.1f}, "
        f"max {max(levels):.1f} dBFS"
    )
    if not np.any(x):
        print(
            "ALL ZEROS: mic permission is probably denied for this app "
            "(System Settings > Privacy & Security > Microphone)."
        )
    else:
        print("mic OK: receiving real signal")


if __name__ == "__main__":
    main()
