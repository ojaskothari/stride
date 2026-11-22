from __future__ import annotations

import numpy as np
import pytest

from stride.audio.vad import (
    FRAME,
    FRAME_S,
    Audio,
    EndReason,
    MicPermissionError,
    NoiseFloor,
    Segmenter,
    Utterance,
    check_signal,
    frame_db,
)

RNG = np.random.default_rng(0)


def noise(db: float) -> Audio:
    """One 20 ms frame of white noise at ``db`` dBFS (RMS)."""
    x = RNG.standard_normal(FRAME)
    out: Audio = (x / np.sqrt(np.mean(x**2)) * 10 ** (db / 20)).astype(np.float32)
    return out


def feed(
    seg: Segmenter, levels: list[float], t0: float = 0.0
) -> tuple[Utterance | None, int | None]:
    """Push frames at the given levels; return (utterance, index of the frame that ended it)."""
    for i, db in enumerate(levels):
        u = seg.push(noise(db), t0 + (i + 1) * FRAME_S)
        if u is not None:
            return u, i
    return None, None


def frames(seconds: float, db: float) -> list[float]:
    return [db] * round(seconds / FRAME_S)


def test_frame_db() -> None:
    assert frame_db(noise(-30.0)) == pytest.approx(-30.0, abs=0.01)
    assert frame_db(np.zeros(FRAME, dtype=np.float32)) == -120.0


def test_utterance_ends_after_silence() -> None:
    seg = Segmenter(t_listen=0.0, floor_db=-60.0, silence_end_s=0.6)
    u, _ = feed(seg, frames(0.5, -60) + frames(1.0, -30) + frames(2.0, -60))
    assert u is not None
    assert u.reason is EndReason.SILENCE
    assert u.t_speech_start == pytest.approx(0.5)
    assert u.t_speech_end == pytest.approx(1.5)
    assert u.hangover_s == pytest.approx(0.6)
    # 0.3 s lead + 1.0 s speech + 0.2 s tail
    assert len(u.audio) == round(1.5 / FRAME_S) * FRAME
    assert u.peak_db == pytest.approx(-30.0, abs=0.5)


def test_short_pauses_inside_speech_do_not_end_it() -> None:
    seg = Segmenter(t_listen=0.0, floor_db=-60.0, silence_end_s=0.6)
    levels = frames(0.4, -30) + frames(0.4, -60) + frames(0.4, -30) + frames(1.0, -60)
    u, _ = feed(seg, levels)
    assert u is not None
    assert u.t_speech_end == pytest.approx(1.2)


def test_click_is_not_speech() -> None:
    seg = Segmenter(t_listen=0.0, floor_db=-60.0, start_timeout_s=1.0)
    u, _ = feed(seg, [-60, -20, -20, -60, *frames(2.0, -60)])  # 40 ms click
    assert u is not None
    assert u.reason is EndReason.NO_SPEECH
    assert len(u.audio) == 0
    assert u.hangover_s is None


def test_gate_is_relative_to_noise_floor() -> None:
    # -45 dBFS is speech in a quiet room but just noise next to a loud crowd.
    quiet = Segmenter(t_listen=0.0, floor_db=-60.0, start_timeout_s=1.0)
    u, _ = feed(quiet, frames(0.5, -45) + frames(1.5, -60))
    assert u is not None
    assert u.reason is EndReason.SILENCE
    loud = Segmenter(t_listen=0.0, floor_db=-50.0, start_timeout_s=1.0)
    u, _ = feed(loud, frames(2.0, -45))
    assert u is not None
    assert u.reason is EndReason.NO_SPEECH


def test_max_length_caps_endless_speech() -> None:
    seg = Segmenter(t_listen=0.0, floor_db=-60.0, max_utterance_s=2.0)
    u, i = feed(seg, frames(10.0, -30))
    assert u is not None
    assert u.reason is EndReason.MAX_LENGTH
    assert i is not None
    assert (i + 1) * FRAME_S == pytest.approx(2.0)


def test_noise_floor_tracks_low_percentile() -> None:
    nf = NoiseFloor(window_s=1.0)
    assert not nf.ready
    for k in range(50):
        nf.update(-30.0 if k % 5 == 0 else -55.0)  # occasional loud frames don't move it
    assert nf.ready
    assert nf.db == pytest.approx(-55.0)
    for _ in range(50):
        nf.update(-40.0)  # the environment got louder; the old window rolled off
    assert nf.db == pytest.approx(-40.0)


def test_all_zero_mic_means_permission_denied() -> None:
    zero = np.zeros(FRAME, dtype=np.float32)
    with pytest.raises(MicPermissionError, match="Privacy"):
        check_signal([zero] * 50)
    # Startup frames are zero even with permission; a real signal afterwards is fine.
    check_signal([zero] * 10 + [noise(-60.0)] * 40)
    with pytest.raises(MicPermissionError):
        check_signal([noise(-60.0)] * 10 + [zero] * 40)
