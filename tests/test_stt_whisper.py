"""End-to-end check on real hardware: `say` -> Whisper -> parser. Skipped in CI (no audio extra)."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from stride.intent import IntentKind, parse

pytest.importorskip("faster_whisper")
if shutil.which("say") is None:
    pytest.skip("needs macOS `say`", allow_module_level=True)

from stride.stt.whisper import WhisperTranscriber


@pytest.fixture(scope="module")
def stt() -> WhisperTranscriber:
    return WhisperTranscriber("tiny.en")


def speak(text: str, tmp: Path) -> np.ndarray:
    out = tmp / "c.wav"
    subprocess.run(
        ["say", "-o", str(out), "--file-format=WAVE", "--data-format=LEI16@16000", "--", text],
        check=True,
    )
    with wave.open(str(out)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return (pcm.astype(np.float32) / 32768).astype(np.float32)


@pytest.mark.parametrize(
    ("spoken", "kind", "seconds"),
    [
        ("start my three minute rest", IntentKind.START_REST, 180),
        ("add thirty seconds", IntentKind.ADD_TIME, 30),
        ("how much time left", IntentKind.STATUS, None),
        ("stop", IntentKind.STOP, None),
    ],
)
def test_spoken_command_round_trip(
    stt: WhisperTranscriber, tmp_path: Path, spoken: str, kind: IntentKind, seconds: int | None
) -> None:
    t = stt.transcribe(speak(spoken, tmp_path))
    got = parse(t.text)
    assert (got.kind, got.seconds) == (kind, seconds), t.text
    assert t.seconds < 1.0
