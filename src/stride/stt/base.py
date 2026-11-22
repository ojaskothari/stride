"""Speech-to-text interface. Backends are swappable so the M4 benchmark can compare them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from stride.audio.vad import Audio

# Passed as Whisper's initial prompt: biases decoding toward the command vocabulary and toward
# writing numbers as digits. Whether it helps in noise is measured in M4 (a "cheap baseline").
VOCAB_PROMPT = (
    "start my rest, rest 90, 3 minute rest, add 30 seconds, another minute, time left, stop."
)


@dataclass(frozen=True)
class Transcript:
    text: str
    seconds: float  # compute time
    avg_logprob: float | None = None  # decoder confidence, logged for M4/M6 analysis
    no_speech_prob: float | None = None


class Transcriber(Protocol):
    @property
    def name(self) -> str: ...

    def transcribe(self, audio: Audio) -> Transcript: ...
