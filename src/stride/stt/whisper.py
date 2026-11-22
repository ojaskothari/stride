"""faster-whisper backend. Needs the ``[audio]`` extra. Model loaded once, on-device, int8 CPU."""

from __future__ import annotations

import os

# Public models only: never send a stored Hugging Face token (an expired one breaks downloads,
# and a valid one shouldn't leak to requests that don't need it). Must precede the import.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import time

import numpy as np
from faster_whisper import WhisperModel

from stride.audio.vad import SAMPLE_RATE, Audio
from stride.stt.base import VOCAB_PROMPT, Transcript


def load_model(model: str, compute_type: str) -> WhisperModel:
    """Use the local copy if there is one (works offline at the track); download only if not."""
    try:
        return WhisperModel(model, device="cpu", compute_type=compute_type, local_files_only=True)
    except Exception:  # not cached yet; faster-whisper raises various hub errors here
        return WhisperModel(model, device="cpu", compute_type=compute_type)


class WhisperTranscriber:
    def __init__(
        self,
        model: str = "base.en",
        compute_type: str = "int8",
        vocab_prompt: bool = True,
        beam_size: int = 5,
    ) -> None:
        self._model_name = model
        self._model = load_model(model, compute_type)
        self._prompt = VOCAB_PROMPT if vocab_prompt else None
        self._beam = beam_size
        # The first call pays one-off setup costs; pay them now, not on the first command.
        self.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))

    @property
    def name(self) -> str:
        return f"faster-whisper/{self._model_name}"

    def transcribe(self, audio: Audio) -> Transcript:
        t0 = time.perf_counter()
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=self._beam,
            initial_prompt=self._prompt,
            without_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,  # stride segments utterances itself
        )
        segs = list(segments)  # a generator: consuming it does the decoding
        text = " ".join(s.text.strip() for s in segs).strip()
        return Transcript(
            text=text,
            seconds=time.perf_counter() - t0,
            avg_logprob=min((s.avg_logprob for s in segs), default=None),
            no_speech_prob=max((s.no_speech_prob for s in segs), default=None),
        )
