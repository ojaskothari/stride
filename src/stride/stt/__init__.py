"""Pluggable speech-to-text. Default backend: faster-whisper, on-device.

``stride.stt.whisper`` needs the ``[audio]`` extra and is imported explicitly.
"""

from stride.stt.base import VOCAB_PROMPT, Transcriber, Transcript

__all__ = ["VOCAB_PROMPT", "Transcriber", "Transcript"]
