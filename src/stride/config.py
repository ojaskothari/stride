"""Typed, validated configuration loaded from ``config.toml``.

Defaults live here so stride runs without a config file. A TOML file overrides any subset of
them. Unknown keys are rejected: a silently ignored typo is a bug you only discover mid-workout.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

ENHANCERS = ("none", "rnnoise", "deepfilternet")
# Cues are pre-rendered clips played in-process; the backend only renders them.
# "none" stays silent (tests, headless runs). See docs/decisions.md.
TTS_BACKENDS = ("say", "none")


class ConfigError(ValueError):
    """Raised when a config file is malformed or contains invalid values."""


@dataclass(frozen=True)
class RestConfig:
    default_seconds: int = 90
    default_add_seconds: int = 30


@dataclass(frozen=True)
class CueConfig:
    # Seconds remaining at which a cue is spoken.
    marks: tuple[int, ...] = (120, 60, 30, 10, 5, 4, 3, 2, 1)
    # mark -> minimum rest length for that mark to be spoken (2:00 only if rest >= 3:00).
    min_rest: dict[int, int] = field(default_factory=lambda: {120: 180})


@dataclass(frozen=True)
class SttConfig:
    model: str = "base.en"
    compute_type: str = "int8"
    vocab_prompt: bool = True


@dataclass(frozen=True)
class EnhanceConfig:
    backend: str = "none"


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    vad_db_above_floor: float = 10.0
    silence_end_ms: int = 600
    max_utterance_s: float = 6.0


@dataclass(frozen=True)
class TtsConfig:
    backend: str = "say"


@dataclass(frozen=True)
class LogConfig:
    dir: str = "logs"
    save_audio: bool = False


@dataclass(frozen=True)
class Config:
    rest: RestConfig = field(default_factory=RestConfig)
    cues: CueConfig = field(default_factory=CueConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    enhance: EnhanceConfig = field(default_factory=EnhanceConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    log: LogConfig = field(default_factory=LogConfig)


def load_config(path: Path | None = None) -> Config:
    """Load ``path`` over the defaults. ``None`` returns the defaults."""
    if path is None:
        return Config()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path}") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: invalid TOML: {e}") from e
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> Config:
    cfg = Config()
    sections = {f.name for f in fields(Config)}
    for name, values in raw.items():
        if name not in sections:
            raise ConfigError(f"unknown section [{name}]; expected one of {sorted(sections)}")
        if not isinstance(values, dict):
            raise ConfigError(f"[{name}] must be a table")
        section = _override(getattr(cfg, name), values, name)
        cfg = replace(cfg, **{name: section})
    _validate(cfg)
    return cfg


def _override(section: Any, values: dict[str, Any], name: str) -> Any:
    known = {f.name: f for f in fields(section)}
    updates: dict[str, Any] = {}
    for key, value in values.items():
        if key not in known:
            raise ConfigError(f"unknown key {name}.{key}; expected one of {sorted(known)}")
        default = getattr(section, key)
        updates[key] = _coerce(value, default, f"{name}.{key}")
    return replace(section, **updates)


def _coerce(value: Any, default: Any, where: str) -> Any:
    # bool is a subclass of int, so check it first.
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be true or false")
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where} must be a number")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string")
        return value
    if isinstance(default, tuple):
        if not isinstance(value, list) or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in value
        ):
            raise ConfigError(f"{where} must be a list of integers")
        return tuple(value)
    if isinstance(default, dict):
        if not isinstance(value, dict):
            raise ConfigError(f"{where} must be a table")
        out: dict[int, int] = {}
        for k, v in value.items():
            if not k.isdigit() or isinstance(v, bool) or not isinstance(v, int):
                raise ConfigError(f"{where} must map whole seconds to whole seconds")
            out[int(k)] = v
        return out
    raise AssertionError(f"unhandled config type at {where}")  # pragma: no cover


def _validate(cfg: Config) -> None:
    if cfg.rest.default_seconds <= 0 or cfg.rest.default_add_seconds <= 0:
        raise ConfigError("rest.default_seconds and rest.default_add_seconds must be positive")

    marks = cfg.cues.marks
    if any(m <= 0 for m in marks):
        raise ConfigError("cues.marks must all be positive")
    if list(marks) != sorted(set(marks), reverse=True):
        raise ConfigError("cues.marks must be unique and in descending order")
    if unknown := set(cfg.cues.min_rest) - set(marks):
        raise ConfigError(f"cues.min_rest refers to marks not in cues.marks: {sorted(unknown)}")

    if not cfg.stt.model:
        raise ConfigError("stt.model must not be empty")
    if cfg.enhance.backend not in ENHANCERS:
        raise ConfigError(f"enhance.backend must be one of {ENHANCERS}")
    if cfg.tts.backend not in TTS_BACKENDS:
        raise ConfigError(f"tts.backend must be one of {TTS_BACKENDS}")

    a = cfg.audio
    if a.sample_rate != 16_000:
        # Whisper expects 16 kHz; resampling belongs in capture, not in config.
        raise ConfigError("audio.sample_rate must be 16000")
    if a.silence_end_ms <= 0 or a.max_utterance_s <= 0 or a.vad_db_above_floor <= 0:
        raise ConfigError("audio thresholds must be positive")
