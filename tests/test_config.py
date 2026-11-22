from __future__ import annotations

from pathlib import Path

import pytest

from stride.config import Config, ConfigError, load_config, parse_config

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.toml"


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_repo_config_matches_builtin_defaults() -> None:
    # config.toml documents the defaults; this keeps the two from drifting apart.
    assert load_config(REPO_CONFIG) == Config()


def test_no_path_returns_defaults() -> None:
    cfg = load_config(None)
    assert cfg.rest.default_seconds == 90
    assert cfg.cues.marks[-5:] == (5, 4, 3, 2, 1)
    assert cfg.cues.min_rest == {120: 180}


def test_partial_override_keeps_other_defaults(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, "[rest]\ndefault_seconds = 180\n"))
    assert cfg.rest.default_seconds == 180
    assert cfg.stt == Config().stt


def test_int_accepted_for_float_field() -> None:
    cfg = parse_config({"audio": {"vad_db_above_floor": 12}})
    assert cfg.audio.vad_db_above_floor == 12.0
    assert isinstance(cfg.audio.vad_db_above_floor, float)


def test_custom_cue_schedule() -> None:
    cfg = parse_config({"cues": {"marks": [60, 10, 3, 2, 1], "min_rest": {}}})
    assert cfg.cues.marks == (60, 10, 3, 2, 1)
    assert cfg.cues.min_rest == {}


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"rset": {}}, "unknown section"),
        ({"rest": {"defualt_seconds": 90}}, "unknown key rest.defualt_seconds"),
        ({"rest": 90}, "must be a table"),
        ({"rest": {"default_seconds": "90"}}, "must be an integer"),
        ({"rest": {"default_seconds": True}}, "must be an integer"),
        ({"rest": {"default_seconds": 0}}, "must be positive"),
        ({"stt": {"vocab_prompt": 1}}, "true or false"),
        ({"cues": {"marks": [10, 30]}}, "descending"),
        ({"cues": {"marks": [10, 10, 1]}}, "unique"),
        ({"cues": {"marks": [10, 0]}}, "positive"),
        ({"cues": {"marks": [10.5]}}, "list of integers"),
        ({"cues": {"marks": [60, 30], "min_rest": {"120": 180}}}, "not in cues.marks"),
        ({"cues": {"min_rest": {"two": 180}}}, "whole seconds"),
        ({"enhance": {"backend": "krisp"}}, "enhance.backend"),
        ({"tts": {"backend": "elevenlabs"}}, "tts.backend"),
        ({"stt": {"model": ""}}, "stt.model"),
        ({"audio": {"sample_rate": 44100}}, "16000"),
        ({"audio": {"silence_end_ms": -1}}, "positive"),
    ],
)
def test_invalid_config_rejected(raw: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(raw)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_invalid_toml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(write(tmp_path, "[rest\n"))
