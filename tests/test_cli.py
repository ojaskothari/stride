from __future__ import annotations

import json
from pathlib import Path

import pytest

from stride import __version__
from stride.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_config_prints_resolved_values(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "c.toml"
    p.write_text("[rest]\ndefault_seconds = 120\n", encoding="utf-8")
    assert main(["--config", str(p), "config"]) == 0
    out = capsys.readouterr().out
    body = json.loads(out.split("\n", 1)[1])
    assert body["rest"]["default_seconds"] == 120


def test_bad_config_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "c.toml"
    p.write_text("[rest]\ndefualt_seconds = 120\n", encoding="utf-8")
    assert main(["--config", str(p), "config"]) == 2
    assert "defualt_seconds" in capsys.readouterr().err
