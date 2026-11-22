"""Command-line entry point: ``stride <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from stride import __version__
from stride.config import ConfigError, load_config

DEFAULT_CONFIG = Path("config.toml")


def _resolve_config_path(arg: str | None) -> Path | None:
    if arg is not None:
        return Path(arg)
    return DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None


def _cmd_config(args: argparse.Namespace) -> int:
    path = _resolve_config_path(args.config)
    cfg = load_config(path)
    print(f"# source: {path or 'built-in defaults'}")
    print(json.dumps(asdict(cfg), indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stride", description=__doc__)
    parser.add_argument("--version", action="version", version=f"stride {__version__}")
    parser.add_argument("--config", help="path to config.toml (default: ./config.toml if present)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("config", help="print the resolved configuration").set_defaults(func=_cmd_config)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except ConfigError as e:
        print(f"stride: config error: {e}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
