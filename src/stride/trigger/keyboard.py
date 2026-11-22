"""Tap-to-talk from the terminal: any key taps, ``q`` quits. Single keypress, no Enter needed."""

from __future__ import annotations

import os
import select
import sys
import termios
import tty
from types import TracebackType
from typing import Any


class Keyboard:
    """Context manager that puts the terminal in cbreak mode and restores it on exit."""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._saved: list[Any] | None = None

    def __enter__(self) -> Keyboard:
        if not os.isatty(self._fd):
            raise RuntimeError("stride run needs an interactive terminal for tap-to-talk")
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)  # keys arrive immediately, unechoed; Ctrl-C still works
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def poll(self) -> str | None:
        """The next key pressed, or None. Never blocks."""
        ready, _, _ = select.select([self._fd], [], [], 0)
        if not ready:
            return None
        return os.read(self._fd, 32).decode(errors="ignore")[:1] or None
