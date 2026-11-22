"""Rest timer: pure state machine over an injected clock.

Every method takes ``now`` (seconds on any monotonic clock), so tests drive time explicitly and the
live loop passes ``time.monotonic()``. ``tick(now)`` returns the cues that are due.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# A mark this close to "now" when it is armed is skipped: "Resting 1 minute 2" followed two seconds
# later by "1 minute left" is noise, not information.
MIN_GAP_S = 3.0


@dataclass(frozen=True)
class Cue:
    mark: int  # seconds remaining when spoken; 0 is "Go"
    due: float  # clock time at which the cue should be *heard*


class RestTimer:
    def __init__(
        self,
        marks: Sequence[int],
        min_rest: Mapping[int, int] | None = None,
        *,
        lead_s: float = 0.0,
        min_gap_s: float = MIN_GAP_S,
    ) -> None:
        """``lead_s`` releases cues early by the audio output latency so they are heard on time."""
        self._marks = (*sorted(set(marks), reverse=True), 0)
        self._min_rest = dict(min_rest or {})
        self._lead = lead_s
        self._min_gap = min_gap_s
        self._end: float | None = None
        self._total = 0.0
        self._pending: list[int] = []

    @property
    def running(self) -> bool:
        return self._end is not None

    @property
    def total(self) -> float:
        """Length of the current rest including any additions."""
        return self._total

    def remaining(self, now: float) -> float | None:
        return None if self._end is None else max(0.0, self._end - now)

    def start(self, now: float, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("rest must be positive")
        self._end = now + seconds
        self._total = seconds
        self._arm(now)

    def extend(self, now: float, seconds: float) -> None:
        if self._end is None:
            raise RuntimeError("no timer running")
        if seconds <= 0:
            raise ValueError("extension must be positive")
        self._end += seconds
        self._total += seconds
        self._arm(now)

    def stop(self) -> None:
        self._end = None
        self._total = 0.0
        self._pending = []

    def tick(self, now: float) -> list[Cue]:
        """Cues due at ``now``. After a stall only the latest overdue cue is returned."""
        if self._end is None:
            return []
        end = self._end
        due = [m for m in self._pending if end - m - self._lead <= now]
        if not due:
            return []
        self._pending = [m for m in self._pending if m not in due]
        latest = min(due)
        if latest == 0:
            self.stop()
        return [Cue(latest, end - latest)]

    def _arm(self, now: float) -> None:
        """Rebuild pending marks for the current end. Re-arms marks an "add" pushed back out."""
        assert self._end is not None
        left = self._end - now
        self._pending = [
            m
            for m in self._marks
            if m == 0 or (left - m >= self._min_gap and self._total >= self._min_rest.get(m, 0))
        ]
