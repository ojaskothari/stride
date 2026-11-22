"""Workout plans and summaries. Pure."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

# A fade smaller than this is noise in hand timing (tap reaction ~0.2 s each end), not a trend.
EVEN_PACE_S = 0.5


@dataclass(frozen=True)
class Plan:
    reps: int
    distance_m: int
    rest_s: int


@dataclass(frozen=True)
class Summary:
    splits: tuple[float, ...]
    average: float
    fastest: tuple[int, float]  # (1-based rep, seconds)
    slowest: tuple[int, float]
    # Last third's mean minus first third's mean: positive = slowed down. None with < 2 reps.
    fade: float | None


def summarize(splits: list[float] | tuple[float, ...]) -> Summary | None:
    if not splits:
        return None
    s = tuple(splits)
    fast = min(range(len(s)), key=lambda i: s[i])
    slow = max(range(len(s)), key=lambda i: s[i])
    fade = None
    if len(s) >= 2:
        k = max(1, len(s) // 3)
        fade = fmean(s[-k:]) - fmean(s[:k])
    return Summary(s, fmean(s), (fast + 1, s[fast]), (slow + 1, s[slow]), fade)
