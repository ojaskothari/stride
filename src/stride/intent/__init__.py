"""Deterministic intent parser: transcript -> intent + slots (seconds, reps, distance)."""

from stride.intent.parser import Intent, IntentKind, parse

__all__ = ["Intent", "IntentKind", "parse"]
