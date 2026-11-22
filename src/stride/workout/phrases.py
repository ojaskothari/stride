"""What stride says, as fragment tuples.

Each phrase is a tuple of fragments ("Resting", "1 minute", "30"). The TTS layer plays the joined
phrase as one pre-rendered clip when it has one, and otherwise stitches the fragment clips. Keeping
the split here, in pure code, means TTS never has to parse text.
"""

from __future__ import annotations

Phrase = tuple[str, ...]

DIDNT_CATCH: Phrase = ("Didn't catch that",)
NO_TIMER: Phrase = ("No timer running",)
STOPPED: Phrase = ("Stopped",)
GO: Phrase = ("Go",)


def duration(seconds: int) -> Phrase:
    """90 -> ("1 minute", "30"); 120 -> ("2 minutes",); 45 -> ("45 seconds",)."""
    if seconds < 60:
        return (f"{seconds} second" if seconds == 1 else f"{seconds} seconds",)
    m, s = divmod(seconds, 60)
    mins = "1 minute" if m == 1 else f"{m} minutes"
    return (mins, str(s)) if s else (mins,)


def rounded_remaining(seconds: float) -> int:
    """Remaining time as spoken in a status reply: exact up to 30 s, else nearest 5 s.

    Nobody mid-rest needs "1 minute 23"; coarser values also keep the clip set small.
    """
    whole = max(1, int(-(-seconds // 1)))  # ceil: never say 0 while the timer is still running
    return whole if whole <= 30 else 5 * round(whole / 5)


def resting(seconds: int) -> Phrase:
    return ("Resting", *duration(seconds))


def added(seconds: int) -> Phrase:
    return ("Added", *duration(seconds))


def remaining(seconds: float) -> Phrase:
    return (*duration(rounded_remaining(seconds)), "left")


def cue(mark: int) -> Phrase:
    """Cue for ``mark`` seconds remaining: "1 minute left", "30 seconds", "10", "5", "Go"."""
    if mark == 0:
        return GO
    if mark >= 60:
        return (*duration(mark), "left")
    if mark > 10:
        return duration(mark)
    return (str(mark),)
