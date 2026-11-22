"""What stride says, as fragment tuples.

Each phrase is a tuple of fragments ("Resting", "1 minute", "30"). The TTS layer plays the joined
phrase as one pre-rendered clip when it has one, and otherwise stitches the fragment clips. Keeping
the split here, in pure code, means TTS never has to parse text.
"""

from __future__ import annotations

from stride.workout.plan import EVEN_PACE_S, Summary

Phrase = tuple[str, ...]

DIDNT_CATCH: Phrase = ("Didn't catch that",)
NO_TIMER: Phrase = ("No timer running",)
STOPPED: Phrase = ("Stopped",)
GO: Phrase = ("Go",)
READY: Phrase = ("Ready",)
SAY_GO: Phrase = ("Say go to start",)
NO_WORKOUT: Phrase = ("No workout running",)
NO_SPLITS: Phrase = ("No splits yet",)
WORKOUT_ENDED: Phrase = ("Workout ended",)
WORKOUT_DONE = "Workout done"


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


# --- M2: workouts ---------------------------------------------------------------------------


def time(seconds: float) -> Phrase:
    """A split, to the tenth: 48.2 -> ("48", "point 2"); 65.3 -> ("1", "oh 5", "point 3")."""
    tenths = int(seconds * 10 + 0.5)
    whole, d = divmod(tenths, 10)
    if whole < 60:
        return (str(whole), f"point {d}")
    m, sec = divmod(whole, 60)
    return (str(m), f"oh {sec}" if sec < 10 else str(sec), f"point {d}")


def rep(k: int) -> str:
    return f"Rep {k}"


def rep_of(k: int, total: int | None) -> Phrase:
    return (rep(k), f"of {total}") if total else (rep(k),)


def distance(meters: int) -> str:
    if meters % 1609 == 0:
        miles = meters // 1609
        return "1 mile" if miles == 1 else f"{miles} miles"
    return str(meters)


def plan_loaded(reps: int, meters: int, rest_s: int) -> Phrase:
    return (f"{reps} by {distance(meters)}", *duration(rest_s), "rest", "Say go when ready")


def go_rep(k: int, total: int | None) -> Phrase:
    """Go cue at the end of a rest. Rep 1 gets a plain "Go"; later reps hear their number."""
    return GO if k == 1 else (*GO, *rep_of(k, total))


def split_then_rest(k: int, split: float, rest_s: int) -> Phrase:
    return (rep(k), *time(split), *resting(rest_s))


def summary(sm: Summary, done: bool = False) -> Phrase:
    head: Phrase = (WORKOUT_DONE,) if done else ()
    if len(sm.splits) == 1:
        return (*head, rep(1), *time(sm.splits[0]))
    out = [*head, "Average", *time(sm.average)]
    out += ["Fastest", rep(sm.fastest[0]), *time(sm.fastest[1])]
    out += ["Slowest", rep(sm.slowest[0]), *time(sm.slowest[1])]
    if sm.fade is not None:
        if abs(sm.fade) < EVEN_PACE_S:
            out.append("Even pace")
        elif sm.fade > 0:
            out += ["Faded", *time(sm.fade)]
        else:
            out += ["Finished faster by", *time(-sm.fade)]
    return tuple(out)
