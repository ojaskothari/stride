"""Which phrases get a whole pre-rendered clip, which are stitched from fragments.

Whole clips sound natural; fragments ("Resting" + "4 minutes" + "17") cover everything else. The
catalog is what a normal session says: rests on 15 s steps to 10:00, common adds, status values,
cues, fixed replies. Rendering it takes ~30 s once, then it's cached on disk.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from stride.workout import phrases as ph
from stride.workout.phrases import Phrase

COMMON_ADDS = (15, 30, 45, 60, 90, 120)
MAX_WHOLE_REST_S = 600
MAX_WHOLE_STATUS_S = 300


def whole_phrases(marks: Iterable[int]) -> set[Phrase]:
    out: set[Phrase] = {ph.DIDNT_CATCH, ph.NO_TIMER, ph.STOPPED, ph.GO}
    out |= {ph.cue(m) for m in marks}
    out |= {ph.resting(s) for s in range(15, MAX_WHOLE_REST_S + 1, 15)}
    out |= {ph.added(s) for s in COMMON_ADDS}
    # Every value a status reply can take up to 5:00 (see phrases.rounded_remaining).
    status = [*range(1, 31), *range(35, MAX_WHOLE_STATUS_S + 1, 5)]
    out |= {(*ph.duration(s), "left") for s in status}
    return out


def fragments() -> set[str]:
    """Every fragment a phrase can be built from, so a stitched phrase never waits on a render."""
    out = {"Resting", "Added", "left"}
    for s in range(1, 60):
        out |= {str(s), *ph.duration(s)}
    for m in range(1, 61):
        out |= set(ph.duration(m * 60))
    return out


def resolve(parts: Phrase, have: Callable[[str], bool]) -> list[str]:
    """Clip texts to play for ``parts``: the whole phrase if cached, else its fragments."""
    whole = " ".join(parts)
    return [whole] if len(parts) == 1 or have(whole) else list(parts)


def texts(phrases: Iterable[Phrase]) -> set[str]:
    return {" ".join(p) for p in phrases}
