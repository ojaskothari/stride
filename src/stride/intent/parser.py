"""Rule-based intent parser: transcript -> :class:`Intent`.

Deterministic by design. The command set is small and closed, so rules are predictable, fully
testable and cost nothing in latency. The parser handles *formatting* variation (digits vs. words,
``1:30``, ``3-minute``, ``90s``, punctuation, fillers). It deliberately does **not** guess at
mishearings ("rest nine tea"): fuzzy matching is a separately measured M4 baseline, and folding it
in here would hide how much raw Whisper gets wrong.

Pipeline: normalize text -> tokens -> merge number words -> find a duration -> classify by keywords.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

MAX_SECONDS = 60 * 60  # anything longer is almost certainly a misparse, not a rest


class IntentKind(StrEnum):
    START_REST = "start_rest"
    ADD_TIME = "add_time"
    STATUS = "status"
    STOP = "stop"
    # M2: workouts
    PLAN = "plan"  # "8 by 300, 90 rest"
    GO = "go"  # start the workout / the next rep
    DONE = "done"  # rep finished
    SUMMARY = "summary"
    REP_STATUS = "rep_status"  # "what rep am I on"
    END_WORKOUT = "end_workout"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Intent:
    kind: IntentKind
    # Duration slot. ``None`` means "not stated": the engine applies the plan or a default.
    seconds: int | None = None
    reps: int | None = None  # PLAN only
    distance_m: int | None = None  # PLAN only
    text: str = field(default="", compare=False)


# --- vocabulary -------------------------------------------------------------------------------

_BREAK = "|"
_MIN = "<min>"
_SEC = "<sec>"
_UNITS = {
    "minute": _MIN, "minutes": _MIN, "min": _MIN, "mins": _MIN, "m": _MIN,
    "second": _SEC, "seconds": _SEC, "sec": _SEC, "secs": _SEC, "s": _SEC,
}  # fmt: skip

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}  # fmt: skip

_FILLERS = {"oh", "um", "uh", "uhh", "umm", "hmm", "okay", "ok", "hey", "coach", "please", "like"}

_STOP = {"stop", "skip", "cancel", "end", "ready", "enough", "quit"}
_DONE = {"done", "finish", "finished", "complete", "completed"}
_WORKOUT = {"workout", "session", "practice"}
_SUMMARY = {"summary", "recap", "splits", "results"}
_BY = ("by", "x", "times")
_DIST_UNITS = {
    "meters": 1, "meter": 1, "metres": 1, "metre": 1,
    "km": 1000, "k": 1000, "kilometer": 1000, "kilometers": 1000,
    "mile": 1609, "miles": 1609,
}  # fmt: skip
MAX_REPS = 50
_START = {"start", "begin", "set", "restart", "reset", "starting", "new"}
_ADD = {"add", "plus", "extend", "extra", "another", "more"}
_REST = {"rest", "resting", "recovery", "recover", "break", "timer"}
# A bare number followed by one of these counts something else ("one more rep", "8 by 300").
_NOT_TIME = {"rep", "reps", "more", "by", "x", "times", "meter", "meters", "set", "sets"}

Token = int | float | str


# --- normalization ----------------------------------------------------------------------------


def _mmss(m: re.Match[str]) -> str:
    return f" {int(m.group(1))} minute {int(m.group(2))} second "


def _normalize(text: str) -> list[str]:
    t = text.lower()
    t = t.replace(chr(0x2019), "'")  # curly apostrophe
    # "1:30" and "1.30" (two digits after the dot, < 60) are minutes:seconds.
    t = re.sub(r"\b(\d{1,2}):([0-5]\d)\b", _mmss, t)
    t = re.sub(r"\b(\d{1,2})\.([0-5]\d)\b", _mmss, t)
    # Plans: "8x300" / "8 x 300" -> "8 by 300"; a unit right after "by N" is a distance, so
    # "by 300m" means meters even though "2m" alone means minutes.
    t = re.sub(r"\b(\d+)\s*[x\u00d7]\s*(\d)", r"\1 by \2", t)
    t = re.sub(r"\b(by|x)\s+(\d+(?:\.\d+)?)\s*(m|meters?|metres?)\b", r"\1 \2 meters", t)
    t = re.sub(r"\b(by|x)\s+(\d+(?:\.\d+)?)\s*(k|km)\b", r"\1 \2 km", t)
    # "90s" / "2m" / "3min" glued to the number.
    t = re.sub(r"\b(\d+(?:\.\d+)?)(s|sec|secs|m|min|mins)\b", r"\1 \2", t)
    t = t.replace("'", "")  # what's -> whats, i'm -> im
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)  # keep decimal points only
    t = re.sub(r"[,;]", f" {_BREAK} ", t)  # a pause: stops "three hundred, ninety" merging
    t = re.sub(r"[^a-z0-9.| ]+", " ", t)  # hyphens, ?! etc.
    return [w for w in t.split() if w not in _FILLERS]


def _to_number(word: str) -> int | float | None:
    if re.fullmatch(r"\d+", word):
        return int(word)
    if re.fullmatch(r"\d+\.\d+", word):
        return float(word)
    return None


def _merge_numbers(words: list[str]) -> list[Token]:
    """Turn number words into numbers. "twenty five" -> 25, "one hundred twenty" -> 120.

    Ones followed by tens are *not* merged: "one thirty" stays [1, 30] (minutes:seconds).
    "a"/"an"/"couple" become numbers only directly before a unit ("a minute", "a couple minutes").
    """
    out: list[Token] = []
    i = 0
    n = len(words)
    while i < n:
        w = words[i]
        nxt = words[i + 1] if i + 1 < n else ""
        if w in ("a", "an") and _UNITS.get(nxt) == _MIN:
            out.append(1)
        elif w == "couple":
            if out and out[-1] == "a":
                out.pop()
            if nxt == "of":
                i += 1
            out.append(2)
        elif w in _UNITS:
            out.append(_UNITS[w])
        elif (num := _to_number(w)) is not None:
            out.append(num)
        elif w in _ONES or w in _TENS or w == "hundred":
            value, i = _read_number_words(words, i)
            out.append(value)
            continue
        elif w != _BREAK:
            out.append(w)
        i += 1
    return out


def _read_number_words(words: list[str], i: int) -> tuple[int, int]:
    """Read one spoken number starting at ``words[i]``. Returns (value, next index).

    Ones never merge forward except into "hundred", so "one thirty" stays two numbers.
    """
    n = len(words)

    def peek(k: int) -> str:
        return words[k] if k < n else ""

    value = 0
    if words[i] in _ONES:
        if peek(i + 1) != "hundred":
            return _ONES[words[i]], i + 1
        value, i = _ONES[words[i]] * 100, i + 2
    elif words[i] == "hundred":
        value, i = 100, i + 1
    if value:
        if peek(i) == "and" and (peek(i + 1) in _ONES or peek(i + 1) in _TENS):
            i += 1
        if peek(i) in _ONES:
            return value + _ONES[peek(i)], i + 1
    if peek(i) in _TENS:
        value, i = value + _TENS[peek(i)], i + 1
        if peek(i) in _ONES and 0 < _ONES[peek(i)] < 10:
            value, i = value + _ONES[peek(i)], i + 1
    return value, i


# --- durations --------------------------------------------------------------------------------


def _is_num(t: Token) -> bool:
    return not isinstance(t, str)


def _canonicalize(toks: list[Token]) -> list[Token]:
    """Rewrite spoken shapes into ``number unit`` form.

    - "two more minutes" -> 2 <min> more
    - a minute-unit with no number before it means one: "another minute", "minute thirty"
    """
    out: list[Token] = []
    for t in toks:
        if t in (_MIN, _SEC) and len(out) >= 2 and out[-1] == "more" and _is_num(out[-2]):
            out.insert(len(out) - 1, t)
            continue
        if t == _MIN and not (out and (_is_num(out[-1]) or out[-1] == "half")):
            out.append(1)
        out.append(t)
    return out


def _find_duration(toks: list[Token]) -> tuple[int, int, int] | None:
    """First duration in ``toks``: (seconds, start, end). Bare numbers <= 10 are minutes."""
    # "half a minute" / "half minute" (but not the tail of "two and a half minutes")
    for i, t in enumerate(toks):
        if t == "half" and not (i >= 2 and toks[i - 2] == "and" and toks[i - 1] in ("a", 1)):
            j = i + 1
            if j < len(toks) and toks[j] in ("a", 1):
                j += 1
            if j < len(toks) and toks[j] == _MIN:
                return 30, i, j + 1

    def at(k: int) -> Token | None:
        return toks[k] if k < len(toks) else None

    def and_a_half(k: int) -> int | None:
        """Index after "and a half" starting at k, else None."""
        if at(k) == "and" and at(k + 1) in ("a", 1) and at(k + 2) == "half":
            return k + 3
        return None

    for i, t in enumerate(toks):
        if isinstance(t, str):
            continue
        # N and a half minutes
        if (h := and_a_half(i + 1)) is not None and at(h) == _MIN:
            return round(t * 60) + 30, i, h + 1
        if at(i + 1) == _MIN:
            base = round(t * 60)
            j = i + 2
            # N minutes and a half
            if (k := and_a_half(j)) is not None:
                return base + 30, i, k
            # N minutes [and] M [seconds]
            if at(j) == "and" and isinstance(at(j + 1), int):
                j += 1
            m = at(j)
            if isinstance(m, int) and m < 60:
                end = j + 2 if at(j + 1) == _SEC else j + 1
                return base + m, i, end
            return base, i, i + 2
        if at(i + 1) == _SEC:
            return round(t), i, i + 2
        # Bare "one thirty" -> 1:30
        m = at(i + 1)
        if (
            isinstance(t, int)
            and t <= 10
            and isinstance(m, int)
            and 10 <= m < 60
            and at(i + 2) not in (_MIN, _SEC)
        ):
            return t * 60 + m, i, i + 2
        # Bare number: small numbers are minutes ("rest 3"), larger are seconds ("rest 90").
        if at(i + 1) in _NOT_TIME:
            continue
        if isinstance(t, float) or t <= 10:
            return round(t * 60), i, i + 1
        return t, i, i + 1
    return None


# --- classification ---------------------------------------------------------------------------


def _is_status(words: set[str], seq: list[str]) -> bool:
    joined = " ".join(seq)
    if any(p in joined for p in ("how much", "how long", "time left", "whats left", "time check")):
        return True
    return bool(words & {"remaining", "status"}) or ("left" in words and "how" in words)


def _find_plan(toks: list[Token]) -> tuple[int, int, int, int] | None:
    """ "8 by 300 [meters]" -> (reps, distance_m, start, end). Distance needs a unit or >= 50."""
    for k, t in enumerate(toks):
        if t not in _BY or k == 0 or k + 1 >= len(toks):
            continue
        reps, dist = toks[k - 1], toks[k + 1]
        if not isinstance(reps, int) or isinstance(reps, bool) or isinstance(dist, str):
            continue
        unit = toks[k + 2] if k + 2 < len(toks) else None
        factor = _DIST_UNITS.get(unit, 0) if isinstance(unit, str) else 0
        end = k + 3 if factor else k + 2
        meters = round(dist * (factor or 1))
        if 1 <= reps <= MAX_REPS and (factor or meters >= 50) and 50 <= meters <= 50_000:
            return reps, meters, k - 1, end
    return None


def _strings(toks: list[Token], skip: tuple[int, int] = (0, 0)) -> list[str]:
    return [
        t
        for k, t in enumerate(toks)
        if isinstance(t, str) and t not in (_MIN, _SEC) and not skip[0] <= k < skip[1]
    ]


def parse(text: str) -> Intent:
    """Parse one transcript. Never raises; unrecognized input is ``UNKNOWN``."""
    words = _normalize(text)
    toks = _canonicalize(_merge_numbers(words))

    def make(kind: IntentKind, secs: int | None = None, **slots: int) -> Intent:
        if secs is not None and not 0 < secs <= MAX_SECONDS:
            return Intent(IntentKind.UNKNOWN, None, text=text)
        return Intent(kind, secs, text=text, **slots)

    if not words:
        return make(IntentKind.UNKNOWN)

    # A plan first: its numbers ("8 by 300") must not be read as durations.
    if plan := _find_plan(toks):
        reps, meters, a, b = plan
        rest = toks[:a] + toks[b:]
        found = _find_duration(rest)
        rest_s = found[0] if found else None
        if rest_s is None and meters % 50 and meters > 100 and set(_strings(rest)) & _REST:
            # Whisper drops the pause: "three hundred, ninety rest" -> "390 rest". Rep distances
            # are multiples of 50 m, so the last two digits are the rest ("3 rest" = 3 minutes).
            meters, tail = meters - meters % 100, meters % 100
            rest_s = tail * 60 if tail <= 10 else tail
        return make(IntentKind.PLAN, rest_s, reps=reps, distance_m=meters)

    found = _find_duration(toks)
    seconds = found[0] if found else None
    rest_words = _strings(toks, (found[1], found[2]) if found else (0, 0))
    wset = set(rest_words)
    joined = " ".join(rest_words)

    if wset & _WORKOUT and wset & (_STOP | _DONE | {"over"}):
        return make(IntentKind.END_WORKOUT)
    if wset & _SUMMARY or "how did i do" in joined:
        return make(IntentKind.SUMMARY)
    if wset & {"rep", "reps"} and wset & {"what", "which", "how", "count", "number", "many"}:
        return make(IntentKind.REP_STATUS)
    if wset & _DONE:
        return make(IntentKind.DONE)
    if "go" in wset or (wset & _WORKOUT and wset & _START):
        return make(IntentKind.GO)
    if wset & _STOP:
        return make(IntentKind.STOP)
    if _is_status(wset, rest_words):
        return make(IntentKind.STATUS)
    if wset & _START:
        return make(IntentKind.START_REST, seconds)
    # "another" / "more" alone ("one more rep") are not an add without a duration or "time".
    if wset & _ADD and (seconds is not None or wset & {"add", "extend", "plus", "extra", "time"}):
        return make(IntentKind.ADD_TIME, seconds)
    if wset & _REST or (seconds is not None and len(wset - {"for", "my", "a", "of"}) == 0):
        return make(IntentKind.START_REST, seconds)
    return make(IntentKind.UNKNOWN)
