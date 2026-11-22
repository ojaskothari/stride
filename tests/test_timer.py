from __future__ import annotations

import pytest

from stride.config import Config
from stride.intent import Intent, IntentKind
from stride.workout import Coach, RestTimer
from stride.workout import phrases as ph

MARKS = (120, 60, 30, 10, 5, 4, 3, 2, 1)
MIN_REST = {120: 180}
STEP = 0.05


def run(timer: RestTimer, start: float, until: float) -> list[tuple[float, int]]:
    """Poll the timer like the live loop does; return (time, mark) for each cue."""
    out: list[tuple[float, int]] = []
    n = round((until - start) / STEP)
    for i in range(n + 1):
        now = start + i * STEP
        out += [(round(now, 2), c.mark) for c in timer.tick(now)]
    return out


def marks(events: list[tuple[float, int]]) -> list[int]:
    return [m for _, m in events]


def test_default_rest_cue_schedule() -> None:
    t = RestTimer(MARKS, MIN_REST)
    t.start(0.0, 90)
    ev = run(t, 0.0, 100)
    assert ev == [
        (30, 60),
        (60, 30),
        (80, 10),
        (85, 5),
        (86, 4),
        (87, 3),
        (88, 2),
        (89, 1),
        (90, 0),
    ]
    assert not t.running


def test_two_minute_mark_only_for_rests_of_three_minutes_or_more() -> None:
    t = RestTimer(MARKS, MIN_REST)
    t.start(0.0, 150)
    assert 120 not in marks(run(t, 0.0, 160))
    t.start(0.0, 180)
    assert marks(run(t, 0.0, 190))[:2] == [120, 60]


def test_marks_too_close_to_start_are_skipped() -> None:
    t = RestTimer(MARKS, MIN_REST)
    t.start(0.0, 62)  # "1 minute left" would land 2 s after the confirmation
    assert marks(run(t, 0.0, 70)) == [30, 10, 5, 4, 3, 2, 1, 0]
    t.start(0.0, 63)  # exactly MIN_GAP: kept
    assert marks(run(t, 0.0, 70))[0] == 60


def test_short_rest_still_ends_with_go() -> None:
    t = RestTimer(MARKS, MIN_REST)
    t.start(0.0, 2)
    assert run(t, 0.0, 5) == [(2, 0)]


def test_add_rearms_marks_pushed_back_into_the_future() -> None:
    t = RestTimer(MARKS, MIN_REST)
    t.start(0.0, 90)
    assert marks(run(t, 0.0, 65)) == [60, 30]  # 25 s left
    t.extend(65.0, 60)  # 85 s left, 150 s total
    ev = run(t, 65.05, 160)
    assert marks(ev) == [60, 30, 10, 5, 4, 3, 2, 1, 0]
    assert ev[0] == (90, 60)
    assert ev[-1] == (150, 0)


def test_add_can_unlock_the_two_minute_mark() -> None:
    t = RestTimer(MARKS, MIN_REST)
    t.start(0.0, 150)
    t.extend(10.0, 60)  # total 210 >= 180, 200 s left
    assert marks(run(t, 10.0, 220))[0] == 120


def test_add_does_not_repeat_a_mark_just_spoken() -> None:
    t = RestTimer(MARKS, MIN_REST)
    t.start(0.0, 90)
    run(t, 0.0, 61)  # "30 seconds" spoken at t=60
    t.extend(61.0, 30)  # 59 s left: the 30 s mark is armed again, 60 is not (in the past)
    assert marks(run(t, 61.0, 125))[0] == 30


def test_stall_speaks_only_latest_overdue_cue() -> None:
    t = RestTimer(MARKS, MIN_REST)
    t.start(0.0, 90)
    run(t, 0.0, 84)
    assert [c.mark for c in t.tick(87.5)] == [3]  # 5 and 4 were missed: dropped, not stacked
    assert [c.mark for c in t.tick(88.0)] == [2]


def test_stall_past_end_still_says_go() -> None:
    t = RestTimer(MARKS, MIN_REST)
    t.start(0.0, 90)
    assert [c.mark for c in t.tick(200.0)] == [0]
    assert not t.running
    assert t.tick(201.0) == []


def test_lead_releases_cues_early_but_reports_true_due_time() -> None:
    t = RestTimer(MARKS, MIN_REST, lead_s=0.2)
    t.start(0.0, 90)
    ev = run(t, 0.0, 100)
    assert ev[0] == (29.8, 60)
    t.start(0.0, 90)
    c = t.tick(29.8)
    assert c[0].due == 30.0


def test_stop_and_remaining() -> None:
    t = RestTimer(MARKS, MIN_REST)
    assert t.remaining(0.0) is None
    t.start(10.0, 90)
    assert t.remaining(40.0) == 60.0
    t.stop()
    assert not t.running
    assert t.tick(1000.0) == []


def test_invalid_durations() -> None:
    t = RestTimer(MARKS, MIN_REST)
    with pytest.raises(ValueError, match="positive"):
        t.start(0.0, 0)
    with pytest.raises(RuntimeError, match="no timer"):
        t.extend(0.0, 30)


# --- phrases ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("secs", "parts"),
    [
        (1, ("1 second",)),
        (45, ("45 seconds",)),
        (60, ("1 minute",)),
        (90, ("1 minute", "30")),
        (120, ("2 minutes",)),
        (135, ("2 minutes", "15")),
    ],
)
def test_duration_phrase(secs: int, parts: tuple[str, ...]) -> None:
    assert ph.duration(secs) == parts


@pytest.mark.parametrize(
    ("left", "spoken"), [(0.2, 1), (12.3, 13), (30.0, 30), (83.4, 85), (61.0, 60), (32.0, 30)]
)
def test_status_rounding(left: float, spoken: int) -> None:
    assert ph.rounded_remaining(left) == spoken


def test_cue_phrases() -> None:
    assert ph.cue(120) == ("2 minutes", "left")
    assert ph.cue(60) == ("1 minute", "left")
    assert ph.cue(30) == ("30 seconds",)
    assert ph.cue(10) == ("10",)
    assert ph.cue(3) == ("3",)
    assert ph.cue(0) == ("Go",)


# --- coach ------------------------------------------------------------------------------------


def i(kind: IntentKind, secs: int | None = None) -> Intent:
    return Intent(kind, secs)


def test_coach_conversation() -> None:
    c = Coach(Config())
    assert c.handle(i(IntentKind.START_REST), 0.0).text == "Resting 1 minute 30"
    assert c.handle(i(IntentKind.STATUS), 10.0).text == "1 minute 20 left"
    assert c.handle(i(IntentKind.ADD_TIME, 30), 20.0).text == "Added 30 seconds"
    assert c.timer.remaining(20.0) == 100.0
    assert c.handle(i(IntentKind.ADD_TIME), 20.0).text == "Added 30 seconds"  # default add
    assert c.handle(i(IntentKind.STOP), 30.0).text == "Stopped"
    assert c.handle(i(IntentKind.STOP), 31.0).text == "No timer running"
    assert c.handle(i(IntentKind.STATUS), 31.0).text == "No timer running"
    assert c.handle(i(IntentKind.UNKNOWN), 31.0).text == "Didn't catch that"


def test_coach_explicit_duration_and_restart() -> None:
    c = Coach(Config())
    assert c.handle(i(IntentKind.START_REST, 180), 0.0).text == "Resting 3 minutes"
    assert c.handle(i(IntentKind.START_REST, 60), 50.0).text == "Resting 1 minute"
    assert c.timer.remaining(50.0) == 60.0


def test_coach_add_after_go_starts_a_new_rest() -> None:
    c = Coach(Config())
    c.handle(i(IntentKind.START_REST, 10), 0.0)
    assert [s.text for s in c.tick(10.0)] == ["Go"]
    assert c.handle(i(IntentKind.ADD_TIME, 60), 12.0).text == "Resting 1 minute"


def test_coach_tick_emits_cue_speech_with_due_time() -> None:
    c = Coach(Config())
    c.handle(i(IntentKind.START_REST, 90), 0.0)
    (s,) = c.tick(30.0)
    assert (s.parts, s.due, s.mark) == (("1 minute", "left"), 30.0, 60)
