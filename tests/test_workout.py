from __future__ import annotations

import pytest

from stride.config import Config
from stride.intent import Intent, IntentKind, parse
from stride.workout import phrases as ph
from stride.workout.coach import Coach, Phase
from stride.workout.plan import summarize

K = IntentKind


def run_until(c: Coach, start: float, end: float, step: float = 0.05) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    n = round((end - start) / step)
    for i in range(n + 1):
        now = round(start + i * step, 3)
        out += [(now, s.text) for s in c.tick(now)]
    return out


def phase(c: Coach) -> Phase:
    """Read through a call so mypy doesn't narrow ``c.phase`` across commands."""
    return c.phase


def say(c: Coach, text: str, now: float, tap: float | None = None) -> str:
    return c.handle(parse(text), now, tap).text


# --- plan --------------------------------------------------------------------------------------


def test_full_planned_workout() -> None:
    c = Coach(Config())
    assert say(c, "3 by 300, 60 rest", 0.0) == "3 by 300 1 minute rest Say go when ready"
    assert phase(c) is Phase.READY
    assert say(c, "go", 10.0) == "Ready"
    cues = run_until(c, 10.0, 14.0)
    assert cues == [(11.0, "3"), (12.0, "2"), (13.0, "1"), (14.0, "Go")]
    assert c.rep_start == 14.0

    # Rep 1: tap at 62.0, transcript handled 1.5 s later; the split ends at the tap.
    assert say(c, "done", 63.5, tap=62.0) == "Rep 1 48 point 0 Resting 1 minute"
    assert phase(c) is Phase.REST
    assert c.timer.remaining(63.5) == pytest.approx(58.5)  # rest timed from the tap

    cues = run_until(c, 63.5, 122.0)
    assert cues[-1] == (122.0, "Go Rep 2 of 3")
    assert c.rep_start == 122.0
    assert say(c, "done", 171.0, tap=169.5) == "Rep 2 47 point 5 Resting 1 minute"
    run_until(c, 171.0, 229.5)
    assert c.rep_start == 229.5

    last = say(c, "done", 280.0, tap=279.0)  # 49.5 s
    assert last.startswith("Rep 3 49 point 5 Workout done Average 48 point 3")
    assert "Fastest Rep 2 47 point 5" in last
    assert "Slowest Rep 3 49 point 5" in last
    assert last.endswith("Faded 1 point 5")
    assert phase(c) is Phase.FINISHED
    assert not c.timer.running

    events = c.drain_events()
    assert [e["type"] for e in events] == ["plan", "split", "split", "split", "summary"]
    assert events[-1]["splits"] == [48.0, 47.5, 49.5]
    assert c.drain_events() == []


def test_plan_without_rest_uses_default() -> None:
    c = Coach(Config())
    say(c, "8 by 200", 0.0)
    assert c.plan is not None
    assert c.plan.rest_s == 90


def test_stop_during_planned_rest_starts_next_rep_now() -> None:
    c = Coach(Config())
    say(c, "4 by 400, 2 minute rest", 0.0)
    say(c, "go", 0.0)
    run_until(c, 0.0, 4.0)
    say(c, "done", 70.0, tap=69.0)
    assert say(c, "stop", 100.0) == "Go Rep 2 of 4"
    assert phase(c) is Phase.REP
    assert c.rep_start == 100.0
    assert not c.timer.running
    assert say(c, "go", 105.0) == "Rep 2 of 4"  # already running: just say where we are


def test_go_during_rest_also_skips_it() -> None:
    c = Coach(Config())
    say(c, "4 by 400, 2 minute rest", 0.0)
    say(c, "go", 0.0)
    run_until(c, 0.0, 4.0)
    say(c, "done", 70.0, tap=69.0)
    assert say(c, "let's go", 90.0) == "Go Rep 2 of 4"


def test_stop_during_countdown_cancels_it() -> None:
    c = Coach(Config())
    say(c, "4 by 400", 0.0)
    say(c, "go", 0.0)
    assert say(c, "stop", 1.5) == "Stopped"
    assert phase(c) is Phase.READY
    assert run_until(c, 1.5, 6.0) == []


def test_done_before_go_asks_for_go() -> None:
    c = Coach(Config())
    say(c, "4 by 400", 0.0)
    assert say(c, "done", 5.0) == "Say go to start"


def test_start_rest_at_end_of_rep_counts_as_done_with_custom_rest() -> None:
    c = Coach(Config())
    say(c, "4 by 400, 90 rest", 0.0)
    say(c, "go", 0.0)
    run_until(c, 0.0, 4.0)
    assert (
        say(c, "start my 3 minute rest", 70.0, tap=68.0) == "Rep 1 1 oh 4 point 0 Resting 3 minutes"
    )


def test_rep_status_and_summary_mid_workout() -> None:
    c = Coach(Config())
    assert say(c, "what rep am I on", 0.0) == "No workout running"
    assert say(c, "summary", 0.0) == "No splits yet"
    say(c, "6 by 200, 60 rest", 0.0)
    assert say(c, "what rep am I on", 1.0) == "Rep 1 of 6"
    say(c, "go", 1.0)
    run_until(c, 1.0, 5.0)
    say(c, "done", 35.0, tap=34.0)
    assert say(c, "what rep am I on", 40.0) == "Rep 2 of 6"
    assert say(c, "summary", 40.0) == "Rep 1 29 point 0"


def test_end_workout_early_gives_summary_and_resets() -> None:
    c = Coach(Config())
    say(c, "6 by 200, 60 rest", 0.0)
    say(c, "go", 0.0)
    run_until(c, 0.0, 4.0)
    say(c, "done", 35.0, tap=34.0)
    assert say(c, "end workout", 50.0) == "Workout done Rep 1 30 point 0"
    assert phase(c) is Phase.IDLE
    assert c.plan is None
    assert not c.timer.running
    assert [e["type"] for e in c.drain_events()][-1] == "summary"
    assert say(c, "end workout", 51.0) == "Workout ended"


def test_go_after_finishing_reruns_the_plan() -> None:
    c = Coach(Config())
    say(c, "1 by 400", 0.0)
    say(c, "go", 0.0)
    run_until(c, 0.0, 4.0)
    say(c, "done", 70.0, tap=69.0)
    assert phase(c) is Phase.FINISHED
    assert say(c, "done", 71.0) == "Workout done"
    assert say(c, "go", 80.0) == "Ready"
    assert c.splits == []
    assert c.plan is not None


def test_new_plan_replaces_old_one() -> None:
    c = Coach(Config())
    say(c, "4 by 400", 0.0)
    say(c, "go", 0.0)
    run_until(c, 0.0, 4.0)
    say(c, "8 by 200", 20.0)
    assert c.plan is not None
    assert c.plan.reps == 8
    assert phase(c) is Phase.READY
    assert c.rep_start is None


# --- freestyle ---------------------------------------------------------------------------------


def test_freestyle_done_logs_split_after_any_rest() -> None:
    c = Coach(Config())
    assert say(c, "done", 0.0) == "Resting 1 minute 30"  # nothing timed yet: just rest
    run_until(c, 0.0, 90.0)
    assert c.rep_start == 90.0
    assert say(c, "done", 141.0, tap=140.0) == "Rep 1 50 point 0 Resting 1 minute 30"
    assert say(c, "what rep am I on", 141.0) == "Rep 2"


def test_freestyle_stop_keeps_m1_meaning() -> None:
    c = Coach(Config())
    say(c, "start my rest", 0.0)
    assert say(c, "stop", 10.0) == "Stopped"
    assert phase(c) is Phase.IDLE
    assert c.rep_start is None


def test_go_with_no_plan_starts_a_freestyle_rep() -> None:
    c = Coach(Config())
    assert say(c, "go", 0.0) == "Ready"
    run_until(c, 0.0, 4.0)
    assert say(c, "done", 50.0) == "Rep 1 46 point 0 Resting 1 minute 30"


def test_unknown_intent() -> None:
    assert Coach(Config()).handle(Intent(IntentKind.UNKNOWN), 0.0).text == "Didn't catch that"


# --- summaries and phrases ---------------------------------------------------------------------


def test_summarize() -> None:
    assert summarize([]) is None
    sm = summarize([48.0, 47.0, 48.5, 49.0, 50.0, 51.0])
    assert sm is not None
    assert sm.average == pytest.approx(48.9167, abs=1e-3)
    assert sm.fastest == (2, 47.0)
    assert sm.slowest == (6, 51.0)
    assert sm.fade == pytest.approx((50.0 + 51.0) / 2 - (48.0 + 47.0) / 2)
    one = summarize([48.0])
    assert one is not None
    assert one.fade is None


def test_summary_phrase_variants() -> None:
    even = summarize([48.0, 48.2, 48.1])
    faster = summarize([50.0, 49.0, 48.0])
    assert even is not None
    assert faster is not None
    assert ph.summary(even)[-1] == "Even pace"
    assert ph.summary(faster)[-3:] == ("Finished faster by", "2", "point 0")


@pytest.mark.parametrize(
    ("secs", "parts"),
    [
        (48.24, ("48", "point 2")),
        (48.25, ("48", "point 3")),
        (59.96, ("1", "oh 0", "point 0")),
        (65.3, ("1", "oh 5", "point 3")),
        (312.0, ("5", "12", "point 0")),
    ],
)
def test_time_phrase(secs: float, parts: tuple[str, ...]) -> None:
    assert ph.time(secs) == parts


def test_distance_phrase() -> None:
    assert ph.distance(300) == "300"
    assert ph.distance(1609) == "1 mile"
    assert ph.distance(3218) == "2 miles"
