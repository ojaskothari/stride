from __future__ import annotations

import pytest

from stride.intent import Intent, IntentKind, parse

S, A, Q, X, U = (
    IntentKind.START_REST,
    IntentKind.ADD_TIME,
    IntentKind.STATUS,
    IntentKind.STOP,
    IntentKind.UNKNOWN,
)


@pytest.mark.parametrize(
    ("text", "kind", "seconds"),
    [
        # Start, no duration -> engine applies plan/default.
        ("start my rest", S, None),
        ("Start my rest.", S, None),
        ("rest", S, None),
        ("Coach, start my rest", S, None),
        ("um, start the timer", S, None),
        ("start rest please", S, None),
        ("restart my rest", S, None),
        ("oh, start my rest", S, None),
        # Start with duration: digits (how Whisper usually writes them with our prompt).
        ("start my 3 minute rest", S, 180),
        ("Start my 3-minute rest.", S, 180),
        ("rest 90", S, 90),
        ("rest 90 seconds", S, 90),
        ("rest 90s", S, 90),
        ("rest 3", S, 180),
        ("rest for 2 minutes", S, 120),
        ("set a timer for 45 seconds", S, 45),
        ("rest 1:30", S, 90),
        ("Rest 1.30", S, 90),
        ("rest 1.5 minutes", S, 90),
        ("rest 2 min", S, 120),
        ("rest 3min", S, 180),
        ("rest 1 minute 30 seconds", S, 90),
        ("rest 1 minute and 30 seconds", S, 90),
        ("rest 300", S, 300),
        ("90 seconds", S, 90),
        ("2 minutes", S, 120),
        # Number words.
        ("start my two minute rest", S, 120),
        ("rest ninety", S, 90),
        ("rest ninety seconds", S, 90),
        ("rest forty-five seconds", S, 45),
        ("rest a minute thirty", S, 90),
        ("rest a minute and a half", S, 90),
        ("rest minute and a half", S, 90),
        ("rest two and a half minutes", S, 150),
        ("rest two minutes and a half", S, 150),
        ("rest one thirty", S, 90),
        ("rest two fifteen", S, 135),
        ("rest half a minute", S, 30),
        ("rest one hundred twenty seconds", S, 120),
        ("rest a hundred and twenty seconds", S, 120),
        ("rest a couple of minutes", S, 120),
        ("start a four minute rest", S, 240),
        ("rest twenty five seconds", S, 25),
        # Add time.
        ("add 30 seconds", A, 30),
        ("Add 30 seconds.", A, 30),
        ("add thirty", A, 30),
        ("add a minute", A, 60),
        ("another minute", A, 60),
        ("give me another minute", A, 60),
        ("give me two more minutes", A, 120),
        ("one more minute", A, 60),
        ("three more minutes", A, 180),
        ("30 more seconds", A, 30),
        ("plus 15", A, 15),
        ("add 2", A, 120),
        ("add 5", A, 300),
        ("add 10", A, 10),
        ("another 20", A, 20),
        ("add 10 minutes", A, 600),
        ("add 1.5", A, 90),
        ("extend by 30 seconds", A, 30),
        ("add a minute to my rest", A, 60),
        ("add time", A, None),
        ("add some more time", A, None),
        # Status.
        ("how much time left?", Q, None),
        ("How much time is left?", Q, None),
        ("how long left", Q, None),
        ("how much longer", Q, None),
        ("time left", Q, None),
        ("what's left", Q, None),
        ("time remaining", Q, None),
        ("status", Q, None),
        ("how long is my rest", Q, None),
        # Stop.
        ("stop", X, None),
        ("Stop.", X, None),
        ("skip", X, None),
        ("skip rest", X, None),
        ("stop my rest", X, None),
        ("cancel the timer", X, None),
        ("I'm ready", X, None),
        ("I\u2019m ready", X, None),
        ("end rest", X, None),
        # Unknown -> "didn't catch that", never a follow-up question.
        ("", U, None),
        ("   ", U, None),
        ("...", U, None),
        ("what's the weather", U, None),
        ("one more rep", U, None),
        ("hello", U, None),
        ("rest 5000", U, None),  # over an hour: a misparse, not a rest
        ("rest 0 seconds", U, None),
    ],
)
def test_parse(text: str, kind: IntentKind, seconds: int | None) -> None:
    got = parse(text)
    assert (got.kind, got.seconds) == (kind, seconds), text


def test_intent_equality_ignores_text() -> None:
    assert parse("rest 90") == Intent(IntentKind.START_REST, 90)
    assert parse("rest 90").text == "rest 90"


def test_parse_never_raises_on_junk() -> None:
    for junk in ["!!!", "1:", ":30", "and a half", "half", "a", "minute", "seconds", "9" * 40]:
        parse(junk)


# --- M2: workouts -----------------------------------------------------------------------------

P = IntentKind.PLAN


@pytest.mark.parametrize(
    ("text", "reps", "meters", "rest"),
    [
        ("8 by 300, 90 rest", 8, 300, 90),
        ("8 by 300 90 rest", 8, 300, 90),
        ("8x300, 90 seconds rest", 8, 300, 90),
        ("8 x 300m with 90 seconds rest", 8, 300, 90),
        ("eight by three hundred, ninety rest", 8, 300, 90),
        ("6 by 400 with 2 minute rest", 6, 400, 120),
        ("6 by 400, rest 2 minutes", 6, 400, 120),
        ("6 by 400, 3 rest", 6, 400, 180),
        ("10 by 200 meters, 1:30 rest", 10, 200, 90),
        ("4 by 1 mile, 3 minutes", 4, 1609, 180),
        ("5 by 1k, 2 minutes recovery", 5, 1000, 120),
        ("12 by 100", 12, 100, None),
        ("ten by one hundred", 10, 100, None),
        ("Coach, 8 times 300, a minute thirty rest.", 8, 300, 90),
        # Whisper merges "three hundred, ninety rest" into one number (seen live).
        ("8 by 390 rest.", 8, 300, 90),
        ("4 by 1290 rest", 4, 1200, 90),
        ("6 by 403 rest", 6, 400, 180),
        ("8 by 330", 8, 330, None),  # no rest word: an odd distance is taken at face value
        ("8 by 345 rest", 8, 300, 45),
    ],
)
def test_parse_plan(text: str, reps: int, meters: int, rest: int | None) -> None:
    got = parse(text)
    assert (got.kind, got.reps, got.distance_m, got.seconds) == (P, reps, meters, rest), text


@pytest.mark.parametrize(
    "text",
    [
        "8 by 2",  # no unit and too short to be meters
        "100 by 300",  # too many reps
        "rest 90",  # not a plan
    ],
)
def test_not_a_plan(text: str) -> None:
    assert parse(text).kind is not P


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("go", IntentKind.GO),
        ("Go.", IntentKind.GO),
        ("let's go", IntentKind.GO),
        ("start the workout", IntentKind.GO),
        ("start workout", IntentKind.GO),
        ("done", IntentKind.DONE),
        ("Done.", IntentKind.DONE),
        ("I'm done", IntentKind.DONE),
        ("rep done", IntentKind.DONE),
        ("finished", IntentKind.DONE),
        ("summary", IntentKind.SUMMARY),
        ("how did I do", IntentKind.SUMMARY),
        ("read my splits", IntentKind.SUMMARY),
        ("what rep am I on", IntentKind.REP_STATUS),
        ("What's rep am I on?", IntentKind.REP_STATUS),  # live transcript
        ("how many reps left", IntentKind.REP_STATUS),
        ("rep count", IntentKind.REP_STATUS),
        ("end workout", IntentKind.END_WORKOUT),
        ("end the workout", IntentKind.END_WORKOUT),
        ("finish workout", IntentKind.END_WORKOUT),
        ("stop the workout", IntentKind.END_WORKOUT),
        ("workout over", IntentKind.END_WORKOUT),
        # M1 commands keep their meaning
        ("stop", IntentKind.STOP),
        ("skip", IntentKind.STOP),
        ("how much time left", IntentKind.STATUS),
        ("start my rest", IntentKind.START_REST),
    ],
)
def test_parse_workout_commands(text: str, kind: IntentKind) -> None:
    assert parse(text).kind is kind, text
