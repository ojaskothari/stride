"""Coach: turns intents and clock ticks into things to say. Pure; the live loop does the I/O.

Two ways to train:
- **Plan:** "8 by 300, 90 rest" -> "go" -> 3-2-1-Go -> run -> "done" (split + auto rest) -> Go ->
  ... -> after the last rep, the summary.
- **Freestyle:** any rest's "Go" starts a rep; "done" logs the split and starts the default rest.

Timing: a rep starts when its "Go" is *heard* (the cue's due time); it ends at the user's tap
(``t_trigger``), not when the transcript arrives ~1-2 s later. Rests are timed from the tap too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from stride.config import Config
from stride.intent import Intent, IntentKind
from stride.workout import phrases
from stride.workout.phrases import Phrase
from stride.workout.plan import Plan, summarize
from stride.workout.timer import RestTimer

COUNTDOWN_S = 4.0  # "3" at +1 s, "2", "1", "Go" at +4 s


@dataclass(frozen=True)
class Speech:
    parts: Phrase
    # When the speech should be heard. ``None`` means now (a reply); cues carry their due time so
    # the player can drop ones that arrive too late to be useful.
    due: float | None = None
    mark: int | None = None  # set for countdown cues
    tag: int | None = field(default=None, compare=False)  # links a reply to its log record

    @property
    def text(self) -> str:
        return " ".join(self.parts)


class Phase(StrEnum):
    IDLE = "idle"  # nothing running (freestyle: waiting for a rest)
    READY = "ready"  # plan loaded, waiting for "go"
    COUNTDOWN = "countdown"
    REP = "rep"
    REST = "rest"
    FINISHED = "finished"  # every planned rep done


class Coach:
    def __init__(self, cfg: Config, *, lead_s: float = 0.0) -> None:
        self._cfg = cfg
        self.timer = RestTimer(cfg.cues.marks, cfg.cues.min_rest, lead_s=lead_s)
        self._countdown = RestTimer((3, 2, 1), lead_s=lead_s, min_gap_s=0.0)
        self.phase = Phase.IDLE
        self.plan: Plan | None = None
        self.splits: list[float] = []
        self.rep_start: float | None = None
        # Workout events for the session log (plan, split, summary); the caller drains them.
        self.events: list[dict[str, Any]] = []

    @property
    def rep_number(self) -> int:
        """The rep in progress, or the next one."""
        return len(self.splits) + 1

    @property
    def _total(self) -> int | None:
        return self.plan.reps if self.plan else None

    # --- commands -----------------------------------------------------------------------------

    def handle(self, intent: Intent, now: float, t_trigger: float | None = None) -> Speech:
        """``t_trigger`` is when the user tapped; it ends a rep precisely. Defaults to ``now``."""
        t = now if t_trigger is None else t_trigger
        kind = intent.kind
        handler = {
            IntentKind.PLAN: self._plan,
            IntentKind.GO: self._go,
            IntentKind.DONE: self._done,
            IntentKind.START_REST: self._start_rest,
            IntentKind.ADD_TIME: self._add,
            IntentKind.STATUS: self._status,
            IntentKind.STOP: self._stop,
            IntentKind.REP_STATUS: self._rep_status,
            IntentKind.SUMMARY: self._summary,
            IntentKind.END_WORKOUT: self._end,
        }.get(kind)
        parts = handler(intent, now, t) if handler else phrases.DIDNT_CATCH
        return Speech(parts)

    def _plan(self, intent: Intent, now: float, t: float) -> Phrase:
        assert intent.reps is not None
        assert intent.distance_m is not None
        rest = intent.seconds or self._cfg.rest.default_seconds
        self._reset()
        self.plan = Plan(intent.reps, intent.distance_m, rest)
        self.phase = Phase.READY
        self.events.append({"type": "plan", "reps": intent.reps, "distance_m": intent.distance_m,
                            "rest_s": rest})  # fmt: skip
        return phrases.plan_loaded(intent.reps, intent.distance_m, rest)

    def _go(self, intent: Intent, now: float, t: float) -> Phrase:
        if self.phase is Phase.REST:
            return self._start_rep_now(now)
        if self.phase in (Phase.REP, Phase.COUNTDOWN):
            return phrases.rep_of(self.rep_number, self._total)
        if self.phase is Phase.FINISHED:  # run the same plan again
            plan = self.plan
            self._reset()
            self.plan = plan
        self._countdown.start(now, COUNTDOWN_S)
        self.phase = Phase.COUNTDOWN
        return phrases.READY

    def _done(self, intent: Intent, now: float, t: float) -> Phrase:
        if self.phase is Phase.REP:
            return self._finish_rep(t, None)
        if self.phase is Phase.REST:
            return self._status(intent, now, t)
        if self.phase in (Phase.READY, Phase.COUNTDOWN):
            return phrases.SAY_GO
        if self.phase is Phase.FINISHED:
            return (phrases.WORKOUT_DONE,)
        # Freestyle, no rep timed yet: "done" just means rest now.
        return self._rest(t, self._cfg.rest.default_seconds)

    def _start_rest(self, intent: Intent, now: float, t: float) -> Phrase:
        if self.phase is Phase.REP:  # "start my 2 minute rest" at the end of a rep
            return self._finish_rep(t, intent.seconds)
        return self._rest(t, intent.seconds or self._cfg.rest.default_seconds)

    def _add(self, intent: Intent, now: float, t: float) -> Phrase:
        secs = intent.seconds or self._cfg.rest.default_add_seconds
        if not self.timer.running:
            # "Another minute" right after "Go" means rest again, not "no timer".
            return self._rest(t, secs)
        self.timer.extend(now, secs)
        return phrases.added(secs)

    def _status(self, intent: Intent, now: float, t: float) -> Phrase:
        left = self.timer.remaining(now)
        return phrases.NO_TIMER if left is None else phrases.remaining(left)

    def _stop(self, intent: Intent, now: float, t: float) -> Phrase:
        if self.phase is Phase.COUNTDOWN:
            self._countdown.stop()
            self.phase = Phase.READY if self.plan else Phase.IDLE
            return phrases.STOPPED
        if self.phase is Phase.REST and self.plan is not None:
            return self._start_rep_now(now)  # in a plan, ending the rest means the next rep
        if not self.timer.running:
            return phrases.NO_TIMER
        self.timer.stop()
        self.phase = Phase.IDLE
        return phrases.STOPPED

    def _rep_status(self, intent: Intent, now: float, t: float) -> Phrase:
        if self.phase is Phase.FINISHED:
            return (phrases.WORKOUT_DONE,)
        if self.plan is None and not self.splits:
            return phrases.NO_WORKOUT
        return phrases.rep_of(self.rep_number, self._total)

    def _summary(self, intent: Intent, now: float, t: float) -> Phrase:
        sm = summarize(self.splits)
        return phrases.NO_SPLITS if sm is None else phrases.summary(sm)

    def _end(self, intent: Intent, now: float, t: float) -> Phrase:
        sm = summarize(self.splits)
        if sm is not None and self.phase is not Phase.FINISHED:
            self._log_summary()
        self._reset()
        return phrases.WORKOUT_ENDED if sm is None else phrases.summary(sm, done=True)

    # --- state changes ------------------------------------------------------------------------

    def _rest(self, t: float, secs: int) -> Phrase:
        self.timer.start(t, secs)
        self.phase = Phase.REST
        return phrases.resting(secs)

    def _finish_rep(self, t: float, rest_override: int | None) -> Phrase:
        assert self.rep_start is not None
        split = max(0.0, t - self.rep_start)
        self.splits.append(split)
        k = len(self.splits)
        self.rep_start = None
        self.events.append({"type": "split", "rep": k, "seconds": round(split, 2),
                            "distance_m": self.plan.distance_m if self.plan else None})  # fmt: skip
        if self.plan is not None and k >= self.plan.reps:
            self.timer.stop()
            self.phase = Phase.FINISHED
            self._log_summary()
            sm = summarize(self.splits)
            assert sm is not None
            return (phrases.rep(k), *phrases.time(split), *phrases.summary(sm, done=True))
        rest = rest_override or (self.plan.rest_s if self.plan else self._cfg.rest.default_seconds)
        self.timer.start(t, rest)
        self.phase = Phase.REST
        return phrases.split_then_rest(k, split, rest)

    def _start_rep_now(self, now: float) -> Phrase:
        self.timer.stop()
        self.rep_start = now
        self.phase = Phase.REP
        return phrases.go_rep(self.rep_number, self._total)

    def _reset(self) -> None:
        self.timer.stop()
        self._countdown.stop()
        self.phase = Phase.IDLE
        self.plan = None
        self.splits = []
        self.rep_start = None

    def _log_summary(self) -> None:
        sm = summarize(self.splits)
        if sm is None:
            return
        self.events.append({
            "type": "summary",
            "splits": [round(s, 2) for s in sm.splits],
            "average": round(sm.average, 2),
            "fastest": [sm.fastest[0], round(sm.fastest[1], 2)],
            "slowest": [sm.slowest[0], round(sm.slowest[1], 2)],
            "fade_s": None if sm.fade is None else round(sm.fade, 2),
            "distance_m": self.plan.distance_m if self.plan else None,
        })  # fmt: skip

    # --- clock --------------------------------------------------------------------------------

    def tick(self, now: float) -> list[Speech]:
        out: list[Speech] = []
        for c in self._countdown.tick(now):
            if c.mark == 0:
                self.rep_start = c.due  # the rep starts when "Go" is heard
                self.phase = Phase.REP
            out.append(Speech(phrases.cue(c.mark), c.due, c.mark))
        for c in self.timer.tick(now):
            parts = phrases.cue(c.mark)
            if c.mark == 0 and self.phase is Phase.REST:
                self.rep_start = c.due
                self.phase = Phase.REP
                parts = phrases.go_rep(self.rep_number, self._total)
            out.append(Speech(parts, c.due, c.mark))
        return out

    def drain_events(self) -> list[dict[str, Any]]:
        out, self.events = self.events, []
        return out
