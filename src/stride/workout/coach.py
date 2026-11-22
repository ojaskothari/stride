"""Coach: turns intents and clock ticks into things to say. Pure; the live loop does the I/O."""

from __future__ import annotations

from dataclasses import dataclass

from stride.config import Config
from stride.intent import Intent, IntentKind
from stride.workout import phrases
from stride.workout.phrases import Phrase
from stride.workout.timer import RestTimer


@dataclass(frozen=True)
class Speech:
    parts: Phrase
    # When the speech should be heard. ``None`` means now (a reply); cues carry their due time so
    # the player can drop ones that arrive too late to be useful.
    due: float | None = None
    mark: int | None = None  # set for countdown cues

    @property
    def text(self) -> str:
        return " ".join(self.parts)


class Coach:
    def __init__(self, cfg: Config, *, lead_s: float = 0.0) -> None:
        self._cfg = cfg
        self.timer = RestTimer(cfg.cues.marks, cfg.cues.min_rest, lead_s=lead_s)

    def handle(self, intent: Intent, now: float) -> Speech:
        kind, timer = intent.kind, self.timer
        if kind is IntentKind.START_REST:
            secs = intent.seconds or self._cfg.rest.default_seconds
            timer.start(now, secs)
            return Speech(phrases.resting(secs))
        if kind is IntentKind.ADD_TIME:
            secs = intent.seconds or self._cfg.rest.default_add_seconds
            if not timer.running:
                # "Another minute" right after "Go" means rest again, not "no timer".
                timer.start(now, secs)
                return Speech(phrases.resting(secs))
            timer.extend(now, secs)
            return Speech(phrases.added(secs))
        if kind is IntentKind.STATUS:
            left = timer.remaining(now)
            return Speech(phrases.NO_TIMER if left is None else phrases.remaining(left))
        if kind is IntentKind.STOP:
            if not timer.running:
                return Speech(phrases.NO_TIMER)
            timer.stop()
            return Speech(phrases.STOPPED)
        return Speech(phrases.DIDNT_CATCH)

    def tick(self, now: float) -> list[Speech]:
        return [Speech(phrases.cue(c.mark), c.due, c.mark) for c in self.timer.tick(now)]
