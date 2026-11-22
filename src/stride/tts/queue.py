"""What to say next. Pure: drops cues that would be heard too late, holds output while listening."""

from __future__ import annotations

from collections import deque

from stride.workout import Speech

# A countdown digit heard half a second late is worse than silence; "30 seconds" 3 s late is wrong.
# "Go" is always said: it's the one cue that must never be lost.
COUNTDOWN_MAX = 10
COUNTDOWN_STALE_S = 0.5
OTHER_STALE_S = 3.0


def is_stale(speech: Speech, now: float) -> bool:
    if speech.due is None or speech.mark in (None, 0):
        return False
    assert speech.mark is not None
    limit = COUNTDOWN_STALE_S if speech.mark <= COUNTDOWN_MAX else OTHER_STALE_S
    return now - speech.due > limit


class SpeechQueue:
    def __init__(self) -> None:
        self._q: deque[Speech] = deque()
        self._held = False
        self.dropped: list[Speech] = []  # for the session log

    def push(self, speech: Speech) -> None:
        self._q.append(speech)

    def push_reply(self, speech: Speech) -> None:
        """A reply to a command changes the timer, so queued cues no longer apply."""
        self.clear_cues()
        self._q.append(speech)

    def clear_cues(self) -> None:
        self._q = deque(s for s in self._q if s.mark is None)

    def hold(self) -> None:
        """Stop releasing speech (the mic is capturing; don't talk over the user)."""
        self._held = True

    def release(self) -> None:
        self._held = False

    def pop(self, now: float) -> Speech | None:
        if self._held:
            return None
        while self._q:
            s = self._q.popleft()
            if not is_stale(s, now):
                return s
            self.dropped.append(s)
        return None

    def __len__(self) -> int:
        return len(self._q)
