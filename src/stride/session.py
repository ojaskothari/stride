"""Session: wires trigger, VAD, STT result, parser, coach, speech and log together.

No audio devices, threads or real clock in here: the live runner (``stride.run``) feeds it frames,
taps, transcripts and times, so the whole flow is unit-testable with fakes.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from stride.audio.vad import FRAME_S, Audio, EndReason, NoiseFloor, Segmenter, Utterance, frame_db
from stride.config import Config
from stride.intent import Intent, IntentKind, parse
from stride.log import SessionLog
from stride.stt import Transcript
from stride.workout import Coach, Speech

PREROLL_S = 0.5
FALLBACK_FLOOR_DB = -60.0  # until the floor tracker has 200 ms of idle audio


class Speaker(Protocol):
    @property
    def output_latency(self) -> float: ...

    def say(self, speech: Speech) -> None: ...

    def hold(self) -> None: ...

    def release(self) -> None: ...


@dataclass(frozen=True)
class Heard:
    """What the live UI prints after an utterance."""

    transcript: str
    intent: Intent
    reply: Speech


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(x, nd)


class Session:
    def __init__(
        self,
        cfg: Config,
        speaker: Speaker,
        log: SessionLog | None = None,
        *,
        lead_s: float = 0.0,
    ) -> None:
        self.cfg = cfg
        self.coach = Coach(cfg, lead_s=lead_s)
        self._speaker = speaker
        self._log = log
        self._floor = NoiseFloor()
        self._preroll: deque[tuple[Audio, float]] = deque(maxlen=round(PREROLL_S / FRAME_S))
        self._seg: Segmenter | None = None
        self._next_id = 1
        self._pending: dict[int, dict[str, Any]] = {}  # utterance records waiting for reply audio
        self._lock = threading.Lock()

    @property
    def listening(self) -> bool:
        return self._seg is not None

    @property
    def floor_db(self) -> float:
        return self._floor.db if self._floor.ready else FALLBACK_FLOOR_DB

    # --- audio in -----------------------------------------------------------------------------

    def on_frame(self, frame: Audio, t: float) -> Utterance | None:
        if self._seg is None:
            self._floor.update(frame_db(frame))
            self._preroll.append((frame, t))
            return None
        u = self._seg.push(frame, t)
        if u is not None:
            self._end_listening()
        return u

    def tap(self, now: float) -> Utterance | None:
        """Start listening; a tap while listening ends the utterance right away."""
        if self._seg is not None:
            u = self._seg.finish(now)
            self._end_listening()
            return u
        a = self.cfg.audio
        self._speaker.hold()  # don't talk over the user (or into the mic)
        self._seg = Segmenter(
            t_listen=now,
            floor_db=self.floor_db,
            threshold_db=a.vad_db_above_floor,
            silence_end_s=a.silence_end_ms / 1000,
            max_utterance_s=a.max_utterance_s,
        )
        for frame, t in self._preroll:
            self._seg.push(frame, t)
        self._preroll.clear()
        return None

    def _end_listening(self) -> None:
        self._seg = None
        self._speaker.release()

    # --- understanding and replying -----------------------------------------------------------

    def heard(self, u: Utterance, transcript: Transcript | None, now: float) -> Heard:
        """Act on a finished utterance. ``transcript`` is None when there was no speech."""
        text = transcript.text if transcript else ""
        intent = parse(text) if u.reason is not EndReason.NO_SPEECH else Intent(IntentKind.UNKNOWN)
        utt_id = self._next_id
        self._next_id += 1
        reply = self.coach.handle(intent, now)
        reply = Speech(reply.parts, reply.due, reply.mark, tag=utt_id)

        if self._log is not None:
            record: dict[str, Any] = {
                "type": "utterance",
                "id": utt_id,
                "wall": datetime.now().isoformat(timespec="seconds"),
                "end": u.reason.value,
                "transcript": text,
                "intent": intent.kind.value,
                "seconds": intent.seconds,
                "reply": reply.text,
                "floor_db": _r(u.floor_db, 1),
                "peak_db": _r(u.peak_db, 1),
                "speech_s": _r(
                    None
                    if u.t_speech_start is None or u.t_speech_end is None
                    else u.t_speech_end - u.t_speech_start
                ),
                "avg_logprob": _r(transcript.avg_logprob) if transcript else None,
                "no_speech_prob": _r(transcript.no_speech_prob) if transcript else None,
                "audio": self._log.save_audio(utt_id, u.audio),
                "latency": {
                    "hangover_s": _r(u.hangover_s),
                    "stt_s": _r(transcript.seconds) if transcript else None,
                    "_speech_end": u.t_speech_end,
                    "_handled": now,
                },
            }
            with self._lock:
                self._pending[utt_id] = record
        self._speaker.say(reply)
        return Heard(text, intent, reply)

    def tick(self, now: float) -> list[Speech]:
        cues = self.coach.tick(now)
        for s in cues:
            self._speaker.say(s)
        return cues

    # --- called from the audio thread as speech starts ----------------------------------------

    def on_spoken(self, speech: Speech, started: float) -> None:
        if self._log is None:
            return
        heard_at = started + self._speaker.output_latency
        if speech.tag is not None:
            with self._lock:
                record = self._pending.pop(speech.tag, None)
            if record is not None:
                self._log.write(_finalize(record, heard_at))
        elif speech.mark is not None and speech.due is not None:
            self._log.write(
                {"type": "cue", "mark": speech.mark, "late_s": _r(heard_at - speech.due)}
            )

    def on_dropped(self, speech: Speech) -> None:
        if self._log is not None and speech.mark is not None:
            self._log.write({"type": "cue_dropped", "mark": speech.mark})

    def flush(self) -> None:
        """Write records whose reply never played (e.g. quit right after speaking)."""
        if self._log is None:
            return
        with self._lock:
            pending, self._pending = list(self._pending.values()), {}
        for record in pending:
            self._log.write(_finalize(record, None))


def _finalize(record: dict[str, Any], heard_at: float | None) -> dict[str, Any]:
    lat = dict(record["latency"])
    end, handled = lat.pop("_speech_end"), lat.pop("_handled")
    # end of speech -> reply audible: the number the < 1.0 s target is about
    lat["end_to_end_s"] = _r(heard_at - end) if heard_at is not None and end is not None else None
    # transcript in hand -> reply audible: parse + queue + playback start + output latency
    lat["reply_s"] = _r(heard_at - handled) if heard_at is not None else None
    return {**record, "latency": lat}
