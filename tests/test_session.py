from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from stride.audio.vad import FRAME, FRAME_S, Audio, EndReason, Utterance
from stride.config import Config
from stride.intent import Intent, IntentKind
from stride.log import SessionLog, read_log
from stride.session import Session
from stride.stt import Transcript
from stride.workout import Speech
from stride.workout import phrases as ph

RNG = np.random.default_rng(1)


def noise(db: float) -> Audio:
    x = RNG.standard_normal(FRAME)
    out: Audio = (x / np.sqrt(np.mean(x**2)) * 10 ** (db / 20)).astype(np.float32)
    return out


class FakeSpeaker:
    output_latency = 0.1

    def __init__(self) -> None:
        self.said: list[Speech] = []
        self.held = False
        self.holds = 0

    def say(self, speech: Speech) -> None:
        self.said.append(speech)

    def hold(self) -> None:
        self.held = True
        self.holds += 1

    def release(self) -> None:
        self.held = False


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def frames(self, s: Session, seconds: float, db: float) -> Utterance | None:
        out = None
        for _ in range(round(seconds / FRAME_S)):
            self.t += FRAME_S
            u = s.on_frame(noise(db), self.t)
            out = out or u
        return out


@pytest.fixture
def log(tmp_path: Path) -> SessionLog:
    return SessionLog(tmp_path, "post_rep", save_audio=False, now=datetime(2026, 11, 21, 7, 30))


def test_tap_speak_reply_and_latency_log(log: SessionLog) -> None:
    spk, clk = FakeSpeaker(), Clock()
    s = Session(Config(), spk, log)
    clk.frames(s, 1.0, -60)  # idle: floor learned
    assert s.floor_db == pytest.approx(-60, abs=1)

    assert s.tap(clk.t) is None
    assert s.listening
    assert spk.held
    assert clk.frames(s, 0.2, -60) is None
    assert clk.frames(s, 0.8, -30) is None  # speaking
    u = clk.frames(s, 1.0, -60)
    assert u is not None
    assert u.reason is EndReason.SILENCE
    assert not s.listening
    assert not spk.held

    stt_done = u.t_done + 0.3
    h = s.heard(u, Transcript("Start my rest.", 0.3, -0.2, 0.01), stt_done)
    assert h.intent.kind is IntentKind.START_REST
    assert spk.said[-1].text == "Resting 1 minute 30"
    assert s.coach.timer.running

    s.on_spoken(spk.said[-1], stt_done + 0.02)
    log.close()
    (rec,) = read_log(log.path)
    assert rec["type"] == "utterance"
    assert rec["transcript"] == "Start my rest."
    assert rec["intent"] == "start_rest"
    assert rec["reply"] == "Resting 1 minute 30"
    assert rec["audio"] is None  # evaluation recording is off by default
    lat = rec["latency"]
    wait = Config().audio.silence_end_ms / 1000
    assert lat["hangover_s"] == pytest.approx(wait, abs=0.03)
    assert lat["stt_s"] == 0.3
    assert lat["reply_s"] == pytest.approx(0.12)  # 20 ms to start + 100 ms output latency
    # end of speech -> heard = hangover + stt + reply
    assert lat["end_to_end_s"] == pytest.approx(wait + 0.3 + 0.12, abs=0.03)
    assert log.path.name == "20261121-073000-post_rep.jsonl"


def test_second_tap_ends_utterance_without_silence_wait() -> None:
    spk, clk = FakeSpeaker(), Clock()
    s = Session(Config(), spk)
    clk.frames(s, 1.0, -60)
    s.tap(clk.t)
    clk.frames(s, 0.6, -30)
    u = s.tap(clk.t)
    assert u is not None
    assert u.reason is EndReason.TAP
    assert u.hangover_s == pytest.approx(0.0, abs=0.001)
    assert not spk.held


def test_tap_with_no_speech_says_didnt_catch_that() -> None:
    spk, clk = FakeSpeaker(), Clock()
    s = Session(Config(), spk)
    clk.frames(s, 1.0, -60)
    s.tap(clk.t)
    u = clk.frames(s, 4.0, -60)
    assert u is not None
    assert u.reason is EndReason.NO_SPEECH
    h = s.heard(u, None, clk.t)
    assert h.intent.kind is IntentKind.UNKNOWN
    assert spk.said[-1].text == "Didn't catch that"


def test_preroll_catches_speech_that_started_before_the_tap() -> None:
    spk, clk = FakeSpeaker(), Clock()
    s = Session(Config(), spk)
    clk.frames(s, 1.0, -60)
    clk.frames(s, 0.2, -30)  # user starts talking a beat before tapping
    s.tap(clk.t)
    clk.frames(s, 0.5, -30)
    u = clk.frames(s, 1.0, -60)
    assert u is not None
    assert u.t_speech_start == pytest.approx(clk.t - 1.7, abs=0.03)


def test_floor_is_not_updated_by_the_users_speech() -> None:
    spk, clk = FakeSpeaker(), Clock()
    s = Session(Config(), spk)
    clk.frames(s, 2.0, -60)
    s.tap(clk.t)
    clk.frames(s, 3.0, -25)
    clk.frames(s, 1.0, -60)
    assert s.floor_db == pytest.approx(-60, abs=1)


def test_cues_flow_to_speaker_and_log(log: SessionLog) -> None:
    spk = FakeSpeaker()
    s = Session(Config(), spk, log)
    s.coach.timer.start(0.0, 90)
    (cue,) = s.tick(30.0)
    assert spk.said == [cue]
    s.on_spoken(cue, 30.05)  # heard at 30.15: 150 ms late
    s.on_dropped(Speech(("5",), 85.0, 5))
    log.close()
    a, b = read_log(log.path)
    assert a == {"v": 1, "type": "cue", "mark": 60, "late_s": pytest.approx(0.15)}
    assert b["type"] == "cue_dropped"


def test_flush_writes_records_whose_reply_never_played(log: SessionLog) -> None:
    spk, clk = FakeSpeaker(), Clock()
    s = Session(Config(), spk, log)
    clk.frames(s, 1.0, -60)
    s.tap(clk.t)
    clk.frames(s, 0.5, -30)
    u = s.tap(clk.t)
    assert u is not None
    s.heard(u, Transcript("stop", 0.2), clk.t)
    s.flush()
    log.close()
    (rec,) = read_log(log.path)
    assert rec["latency"]["end_to_end_s"] is None


def test_save_audio_writes_wav_next_to_log(tmp_path: Path) -> None:
    log = SessionLog(tmp_path, None, save_audio=True, now=datetime(2026, 11, 21, 7, 30))
    spk, clk = FakeSpeaker(), Clock()
    s = Session(Config(), spk, log)
    clk.frames(s, 1.0, -60)
    s.tap(clk.t)
    clk.frames(s, 0.5, -30)
    u = s.tap(clk.t)
    assert u is not None
    s.heard(u, Transcript("stop", 0.2), clk.t)
    s.flush()
    log.close()
    (rec,) = read_log(log.path)
    assert rec["audio"] == "20261121-073000/utt-001.wav"
    assert (tmp_path / rec["audio"]).stat().st_size > 1000


def test_env_tag_is_sanitized_in_filename(tmp_path: Path) -> None:
    log = SessionLog(tmp_path, "Wind / 20mph", save_audio=False, now=datetime(2026, 1, 2, 3, 4, 5))
    assert log.path.name == "20260102-030405-wind-20mph.jsonl"
    log.close()


def test_done_split_ends_at_the_tap_and_is_logged(log: SessionLog) -> None:
    spk, clk = FakeSpeaker(), Clock()
    s = Session(Config(), spk, log)
    clk.frames(s, 1.0, -60)
    s.coach.handle(Intent(IntentKind.GO), clk.t)
    for _ in range(round(4.2 / 0.05)):  # a little past Go: float steps drift
        clk.t += 0.05
        s.tick(clk.t)
    go_at = s.coach.rep_start
    assert go_at is not None

    clk.frames(s, 1.0, -60)
    tap_at = clk.t
    s.tap(tap_at)
    clk.frames(s, 0.4, -30)
    u = clk.frames(s, 1.0, -60)
    assert u is not None
    h = s.heard(u, Transcript("done", 0.3), clk.t + 0.3)  # handled ~1.3 s after the tap
    split = tap_at - go_at
    assert h.reply.parts[:3] == ("Rep 1", *ph.time(split))
    s.flush()
    log.close()
    split_rec = next(r for r in read_log(log.path) if r["type"] == "split")
    assert split_rec["seconds"] == pytest.approx(split, abs=0.01)
