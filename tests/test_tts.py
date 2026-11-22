from __future__ import annotations

import array
import re
import wave
from pathlib import Path

import pytest

from stride.config import Config
from stride.tts import catalog
from stride.tts.cache import Clip, ClipCache, read_wav, split_on_silence, trim, write_wav
from stride.tts.queue import SpeechQueue, is_stale
from stride.workout import Speech
from stride.workout import phrases as ph

RATE = 1000  # tiny sample rate keeps fake clips small


class FakeRenderer:
    """Mimics `say -o`: 100 ms silence, then 200 ms of 'speech' per phrase, phrases separated by
    their [[slnc N]] markers, then 100 ms silence. ``merge`` drops the gaps (a bad split)."""

    def __init__(self, merge: bool = False) -> None:
        self.calls: list[str] = []
        self.merge = merge

    def __call__(self, text: str, voice: str, out: Path, sample_rate: int) -> None:
        self.calls.append(text)
        pieces = re.split(r"\s*\[\[slnc (\d+)\]\]\s*", text)
        tone = [8000 if i % 2 else -8000 for i in range(200)]
        samples = [0] * 100 + tone
        for gap_ms in pieces[1::2]:
            samples += ([] if self.merge else [0] * int(gap_ms)) + tone
        samples += [0] * 100
        write_wav(out, Clip(array.array("h", samples), sample_rate))


@pytest.fixture
def renderer() -> FakeRenderer:
    return FakeRenderer()


@pytest.fixture
def cache(tmp_path: Path, renderer: FakeRenderer) -> ClipCache:
    return ClipCache(tmp_path / "tts", render=renderer, sample_rate=RATE)


# --- cache ------------------------------------------------------------------------------------


def test_render_once_then_cached(cache: ClipCache, renderer: FakeRenderer) -> None:
    assert not cache.has("Go")
    clip = cache.get("Go")
    assert cache.has("Go")
    cache.get("Go")
    assert renderer.calls == ["Go."]  # trailing period: sentence-final intonation
    assert clip.sample_rate == RATE
    # Trimmed to the tone plus 20 ms padding each side.
    assert len(clip.samples) == 200 + 2 * 20


def test_cache_survives_restart(tmp_path: Path, renderer: FakeRenderer) -> None:
    ClipCache(tmp_path, render=renderer, sample_rate=RATE).get("Stopped")
    again = ClipCache(tmp_path, render=renderer, sample_rate=RATE)
    assert again.has("Stopped")
    again.get("Stopped")
    assert renderer.calls == ["Stopped."]


def test_cache_key_depends_on_voice_and_text(tmp_path: Path) -> None:
    a = ClipCache(tmp_path, voice="Samantha")
    b = ClipCache(tmp_path, voice="Fred")
    assert a.path("Go") != b.path("Go")
    assert a.path("Go") != a.path("go")


def test_prerender_batches_and_splits(cache: ClipCache, renderer: FakeRenderer) -> None:
    cache.get("5")
    seen: list[tuple[int, int]] = []
    texts = ["5", "4", "3", "2", "1", "Go", "4"]
    n = cache.prerender(texts, workers=2, progress=lambda d, t: seen.append((d, t)))
    assert n == 5
    assert len(renderer.calls) == 1 + 2  # the earlier single render, then one call per batch
    assert all("[[slnc" in c for c in renderer.calls[1:])
    assert seen[-1] == (5, 5)
    for t in texts:
        assert len(cache.get(t).samples) == 200 + 2 * 20
    assert len(renderer.calls) == 3  # everything came from the cache


def test_prerender_falls_back_when_split_count_mismatches(tmp_path: Path) -> None:
    r = FakeRenderer(merge=True)
    c = ClipCache(tmp_path, render=r, sample_rate=RATE)
    assert c.prerender(["a", "b", "c"], workers=1) == 3
    assert r.calls[1:] == ["a.", "b.", "c."]  # batch failed to split -> one render each
    assert all(c.has(t) for t in "abc")


def test_prerender_nothing_missing(cache: ClipCache, renderer: FakeRenderer) -> None:
    cache.get("Go")
    assert cache.prerender(["Go"]) == 0
    assert renderer.calls == ["Go."]


def test_split_on_silence() -> None:
    tone = [9000] * 10
    x = array.array("h", [0] * 5 + tone + [0] * 30 + tone + [0] * 5 + tone + [0] * 5)
    segs = split_on_silence(x, RATE, min_gap_s=0.02)
    assert len(segs) == 2  # the 5-sample gap is inside a phrase, not between phrases
    assert split_on_silence(array.array("h", [0] * 50), RATE, 0.02) == []


def test_failed_render_leaves_no_partial_file(tmp_path: Path) -> None:
    def broken(text: str, voice: str, out: Path, sample_rate: int) -> None:
        out.write_bytes(b"not a wav")
        raise RuntimeError("say crashed")

    c = ClipCache(tmp_path, render=broken, sample_rate=RATE)
    with pytest.raises(RuntimeError, match="say crashed"):
        c.get("Go")
    assert list(tmp_path.iterdir()) == []


def test_trim_keeps_all_silent_clip() -> None:
    silent = array.array("h", [0] * 50)
    assert trim(silent, RATE) == silent


def test_read_wav_rejects_stereo(tmp_path: Path) -> None:
    p = tmp_path / "s.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"\0\0" * 20)
    with pytest.raises(ValueError, match="mono"):
        read_wav(p)


# --- catalog ----------------------------------------------------------------------------------


def test_resolve_prefers_whole_clip() -> None:
    parts = ("Resting", "1 minute", "30")
    assert catalog.resolve(parts, lambda t: True) == ["Resting 1 minute 30"]
    assert catalog.resolve(parts, lambda t: t in parts) == list(parts)
    # A fragment is missing ("8 by 300"): render the whole phrase once, not each fragment.
    assert catalog.resolve(parts, lambda t: False) == ["Resting 1 minute 30"]
    assert catalog.resolve(("Go",), lambda t: False) == ["Go"]


def test_catalog_covers_a_normal_session() -> None:
    whole = catalog.texts(catalog.whole_phrases(Config().cues.marks))
    assert "Resting 1 minute 30" in whole
    assert "Resting 10 minutes" in whole
    assert "Added 30 seconds" in whole
    assert "1 minute 25 left" in whole
    assert "Go" in whole
    assert "Didn't catch that" in whole
    assert len(whole) < 200  # first-run render stays around half a minute


def test_every_spoken_duration_is_whole_or_fully_fragmented() -> None:
    # A stitched phrase must never stall on a fragment render mid-workout.
    whole = catalog.texts(catalog.whole_phrases(Config().cues.marks))
    frags = catalog.fragments()
    for secs in range(1, 3601):
        for phrase in (ph.resting(secs), ph.added(secs), ph.remaining(secs)):
            assert " ".join(phrase) in whole or set(phrase) <= frags, phrase


# --- queue ------------------------------------------------------------------------------------


def cue(mark: int, due: float) -> Speech:
    return Speech(ph.cue(mark), due, mark)


def test_stale_rules() -> None:
    assert not is_stale(cue(3, 10.0), 10.5)
    assert is_stale(cue(3, 10.0), 10.51)
    assert not is_stale(cue(30, 10.0), 13.0)
    assert is_stale(cue(30, 10.0), 13.1)
    assert not is_stale(cue(0, 10.0), 99.0)  # "Go" is never dropped
    assert not is_stale(Speech(ph.STOPPED), 99.0)  # replies are never stale


def test_queue_drops_stale_and_keeps_order() -> None:
    q = SpeechQueue()
    for m, due in [(5, 85.0), (4, 86.0), (3, 87.0)]:
        q.push(cue(m, due))
    popped = q.pop(86.2)
    assert popped is not None
    assert popped.mark == 4
    assert [s.mark for s in q.dropped] == [5]
    assert q.pop(86.3) is not None
    assert q.pop(86.3) is None


def test_hold_blocks_output_until_release() -> None:
    q = SpeechQueue()
    q.push(cue(60, 30.0))
    q.hold()
    assert q.pop(30.0) is None
    q.release()
    assert q.pop(30.1) is not None


def test_reply_clears_queued_cues() -> None:
    q = SpeechQueue()
    q.push(cue(2, 88.0))
    q.push(cue(1, 89.0))
    q.push_reply(Speech(ph.STOPPED))
    assert len(q) == 1
    popped = q.pop(88.0)
    assert popped is not None
    assert popped.parts == ph.STOPPED


def test_workout_phrases_are_fully_fragmented() -> None:
    frags = catalog.fragments()
    whole = catalog.texts(catalog.whole_phrases(Config().cues.marks))
    for tenths in range(1, 15 * 600):  # every split up to 15 minutes
        assert set(ph.time(tenths / 10)) <= frags, tenths
    for total in range(1, 51):
        for k in range(1, total + 1):
            assert set(ph.go_rep(k, total)) <= frags
    for p in (ph.READY, ph.SAY_GO, ph.NO_WORKOUT, ph.NO_SPLITS, ph.WORKOUT_ENDED):
        assert " ".join(p) in whole
