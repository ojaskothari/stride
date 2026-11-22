"""``stride run``: the live loop. Needs the ``[audio]`` extra.

One loop on the main thread at 10 ms resolution: drain mic frames, poll the keyboard, collect STT
results, tick the timer. STT runs on a worker thread so a 300 ms decode never delays a cue.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from stride import __version__
from stride.audio.vad import MicPermissionError, Utterance, check_signal
from stride.config import Config
from stride.log import SessionLog
from stride.session import Session
from stride.stt import Transcript
from stride.workout import Speech

TICK_S = 0.01


def _progress(done: int, total: int) -> None:
    print(f"\r  rendering voice cues {done}/{total} (first run only)", end="", flush=True)


def run(cfg: Config, env: str | None) -> int:
    if cfg.enhance.backend != "none":
        print(f"stride: enhance.backend={cfg.enhance.backend!r} arrives in M4; use 'none'")
        return 2
    print("stride: loading…", flush=True)

    # Imported here so `stride config` etc. work without the [audio] extra.
    from stride.audio.mic import Mic
    from stride.stt.whisper import WhisperTranscriber
    from stride.tts.player import AudioPlayer, SilentPlayer

    session: Session | None = None

    def on_spoken(speech: Speech, started: float) -> None:
        if session is not None:
            session.on_spoken(speech, started)

    player: AudioPlayer | SilentPlayer
    if cfg.tts.backend == "say":
        from stride.tts import catalog
        from stride.tts.cache import ClipCache, default_cache_dir

        cache = ClipCache(default_cache_dir(), voice=cfg.tts.voice)
        texts = catalog.texts(catalog.whole_phrases(cfg.cues.marks)) | catalog.fragments()
        if cache.prerender(texts, progress=_progress):
            print()
        player = AudioPlayer(cache, on_spoken=on_spoken)
    else:
        player = SilentPlayer(on_spoken=on_spoken)

    t0 = time.perf_counter()
    stt = WhisperTranscriber(cfg.stt.model, cfg.stt.compute_type, cfg.stt.vocab_prompt)
    print(f"  {stt.name} ready in {time.perf_counter() - t0:.1f}s")

    mic = Mic()
    mic.start()
    try:
        warmup = [f for f, _ in mic.read(1.0)]
        deadline = time.monotonic() + 0.6
        while time.monotonic() < deadline:
            warmup += [f for f, _ in mic.read(0.05)]
        check_signal(warmup)
    except MicPermissionError as e:
        mic.close()
        player.close()
        print(f"stride: {e}", file=sys.stderr)
        return 3

    log = SessionLog(Path(cfg.log.dir), env, cfg.log.save_audio, datetime.now())
    session = Session(cfg, player, log, lead_s=player.output_latency)
    log.write(
        {
            "type": "session_start",
            "wall": datetime.now().isoformat(timespec="seconds"),
            "stride": __version__,
            "env": env,
            "stt": stt.name,
            "enhancer": cfg.enhance.backend,
            "mic": mic.name,
            "output_latency_s": round(player.output_latency, 3),
            "config": asdict(cfg),
        }
    )
    print(
        f"  mic: {mic.name} · output latency {player.output_latency * 1000:.0f} ms"
        f" · log: {log.path}\n"
        "\nTap any key, then speak. Tap again to finish early. q to quit.\n"
        '  try: "start my rest" · "add 30 seconds" · "how much time left" · "stop"\n'
    )

    from stride.trigger.keyboard import Keyboard

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stride-stt")
    jobs: list[tuple[Utterance, Future[Transcript]]] = []
    utterances = 0

    def submit(u: Utterance) -> None:
        if len(u.audio) == 0:
            _show(session, u, None)
            return
        jobs.append((u, pool.submit(stt.transcribe, u.audio)))

    try:
        with Keyboard() as kb:
            while True:
                for frame, t in mic.read(TICK_S):
                    if (u := session.on_frame(frame, t)) is not None:
                        submit(u)
                key = kb.poll()
                if key == "q":
                    break
                if key is not None:
                    if (u := session.tap(time.monotonic())) is not None:
                        submit(u)
                    elif session.listening:
                        print("  ● listening…", flush=True)
                for u, fut in [j for j in jobs if j[1].done()]:
                    jobs.remove((u, fut))
                    utterances += 1
                    _show(session, u, fut.result())
                for cue in session.tick(time.monotonic()):
                    print(f"  stride: {cue.text}", flush=True)
                if isinstance(player, AudioPlayer):
                    for s in player.dropped():
                        session.on_dropped(s)
    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown(wait=True)
        mic.close()
        player.close()
        session.flush()
        log.write({"type": "session_end", "wall": datetime.now().isoformat(timespec="seconds")})
        log.close()
        print(f"\nstride: session saved to {log.path}")
    return 0


def _show(session: Session, u: Utterance, t: Transcript | None) -> None:
    h = session.heard(u, t, time.monotonic())
    said = f'"{h.transcript}"' if h.transcript else "(no speech)"
    slot = f" {h.intent.seconds}s" if h.intent.seconds is not None else ""
    stt_ms = f" · stt {t.seconds * 1000:.0f} ms" if t else ""
    hang = f" · wait {u.hangover_s * 1000:.0f} ms" if u.hangover_s is not None else ""
    print(f"  you: {said} → {h.intent.kind.value}{slot}{stt_ms}{hang}", flush=True)
    print(f"  stride: {h.reply.text}", flush=True)
