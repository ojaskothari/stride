# stride

[![CI](https://github.com/ojaskothari/stride/actions/workflows/ci.yml/badge.svg)](https://github.com/ojaskothari/stride/actions/workflows/ci.yml)

A hands-free voice timer for track and gym workouts, built for the conditions where voice control
usually fails: out of breath, in the wind, in a crowd.

Finish a rep, tap your earbud, say "start my rest." stride starts the timer and talks you through it:
"1 minute left" … "30 seconds" … "5, 4, 3, 2, 1, go." No phone, no watch, no internet. Everything
runs on-device.

The engineering core is a **noise study**: measuring how breathing, wind and crowd noise break
on-device speech recognition, how much noise suppression and a Whisper model fine-tuned on workout
speech recover, and how much gap remains, all evaluated on held-out speakers, noise and phrasings.

> **Status:** M1 (timer product) is built: tap a key, speak, and stride starts, extends, reports and
> stops a rest with spoken countdown cues. It's tested end to end on synthesized speech; live trials
> with real voices are next. Earbud tap, workout plans and the noise study are later milestones, so
> there are no accuracy results yet.

## Try it (macOS)

```sh
.venv/bin/pip install -e '.[dev,audio]'
.venv/bin/stride run                 # first run: ~1 min to render voice cues, then instant
.venv/bin/stride run --env post_rep  # tag the session's conditions for later scoring
```

Tap any key, then speak. stride ends the utterance after 0.6 s of silence, or tap again to finish
right away. `q` quits. macOS asks for microphone access for your terminal on first run.

| Say | stride |
|---|---|
| "start my rest" / "rest" | starts the default rest (90 s): "Resting 1 minute 30" |
| "start my 3 minute rest" / "rest 1:30" / "rest a minute thirty" | starts that rest |
| "how much time left?" | "1 minute 25 left" |
| "add 30 seconds" / "another minute" | extends the rest |
| "stop" / "skip" / "I'm ready" | ends the rest |

Cues: "2 minutes left" (rests of 3:00 or more), "1 minute left", "30 seconds", "10", "5, 4, 3, 2,
1", "Go". Anything unrecognized gets a short "Didn't catch that", never a follow-up question.

Each session is logged to `logs/<time>[-env].jsonl`: transcript, intent, reply, noise floor, and
latency per stage (silence wait, speech recognition, reply playback, end of speech → reply heard).
Raw audio is saved only when `log.save_audio = true`.

### How it works

```
key tap ─► mic (always open, 0.5 s pre-roll) ─► energy VAD ─► Whisper (on-device) ─► intent parser
                                                                                          │
         spoken reply / cues ◄─ pre-rendered clips ◄─ coach ◄─ rest timer (pure, fake-clock tested)
```

- **Parser** is deterministic rules: digits and number words, "1:30", "two and a half minutes". It
  deliberately does no fuzzy matching of mishearings. That's a measured baseline in the noise study.
- **Cues** are clips rendered once with macOS `say` and played in-process. Spawning `say` per cue
  measured ~1.2 s of delay, so a countdown can't work that way ([spikes](docs/spikes.md)).
- **Speech recognition** runs on a worker thread, so decoding never delays a countdown cue.

## Docs

- [Decision log](docs/decisions.md): design decisions and the evidence behind them
- [Spikes](docs/spikes.md): hardware and library measurements on the target laptop

## Development

Requires Python 3.11+.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest            # tests
.venv/bin/ruff check .      # lint
.venv/bin/mypy              # type check (strict)
.venv/bin/stride config     # print the resolved config
```

Configuration lives in [`config.toml`](config.toml). Every value has a built-in default, so you
only need to keep the keys you change, and unknown keys are rejected.

**macOS + iCloud Drive:** if the repo lives in an iCloud-synced folder (e.g. Desktop), iCloud
re-applies the `hidden` flag to the venv's `.pth` file, and Python 3.12+ skips hidden `.pth` files,
so `import stride` fails. iCloud ignores names ending in `.nosync`, so create the venv as
`python3.12 -m venv .venv.nosync && ln -s .venv.nosync .venv`. That also keeps iCloud from
uploading the venv.

Live audio needs the optional extra: `.venv/bin/pip install -e '.[dev,audio]'`.

## Layout

```
src/stride/
  trigger/   push-to-talk, earbud media key, wake word
  audio/     capture, noise-floor calibration, VAD, segmentation
  enhance/   noise suppression (none / RNNoise / DeepFilterNet)
  stt/       speech-to-text (faster-whisper)
  intent/    deterministic intent parser
  workout/   timer, cues, plans, splits, summaries (pure logic)
  tts/       spoken cues (non-blocking)
  log/       JSONL session logs for live accuracy scoring
  config.py  typed, validated config
  cli.py     `stride` command
```

## License

MIT
