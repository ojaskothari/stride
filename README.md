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

> **Status:** early development (M0: foundation). Nothing here is usable at the track yet, and there
> are no results to report yet.

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

**macOS note:** if `import stride` fails after an editable install, the `.pth` file pip wrote may
carry the macOS `hidden` flag (common in iCloud-synced folders), and Python 3.12+ skips hidden
`.pth` files. Fix it with `chflags nohidden .venv/lib/python3.*/site-packages/*.pth`.

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
