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
