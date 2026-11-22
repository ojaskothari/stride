# Hardware and library spikes

Small experiments run before building, so design decisions rest on measurements. Scripts live in
[`spikes/`](../spikes) and can be rerun. Machine: MacBook Air (Apple Silicon, arm64, 8 cores),
macOS 26.6, Python 3.12.

## 1. Speaking cues with macOS `say` (2026-11-21)

`spikes/say_latency.py`: overhead = wall time of `say <text>` minus the audio length of the same text.

| | Overhead |
|---|---|
| First call (cold) | 6,207 ms |
| Warm, per call, default voice (median over phrases) | ~1,180–1,280 ms |
| `say ""` (process spawn alone) | ~630 ms |
| Render to file: default voice / Samantha / Fred | 1,210 / 770 / 550 ms |

**Conclusion:** spawning `say` per cue adds over a second of delay. Most of it is process and
speech-framework startup, so changing voices doesn't fix it. A countdown of separate calls drifts,
and confirmations alone exceed the 1.0 s budget.

`spikes/prerendered_cues.py`: render clips once with `say -o`, trim silence, play with `sounddevice`.

| | Result |
|---|---|
| Call → first audio callback | median 9 ms, max 31 ms |
| Device output latency (built-in speakers) | 24 ms |
| Trimmed digit clips "5"…"1", "Go" (default voice) | 274–452 ms each |

**Decision:** cues are pre-rendered clips (cached on disk, keyed by text + voice) played in-process.
Start latency drops from ~1,200 ms to ~35 ms, and every countdown digit fits inside its 1 s slot.
Dynamic phrases ("Resting 2 minutes 15") are composed from fragments. The countdown scheduler will
start cues early by the output stream's reported latency, since Bluetooth earbuds add an expected
150–250 ms (not yet measured).

## 2. faster-whisper latency (2026-11-21)

`spikes/stt_latency.py`: four synthetic commands (mean 1.27 s), int8 on CPU, vocabulary prompt on,
8 reps. Synthetic speech, so this measures **speed only, not accuracy** (accuracy is M4).

| Model | Median (beam 1) | p90 | Median (beam 5) | Transcripts |
|---|---|---|---|---|
| `tiny.en` | 146 ms | 150 ms | 157 ms | 4/4 correct |
| `base.en` | 316 ms | 337 ms | 335 ms | 4/4 correct |
| `small.en` | 1,086 ms | 1,135 ms | 1,136 ms | 4/4 correct |

**Conclusions:**
- `small.en` can't run live (STT alone exceeds the budget). It stays in the M4 grid as an accuracy ceiling.
- Projected end-of-speech → cue start = 600 ms silence hangover + STT + ~35 ms playback:
  **`base.en` ≈ 950 ms, `tiny.en` ≈ 780 ms.** The silence hangover, not the model, is the
  biggest cost. Live sessions will log each stage to confirm.
- Greedy decoding (beam 1) is ~5% faster than beam 5. Its accuracy impact is an M4 question.
- Whisper writes numbers as digits ("3 minute", "30 seconds") with this prompt. The parser handles
  both digits and words.

## 3. Microphone access (2026-11-21)

`spikes/mic_check.py`: 3 s recording at 16 kHz from the built-in mic.

- macOS **returns silence, not an error**, when mic permission is missing, and the first run blocks
  on a permission prompt. `stride run` must check for an all-zero signal and say how to fix it.
- Room floor: median −56.6 dBFS per 20 ms frame.
- The quietest frame was −173 dBFS (digital silence while the stream starts), so **noise-floor
  calibration discards the first ~200 ms** after opening the stream.
- Native rate is 48 kHz, and requesting 16 kHz works (CoreAudio resamples).

## 4. Rendering the cue catalog (2026-11-21)

The catalog is 315 clips: whole phrases for a normal session (rests on 15 s steps to 10:00, status
values, common adds, cues, replies) plus the fragments any other phrase is stitched from.

| Method | Time for 315 clips |
|---|---|
| One `say -o` per clip, 1 worker | ~1,317 ms/clip (≈ 7 min) |
| One `say -o` per clip, 4 / 8 workers | 642 / 571 ms/clip (≈ 3 min): the speech service serializes |
| **Batches: one `say` call per ~79 phrases, separated by `[[slnc 500]]`, split on silence, 4 in parallel** | **54.7 s** |

Every batch split into exactly one segment per phrase (checked on 4- and 8-way chunking). The code
still checks the count and falls back to per-phrase renders for a batch that doesn't match, so a
bad split can't attach the wrong audio to a phrase. Each phrase ends with a period so it keeps
sentence-final intonation, as a standalone render would.

One run took over 5 minutes (≈ 0.5× realtime) right after several minutes of continuous rendering
and killed `say` processes; a rerun on an idle machine took 54.7 s again. Likely thermal throttling
(fanless MacBook Air) or a wedged speech daemon. It's a first-run-only cost either way.
