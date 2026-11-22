# Decision log

Design decisions and the evidence behind them, newest last. When a measurement changes a decision,
the old entry stays and a new one says what changed and why.

Milestones referenced below: **M1** timer product · **M2** workout plans · **M3** noise-study data ·
**M4** benchmark · **M5** Whisper fine-tuning · **M6** real-world validation · **M7** earbud trigger.

| Date | Decision | Why |
|---|---|---|
| 2026-11-21 | Python, on-device faster-whisper, rule-based intent parser | Instant, offline and private. The command set is small and closed, so a deterministic parser is predictable and fully testable. |
| 2026-11-21 | No LLM agent or hosted LLM API | A closed command set doesn't need one, and a network round trip breaks the latency and offline goals. The hard problem is recognizing speech in noise, so that's where the ML work goes. |
| 2026-11-21 | Noise study: own recordings + public noise, mixed at controlled SNRs | Feasible solo and a standard method. Labels carry over from the clean clips, so thousands of test clips come labeled for free. |
| 2026-11-21 | Whisper LoRA fine-tuning (M5) after the benchmark; custom wake word is optional | Completes the measure → diagnose → train → re-measure loop. |
| 2026-11-21 | A fine-tuned model must beat vocabulary prompting + fuzzy matching on held-out speakers, noise and phrasings | Beating raw Whisper alone would overstate the gain. Held-out sets catch memorization. |
| 2026-11-21 | src layout (`src/stride/`), hatchling; CI on Python 3.11 + 3.12 × Ubuntu + macOS | Tests run against the installed package, not the working directory. 3.11 is the floor (`tomllib`). |
| 2026-11-21 | Config defaults live in code; `config.toml` is a partial override; unknown keys are errors; a test pins `config.toml` to the defaults | A typo silently ignored mid-workout is the worst failure mode, and the documented file can't drift from the code. |
| 2026-11-21 | mypy `--strict` from the first commit | Cheap to keep strict from the start, expensive to retrofit. |
| 2026-11-21 | Live-audio dependencies (faster-whisper, sounddevice, numpy) are an optional `[audio]` extra | Core logic and CI stay lightweight. Pure-logic tests never import audio libraries. |
| 2026-11-21 | `small.en` excluded from live use; kept in the M4 grid as an accuracy ceiling | Measured 1,086 ms median STT on the target laptop, above the 1.0 s end-to-end budget on its own. `tiny.en` 146 ms, `base.en` 316 ms. [spikes §2](spikes.md#2-faster-whisper-latency-2026-11-21) |
| 2026-11-21 | Latency is logged per stage (silence hangover, STT, cue start); the end-of-speech detector and model are chosen in M4 with data | The 600 ms silence hangover is the biggest single cost. `base.en` projects to ~950 ms against the 1.0 s target, too close to call without real measurements. |
| 2026-11-21 | Mic calibration discards the first ~200 ms of a stream; an all-zero signal is reported as missing mic permission | Startup frames measured at −173 dBFS (digital silence), and macOS denies mic access silently. [spikes §3](spikes.md#3-microphone-access-2026-11-21) |
| 2026-11-21 | Dev venv lives at `.venv.nosync` (symlinked as `.venv`) | iCloud Drive re-applies the `hidden` flag to the editable-install `.pth` file, which Python 3.12+ skips, breaking `import stride`. iCloud ignores `.nosync` names. |
