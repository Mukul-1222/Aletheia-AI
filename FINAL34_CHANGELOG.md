# Aletheia FINAL34 — Changelog

**Release:** `0.34.0`

## Added

- Local-only FastAPI web application and a responsive single-page browser interface.
- Multiple durable chat sessions backed by SQLite, feedback records, current-conversation export, and full conversation export.
- Retrieval of conversation experience across chat sessions, while keeping each model response within a finite context window.
- Reactive companion orb states for listening, thinking, speaking, learning, and visual observation.
- Global light/dark theme and responsive desktop/mobile navigation.
- Drag-and-drop and picker-based document/text submission to FINAL33's durable learning queue, queued-job progress, report access, retry controls where eligible, and auto-processing preference.
- Explicit, permission-controlled webcam preview/frame observation and short audio-observation samples.
- Browser-provided speech recognition and speech synthesis where available, plus an optional learned-TTS path when an eligible Aletheia voice model exists.
- Consent-gated voice sample enrollment, speaker selection, memory search, and configuration/preferences.
- Backend validation, same-origin/local-host checks, CSP and other response headers, lazy agent initialization, and serialized access between model operations and training jobs.
- Regression coverage for persistence, media payload validation, local-only checks, lazy initialization, voice consent, and the learning UI integration.

## Changed

- Release/package metadata aligned to `0.34.0`.
- Package metadata includes web UI assets and an optional `webapp` dependency extra.
- Current-release tests and test runner labels now identify FINAL34.

## Deliberate non-goals / not complete

- No pretrained neural model or hosted language-model API is bundled as the core conversation model.
- Voice sample enrollment is not voice cloning. Learned TTS and from-scratch speech recognition require their own trained data/weights and quality evaluation.
- Browser speech recognition may use browser-vendor services; it is not guaranteed to be on-device or private to the laptop.
- Camera and microphone are never silently enabled; browser permissions and a user action are required.
- General image generation, broad visual captioning, an ad-free mini-browser, public API catalogue connectors, other-agent integrations, and research-grade self-evolution remain future work.
