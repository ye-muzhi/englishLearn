# AGENTS.md

Guidance for coding agents working in this repository.

## Run and test

```bash
./scripts/run_app.sh
.venv/bin/python -B -m pytest -q -p no:cacheprovider
```

On macOS, `EnglishLearn.command` is the user-facing double-click launcher.
Python dependencies live in `requirements.txt`; the isolated runtime is `.venv/`.

## Source layout

```text
app.py                         Streamlit entrypoint and UI composition
englishlearn/
  paths.py                     Canonical project/work paths
  i18n.py                      UI strings and target languages
  media/media_manager.py       Download, subtitle parsing, media/lookup server
  processing/asr_engine.py     faster-whisper wrapper
  processing/sentence_segmenter.py
  processing/ai_segmenter.py
  processing/pipeline_runner.py
  storage/project_store.py     Project/settings persistence
  storage/collections_store.py Favorites and wordbook persistence
  translation/llm_translator.py
  translation/hy_mt2_local.py  On-device Tencent Hy-MT2 runtime
scripts/                       Setup and CLI launchers
tests/                         Regression tests
work/                          User data and local model; never treat as source
```

Use package imports (`englishlearn...`), not root-level compatibility modules.
Resolve shared data paths through `englishlearn.paths`; honor
`ENGLISHLEARN_WORK_DIR` in tests and custom deployments.

## Processing pipeline

```text
Resolve/download source or accept local file
  -> use source captions when available
  -> otherwise extract audio and transcribe with word timestamps
  -> timing-aware sentence segmentation
  -> batched translation
  -> persist project-scoped raw/translated subtitles
  -> custom learning player
```

`englishlearn.processing.pipeline_runner` owns background task state so it
survives Streamlit reruns. Project-card progress uses fragments and must not
rerender the player every two seconds.

## Persistence

```text
work/
  projects.json
  model_presets.json
  user_settings.json
  favorites.json
  wordbook.json
  models/Hy-MT2-1.8B/
  projects/{project_id}/
    subtitles_raw.json
    subtitles_translated.json
```

Project writes must remain atomic. Never delete user-selected files outside
`work/`. Do not interrupt a running import merely to reload source code.

## UI rules

- Use `t(key, lang, **fmt)` for every user-visible string.
- Keep video-overlay and subtitle-list bilingual controls independent.
- Player settings belong beside player controls, not permanently above the list.
- Seeking must not autoplay, leave fullscreen, or accumulate active subtitle rows.
- A foreground interaction must either complete or present failure feedback within
  two seconds; long media/model processing is explicitly background work.

## Translation engines

The app supports OpenAI-compatible APIs, Ollama, and local Hy-MT2. Local
Hy-MT2 loading is lazy and must follow Tencent's prompt/inference contract.
Local endpoints bypass configured external proxies.

## Navigation

`st.navigation` exposes Home, Settings, and Collections pages. Selecting an
existing project loads its saved subtitles directly. Re-translate preserves raw
timings; re-process regenerates ASR/timing and resets the previous manual offset.
