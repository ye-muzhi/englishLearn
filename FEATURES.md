# EnglishLearn — Functional Spec

## Overview

Streamlit app that downloads/extracts video, transcribes with Whisper, segments with AI/rule-based, translates with LLM, and plays back with synced bilingual subtitles. Pipeline runs in a background thread so the UI stays responsive.

## Pages (st.navigation)

1. **Home** (`home_page`) — project library + main panel
2. **Settings** (`settings_page`) — full-width configuration
3. **Notes** (`notes_page`) — AI/manual learning notes + wordbook

---

## Home Page

### Persistent navigation rail: Project Library (`_render_project_list`)

- Global page navigation remains at the top of the rail; the current page uses
  a neutral filled state instead of a separate floating navigation block.
- The project list is compact and always available for fast switching. Its
  delete-all action remains beside the section heading.
- Project search lives in the main masthead and filters both this rail and the
  recent-project gallery by title.
- **Project cells** (one per existing project):
  - Status icon: ✅ completed, ⏳ processing, 📦 legacy, 🔄 active background task, ❓ unknown.
  - Creation date/time, current step, and progress bar are rendered inside the same card as its title.
  - Starting a new task leaves the main panel in its welcome state; progress is never duplicated in a detached main-panel status block.
  - Click loads the project into the main panel.
  - When 🔄 is shown, an inline progress bar + step label also render.
- **Deletion**:
  - Deleting one project permanently removes its subtitle files, app-managed
    downloaded/uploaded video and audio, project-scoped notes/wordbook,
    and its library record. A local source file outside `work/` is never
    deleted.
  - The project-library trash menu provides **Delete all learning data**. It
    requires an acknowledgement, is disabled while imports run, and removes
    all projects, managed media, subtitles, notes, and wordbook entries.
    Model downloads, presets, and app settings remain intact.

### Main workspace (`_render_home`)

The main workspace follows a media-site hierarchy without copying YouTube
branding: a sticky product masthead contains the global project search and the
single primary **New Task** entry. The viewing surface remains visually first;
secondary metadata and actions do not displace it.

The application uses a dark viewing theme by default: near-black canvas,
slightly elevated charcoal surfaces, muted gray secondary text, and violet
only for brand, focus, and selected states. Streamlit's native theme is also
dark so initial loading, menus, file upload, and form controls do not flash or
fall back to white.

Routes by session_state:
- `pipeline_error` set → red error banner, cleared after display.
- `current_project` set → `_render_project_detail`.
- `show_new_task` set → `_render_new_task_form`.
- Otherwise → `_render_welcome`.

### Welcome (`_render_welcome`)

Compact product proposition followed by a three-column, 16:9 recent-learning
gallery. Each card shows a cover, playback mode, title, status, and creation
time and opens the same project as its rail entry. A downloaded URL project
prefers the source website's thumbnail; an uploaded/local video or a failed
thumbnail request falls back to a representative frame near 12% of the video.
The cover is stored with the project so opening Home never decodes the video.

### New Task form (`_render_new_task_form`)

- A required, visible **Playback & processing mode** choice appears before the
  source field. **Local complete mode** is selected by default: it downloads
  the source video, can fall back to ASR, and opens with the app's native
  learning player. **Online captions mode** is an explicit opt-in: it does
  not download the video, requires an existing YouTube caption track, and
  keeps YouTube's compliant embedded playback surface.
- URL text input + file uploader. Uploaded files always use Local complete
  mode; selecting Online captions mode with a file is rejected before a task
  is created.
- **Use original target-language subtitles** is selected by default for URL
  sources. When the platform exposes the configured target track, the app
  preserves it, pairs it with the platform English track when available, and
  skips ASR, re-segmentation, and machine translation. If no target track is
  available, the normal caption/ASR and translation pipeline runs.
- "Process Video" button — starts background pipeline task (kind="new"), then `st.rerun()`.

### Project Detail (`_render_project_detail`)

- Player first, using the available main-workspace width.
- Title with target-language flag and source/timing metadata below the player.
- Rounded actions below the title: download plus a single **Manage project**
  popover containing rename, retranslate, reprocess, and confirmed delete.
- If a background task is active for this project → render `_render_processing_status` instead of the player.
- Else if project has subtitles → render player + table expander + download button.
- Else → "no subtitles, click Process Now" prompt.

### Application-shell acceptance

| ID | Scenario | Expected result |
|---|---|---|
| UI-01 | Open Home | Sticky masthead shows brand, project search, and one primary New Task action |
| UI-02 | Search for a project | Rail and recent-learning gallery show the same filtered project set |
| UI-03 | Open a recent project | The player appears before title, metadata, and secondary action chips |
| UI-04 | Scroll a watch page | Masthead remains available; Streamlit's empty fixed header never appears |
| UI-05 | Use rail controls | Navigation and project targets remain at least 40px high with visible selected state |
| UI-06 | Open project overflow | Destructive deletion remains confirmed and is not presented as a primary action |
| UI-07 | Narrow viewport | Content uses compact padding and the native Streamlit rail can collapse |
| UI-08 | Open any page or overlay | Canvas, rail, form, input, menu, and popover remain dark with readable contrast |
| UI-09 | Collapse and restore the application rail | One app-owned dark hamburger remains visible in both states; it collapses and restores the rail without depending on Streamlit's header, and the watch surface uses the released width |
| UI-10 | Open Manage project | Rename, retranslate, reprocess, and confirmed delete share one secondary menu |
| UI-11 | Complete a URL/local import and return Home | Its 16:9 project card shows the website cover or a generated video frame without delaying Home rendering |
| UI-12 | Click a recent-project cover or its title | Both targets open the same project; the cover has a visible hover/focus affordance |
| UI-13 | Leave the app idle with no background task | No two-second polling fragment is mounted and no deprecated layout warnings are emitted, so the player, scroll position, and controls do not reset periodically |

### Processing status panel (`_render_processing_status`)

- 🔄 or ❌ icon + "后台处理中" subheader.
- "Step X/4 — {label}" caption.
- Batch counter (current/total) when translating.
- Progress bar with step label as text.
- Elapsed time MM:SS.

### Polling fragment (`_poll_tasks_fragment`, `@st.fragment(run_every=2)`)

- Reads tasks for the session from `pipeline_runner.active_tasks`.
- If any task is pending → call `st.rerun()` every 2s (live progress).
- For tasks just completed: reload projects, set `pipeline_error` on failure, toast, refresh `current_project` subtitles if it matches, then `st.rerun()`.

---

## Background Pipeline (`pipeline_runner.py`)

Module-level state (`active_tasks`, `active_tasks_lock`) persists across Streamlit reruns because imported modules are cached.

### Task lifecycle

1. Button handler builds `params` dict.
2. `_start_pipeline_task(params)` registers task in `active_tasks[session_token][task_id]` and spawns daemon thread.
3. Thread runs `run_pipeline_thread(task, params, work_dir)`:
   - Step 1: resolve source (download via yt-dlp or copy upload).
   - Cover: save the platform thumbnail into the project directory; if it is
     missing or cannot be downloaded, extract a 640px-wide JPEG frame from the
     local video. Cover failure never fails the learning task.
   - Try `extract_subtitles` for YouTube/Bilibili URLs.
   - If the user prefers original target subtitles and that platform track is
     available, pair it with the platform English track by cue overlap, save
     both unchanged, and skip steps 2–4.
   - Step 2: extract audio (ffmpeg, 16kHz mono WAV) — skipped if audio exists.
   - Step 3: ASR via `ASREngine.transcribe` — skipped if existing subs were pulled.
   - Re-segment: AI (`ai_sentence_segment`) for instruction-following models;
     the subtitle-tuned rule segmenter (`sentence_segment`, 0.55s pause /
     18-word target with a four-word natural-sentence allowance) for MT models.
     Source cues are first split at internal sentence punctuation; unfinished
     clauses may then be joined across cue boundaries. Long sentences prefer
     comma/conjunction boundaries. Explicit completed sentences remain
     separate cues, including short speaker turns, so translation alignment is
     never obscured by display-only merging.
   - Timing alignment: faster-whisper word timestamps and YouTube JSON3
     `tOffsetMs` offsets are retained through segmentation. Sentence/length
     splits use the first and last spoken-word boundaries instead of dividing a
     source cue by word count. Long silent gaps are not assigned to the
     preceding word. Word timing stays in `subtitles_raw.json` for future
     retranslation but is removed from player-facing translated JSON.
   - Step 4: translate via `translate_subtitles` (batched, concurrent).
   - Save: update or create project on disk.
4. Thread writes status fields (`step`, `step_label`, `progress`, `batch_current`, `batch_total`, `completed`, `consumed`, `error`, `result_project_id`).
5. On exception: `task["error"]` set, `completed=True`, `consumed=False`.

### Task kinds

- `new` — URL or upload, full pipeline.
- `reprocess` — existing project, re-runs ASR + translation from source video.
- `retranslate` — existing project, re-runs segmentation + translation only (uses raw subs).

---

## Settings Page (`settings_page`)

Sections:
1. **UI Language** — English / 中文 selectbox.
2. **ASR** — Whisper model selectbox.
3. **Translation** — Target language, model preset selectbox, model override input + reset.
4. **Concurrency** — `max_workers` slider (1–16, default 6).
5. **Preset editor** — cards with Use / Edit / Save / Delete per preset.
6. **Network** — proxy text input (auto-detected from env).
7. **Actions** — Clear & Reset button.

Credential preflight:

- OpenAI presets require a non-empty `api_key`, `OPENAI_API_KEY`, or
  `OPENAI_ADMIN_KEY` before a task is queued.
- Ollama presets default to the local SDK placeholder key `ollama` when their
  endpoint is local and the key field is blank.
- The on-device Hy-MT2 preset is validated by its local model readiness check.
- A missing cloud credential is shown before downloading or transcribing, with
  a direct instruction to add the key or switch to local Hy-MT2.

### Hy-MT2 local inference contract

- The on-device adapter uses the official Hy-MT2 default Chinese instruction,
  with no system prompt, and wraps it with
  `tokenizer.apply_chat_template(..., add_generation_prompt=True)`.
- The generated suffix is decoded only (the prompt tokens are never shown as a
  translation).
- The official 1.8B/7B sampling values are pinned: temperature 0.7, top-p
  0.6, top-k 20, repetition penalty 1.05, and a 4096-token ceiling. For
  subtitle-sized cues the ceiling is reduced dynamically to avoid wasting
  decode steps on short lines.
- Duplicate subtitle cues use a bounded process-local cache; timing and output
  alignment remain one cue per result.

---

## Notes Page (`notes_page`)

`st.tabs(["📝 All notes", "📖 Wordbook"])`:

- **Notes tab**: searchable learning-note cards with title, source subtitle,
  project/timestamp, tags, editable body, delete, and "open in player".
  A manual-note form supports both standalone notes and notes linked to an
  existing project.
- **Wordbook tab**: search box, count, list of word cards (word, POS, translation, pronunciation, other_meanings, context, source project, delete, "open in player").

"Open in player" loads the source project and seeks to the saved timestamp.

### Contextual note workflow

1. The player pencil or a subtitle-row pencil pauses playback and captures the
   current timestamp plus the active source/translated sentence.
2. An inline composer appears below that same project. The learner may record
   an idea, type it, or combine both.
3. Voice is transcribed locally with the selected Whisper model. Transcription
   and AI refinement run in a persisted background task, so Streamlit reruns do
   not block playback or lose progress.
4. An instruction-capable OpenAI/Ollama preset returns an editable title, note,
   summary, key points, vocabulary, and tags. Hy-MT2 is translation-only, so it
   preserves the transcript as an honest editable draft instead of fabricating
   an AI-refined result.
5. The learner reviews and edits the result before saving. Temporary recording
   files are removed when processing is consumed; API credentials are never
   persisted in the task file.
6. Existing sentence favorites are copied into Notes exactly once on upgrade.
   The legacy file is retained for rollback compatibility, so no historical
   learning data becomes inaccessible when the old Collections page disappears.

---

## Player (HTML iframe)

Bilibili-style custom player:
- 58%/42% true split between video and right panel; the list consumes its own column and never covers the video.
- Subtitle overlay at bottom (max-width 96%).
- Right panel tabs: Subtitles / Notes / Wordbook.
- The subtitle header keeps high-frequency follow/bilingual controls visible.
  Search and list text size are progressively disclosed behind one tool button;
  Ctrl/Cmd+F focuses incremental source/translation search and Escape clears it.
- Click subtitle row → seek without starting playback; the learner controls play explicitly.
- ✎ button per subtitle → pause and open a note composer with that sentence and timestamp.
- A player-level ✎ control captures the currently active subtitle, so a note
  can also begin from the learner's exact playback position.
- Text selection remains native so a learner can select an entire sentence without a popup interrupting it.
- Hold ArrowRight → 3× speed after 400ms.
- Video subtitle visibility, video bilingual display, and subtitle-list
  bilingual display are three independent controls. Turning off a video
  translation never changes the list, and turning off a list translation never
  changes the video overlay; the original line remains available in both.
- Video-overlay and list text sizes are adjusted independently and persist for
  the next project open. A dedicated overlay grip lets the learner drag video
  subtitles to a readable vertical position; its position also persists.
- **Subtitle timing** lives in the lower-right player controls next to the
  progress/timeline area, not in the subtitle list. It has a project-specific
  slider from −5.0s to +5.0s, plus −0.1s, +0.1s, and Reset controls. Negative
  values advance subtitles; positive values delay them. It updates the overlay
  and active subtitle row immediately without moving playback, and clicking a
  subtitle/note/word context seeks to its corrected video time. The
  selected value is saved to the project and restored when it is reopened.
- Hovering a source-language word for 200ms highlights it only. Clicking the
  word opens a compact dictionary card. It checks the local wordbook/cache
  first, then queries the online dictionary; it never silently substitutes a
  machine translation for a dictionary result.
- The card exposes an explicit **Context translation** action. It uses the
  free translation endpoint through a local, CORS-safe bridge inside the
  player iframe, so it never reloads Streamlit or exits immersive fullscreen.
  It does not add a word to the wordbook.
- Dictionary lookup queries the dictionary and free translation fallback in
  parallel through the same local bridge within a 2-second budget. The
  dictionary definition remains the primary result; a free translation is
  shown alongside it or by itself when no dictionary entry exists. Successful
  results are cached locally; failed attempts are not cached and can be
  retried immediately.
- Subtitle text size is adjustable with a draggable range control and persists for
  the next project open.
- Playback speed is adjustable from 0.5× to 2× for local and online playback;
  online playback keeps YouTube's timeline as the only progress bar.
- Interactive lookup/save actions have a hard 2-second budget. If the action has
  not completed by then, it enters an explicit failure state, unlocks the
  control, and tells the learner to retry.

### Immersive player controls (acceptance baseline)

The learning player owns three layout states and must behave identically for
local video and YouTube online playback:

1. **Overlay view (default)** — video always uses the full learning viewport;
   the subtitle/notes/wordbook panel occupies its own resizable right column.
2. **Video focus** — dragging the subtitle separator to the outer edge hides
   the panel; a narrow edge restore control remains visible.
3. **Immersive fullscreen** — the complete learning shell (video plus the right
   panel when open) enters browser fullscreen. If an embedded browser rejects
   the native Fullscreen API, the host iframe expands to a viewport-covering
   immersive fallback. Exiting restores the previous split/collapsed state.
4. **Resizable split** — whenever the right panel is open, a visible separator
   between video and subtitles can be dragged to change the subtitle-panel
   share, including in the normal embedded layout and fullscreen.

Fullscreen resize rules:

- Desktop subtitle width is constrained to 24–55% of the learning viewport;
  dragging below the collapse threshold hides it instead of leaving a sliver.
- Narrow/mobile fullscreen uses a horizontal separator and constrains subtitle
  height to 28–65%.
- Arrow keys on the focused separator adjust the split in 2% increments;
  Home/End jump to the minimum/maximum and double-click restores the default.
- The last desktop/mobile proportions are stored locally and restored the next
  time immersive fullscreen opens.
- Collapsing and restoring the subtitle panel preserves its previous ratio.

Subtitle following rules:

- Playback may highlight the active subtitle and center it only inside the
  subtitle scroll container.
- Wheel, touch, pointer/scrollbar, or subtitle-list keyboard navigation pauses
  automatic following immediately, so reading older/newer lines never moves
  the page back unexpectedly.
- A visible "Resume follow" control restores automatic following and centers
  the current subtitle.
- Clicking a subtitle explicitly restores following, seeks to its timestamp,
  and keeps the existing click-to-seek behavior.
- Seeking, changing subtitle timing, or resuming follow clears the previous
  active-row state before resolving the new cue; at most one subtitle row may
  be highlighted at any time.

Acceptance checklist:

| ID | Scenario | Expected result |
|---|---|---|
| P-01 | Enter immersive fullscreen | Video and right panel stay together in one fullscreen learning view |
| P-02 | Exit fullscreen | Previous panel state and playback position are preserved |
| P-03 | Drag right panel to the outer edge | Panel disappears; the full-width video remains and an edge restore control appears |
| P-04 | Restore right panel | Panel returns without reloading or resetting playback |
| P-05 | Manually scroll subtitles during playback | Automatic following pauses and neither the iframe nor outer page jumps |
| P-06 | Press Resume follow | Current subtitle is centered inside the subtitle container only |
| P-07 | Click a subtitle while follow is paused | Player seeks and automatic following resumes |
| P-08 | Local/online parity | All layout controls work for both sources; each source keeps its correct playback controls |
| P-09 | Drag separator in normal/fullscreen view | The real video/list column split changes continuously; neither surface overlaps the other |
| P-10 | Drag beyond limits | Subtitle panel stops at its documented minimum/maximum |
| P-11 | Exit and re-enter fullscreen | Last chosen proportion is restored |
| P-12 | Keyboard-adjust separator | Arrow/Home/End update the proportion and accessible value |
| P-13 | Collapse after resizing | Restore returns the panel at its previous proportion |
| P-14 | Toggle video bilingual subtitles | Only the video-overlay translation line hides/shows; the subtitle list stays unchanged |
| P-15 | Toggle subtitle-list bilingual display | Only the list translation line hides/shows; the video overlay stays unchanged |
| P-16 | Click subtitle row while paused | Player seeks to the row timestamp and remains paused |
| P-20 | Adjust subtitle timing | Overlay and active row shift immediately; video playhead does not move |
| P-21 | Reopen a project after timing adjustment | Its saved −5.0s to +5.0s timing offset is restored |
| P-22 | Click subtitle/note/word context after timing adjustment | Seek lands on the corrected subtitle position |
| P-23 | Delete one project | Project subtitles, managed media, words, and notes are removed; external local source remains |
| P-24 | Delete all learning data | Acknowledgement is required; all project/media/collection data clears while settings/models remain |
| P-25 | Attempt delete all during import | Action is disabled and explains that the active task must finish |
| P-17 | Click a word with a dictionary entry | Dictionary definition is shown with its free translation, without blocking the player |
| P-18 | Click a word without a dictionary entry | A free translation is shown when available; otherwise the card gives a retry message |
| P-19 | Click Context translation in immersive fullscreen | Translation appears in the same popup; fullscreen and playback position remain unchanged |
| P-16 | Hover a source word in list for 200ms | Only that word is highlighted; no card interrupts selection |
| P-17 | Click a highlighted source word in list/overlay | A button-free dictionary card appears near the word and stays inside player viewport |
| P-18 | Drag subtitle size control | List and overlay font sizes change continuously and the value persists |
| P-19 | Adjust playback speed | Local and online adapters receive the selected rate without a duplicate timeline |
| P-20 | Click subtitle pencil | Playback pauses and a project-scoped note composer opens with the sentence and timestamp |
| P-21 | Dictionary lookup takes longer than 2 seconds | Lookup is aborted and the card shows a retry hint |
| P-22 | Open a new project form | "Local complete mode (default)" is visibly selected, with its download/ASR/native-player consequence stated |
| P-23 | Select online captions mode and submit a valid YouTube link | The saved project is `playback_mode=online`, has no downloaded video path, and opens with the YouTube embed |
| P-24 | Leave the default mode and submit a source | The task is created as `playback_mode=local`; completed playback uses the app's native learning player |
| P-25 | Adjust video/list font sliders independently | Video size is adjusted from the player’s subtitle-settings menu; list size remains in the list and both values restore after reopening |
| P-26 | Drag either video subtitle grip | Source and translated overlays move independently in both axes and restore their separate positions after reopening |
| P-27 | Use a subtitle-grip Arrow/Home/End keys | The selected overlay moves in bounded increments without changing playback or list scroll position |
| P-28 | Turn off translated video subtitles | Source subtitles continue at their saved position; the subtitle-list bilingual setting is unchanged |
| P-29 | Turn off source video subtitles | Translated subtitles continue at their saved position; the master CC switch still controls both layers |
| P-30 | Open subtitle settings beside CC | Source/translated switches, video size, and timing are grouped in a temporary player menu; no timing bar remains over the video or in the reading panel |
| P-31 | Open an online project’s subtitle settings | The floating player gear exposes the same menu without adding a duplicate playback timeline |
| P-32 | Turn off Click word to look up | Word hover/click lookup is disabled in both video subtitles and the subtitle list; clicking a list word follows the normal row-seek behavior |
| P-33 | Reopen a project after disabling word lookup | The player restores the disabled preference until the user enables it again from subtitle settings |
| P-34 | Inspect subtitle-list header | Only Follow and List bilingual remain; fullscreen and collapse buttons are absent |
| P-35 | Create with original target subtitles enabled and target track present | Platform source/target captions are preserved and no ASR, segmentation, or translation is invoked |
| P-36 | Create with original target subtitles enabled but target track absent | The normal existing-caption/ASR plus translation pipeline runs automatically |
| P-34 | Play, seek across several subtitle rows, then adjust subtitle timing | Every refresh removes stale active states; the subtitle list contains zero or one highlighted row, never several |

### Player action dispatch (`_handle_player_actions`)

Reads URL query params sent from iframe:
- `action=translate_word&word=...&context=...&time=...&subtitle_id=...&project_id=...`
- `action=open_note&text=...&translation=...&time=...&subtitle_id=...&project_id=...`
- `action=delete_word&id=...`
- Legacy `toggle_favorite` / `delete_favorite` URLs remain readable for data
  compatibility, but no current page exposes a sentence-favorite control.
- `action=seek&time=...`

Processed before render, then params cleared and `st.rerun()`.

### Dictionary, wordbook, and note workflow

Selecting a sentence does not open the dictionary; clicking one word does. The
compact dictionary card contains a star for adding/removing the current word
from the Wordbook, available phonetics and pronunciation buttons, Chinese
translations, and multiple English definitions. Sentence-level learning ideas
use the subtitle pencil and are stored as editable notes instead of favorites.

Wordbook writes use the loopback player API and update the iframe state
directly. Because Streamlit isolates the player in an opaque-origin iframe,
writes use a one-pixel media receipt rather than a cross-origin fetch. Opening
a note intentionally pauses and refreshes the host once to mount Streamlit's
native recorder; the source project, timestamp, and draft are restored.

Lookup order:

1. Existing Wordbook entry for immediate display.
2. Optional local ECDICT (`work/dictionaries/ecdict.mini.csv`,
   `work/dictionaries/ecdict.db`, or `ENGLISHLEARN_ECDICT_PATH`) for fast
   phonetics and multiple Chinese meanings.
3. Free Dictionary API for multiple parts of speech, definitions, examples,
   phonetics, and pronunciation audio.
4. Free translation fallback, with the explicit context-translation action
   available when the dictionary result is insufficient.

If a dictionary entry has phonetics but no recording, the pronunciation button
falls back to the device's English speech synthesizer. It therefore remains
playable instead of silently disappearing.

#### Optional local dictionary

Oxford dictionary data is not bundled or downloaded: its offline/cache use is
subject to Oxford's commercial/Enterprise licensing. The supported offline
alternative is the MIT-licensed ECDICT dataset. The user can install its compact
CSV explicitly:

```bash
mkdir -p work/dictionaries
curl -L https://raw.githubusercontent.com/skywind3000/ECDICT/master/ecdict.mini.csv \
  -o work/dictionaries/ecdict.mini.csv
```

The application also accepts an ECDICT SQLite database at
`work/dictionaries/ecdict.db` or a custom file selected with
`ENGLISHLEARN_ECDICT_PATH`. No dictionary download happens automatically.

Action guarantees:

- Dictionary card positioning uses its measured dimensions and stays inside the
  player viewport.
- Legacy URL-dispatched collection actions remain supported, but current
  player collection writes use bounded local media-receipt requests.
- Empty/failed translations are not written as successful wordbook entries.
- Adding a word or note preserves the source timestamp.
- Duplicate word actions update the existing matching entry.
- Wordbook and note entries from online projects can reopen the online
  player at the stored timestamp without requiring a local video file.

Acceptance checklist:

| ID | Scenario | Expected result |
|---|---|---|
| C-01 | Select subtitle text | Native sentence selection remains available; no dictionary card appears |
| C-02 | Click one source word | Dictionary card shows that word, available phonetics/audio, translations, multiple meanings, and a Wordbook star |
| C-03 | Click a saved word | Saved data appears immediately and its star is active |
| C-04 | Dictionary request times out | Card shows an actionable retry hint and remains non-blocking |
| C-05 | Click subtitle pencil | Video pauses and the composer opens with the correct subtitle and timestamp |
| C-06 | Open online collection entry | Online project loads and seeks without a local video path |
| C-07 | Click the dictionary star | Word, meanings, phonetic, audio URL, context, and timestamp are saved without leaving the player |
| C-08 | Play a pronunciation | Available UK/US or generic dictionary audio plays and the player/fullscreen state is unchanged |
| C-09 | Install ECDICT mini | Subsequent matching lookups use its multiple Chinese meanings locally before online fallbacks |
| N-01 | Click the player pencil between subtitle cues | Video pauses and the composer opens at the exact playback time with empty context allowed |
| N-02 | Record an idea and submit | Transcription/refinement runs in the background and visibly reports its stage |
| N-03 | Selected model cannot follow note instructions | An editable transcript draft appears with a clear warning; no content is lost |
| N-04 | Edit the AI draft and save | The edited note appears in both the player Notes tab and global Notes page |
| N-05 | Switch projects while a note is processing | The draft is never displayed or saved under the wrong project; returning restores its progress |
| N-06 | Delete a project/delete all data | Its notes/all notes are removed with the same confirmation rules as other learning data |

---

## Persistence (`work/` directory)

```
work/
  projects.json              # project index
  model_presets.json         # LLM presets
  wordbook.json              # global wordbook entries
  notes.json                 # global project-linked and standalone notes
  note_tasks.json            # persisted public state for active note jobs
  dictionaries/
    ecdict.mini.csv          # optional MIT-licensed offline dictionary
    ecdict.db                # optional SQLite ECDICT alternative
  projects/{project_id}/
    note_audio/              # temporary private recordings, removed after processing
    subtitles_raw.json       # pre-translation segments
    subtitles_translated.json
```

Legacy migration: on first load, `work/` is scanned for `.mp4`+`.wav` pairs and project records created with `status="legacy"`.

---

## i18n

All UI strings in `englishlearn/i18n.py:STRINGS` keyed by message_id with `en`/`zh` variants. `t(key, lang, **fmt)` helper. Default UI language is `zh`.

---

## Packaging and launch

Application modules live under the `englishlearn` package and are grouped by
responsibility. `app.py` is the only root Python entry point. User data stays
under `work/` and is never coupled to the source-package layout.

On macOS, `EnglishLearn.command` is the double-click entry point. The shared
launcher in `scripts/run_app.sh` resolves the project directory, checks the
isolated runtime, reuses an already healthy instance, waits for Streamlit's
health endpoint, and opens the app only after startup succeeds.

Acceptance checklist:

| ID | Scenario | Expected result |
|---|---|---|
| L-01 | Double-click `EnglishLearn.command` from Finder | The app starts from the project directory and opens on port 8501 |
| L-02 | Launch while the app is already healthy | The existing instance is reused instead of creating a second server |
| L-03 | Launch without `.venv` | The terminal shows the exact setup command and exits with an error |
| L-04 | Streamlit does not become healthy in 8 seconds | The launcher reports failure and stops the child process |
| L-05 | Import any application module | Imports resolve through `englishlearn.*`; no legacy root module is required |

---

## Concurrency & Performance

- Translation: `BATCH_SIZE=30` (JSON LLM), `LINE_BATCH_SIZE=10` (MT), default 6 workers.
- Segmentation: `SEGMENT_BATCH_SIZE=80`, default 5 workers.
- Retry only when >30% of segments lack ending punctuation.
- OpenAI client shared across batches per pipeline run.
- httpx.Client with proxy bypass for local IPs.

### Segmentation acceptance

| ID | Scenario | Expected |
|---|---|---|
| S-01 | One source cue contains several sentences | Split at `.`, `?`, `!` or equivalent punctuation, with continuous proportional timing |
| S-02 | A sentence spans several source cues | Join only the unfinished clause and retain the natural final punctuation |
| S-03 | A completed sentence is slightly over 18 words | Keep it intact up to 22 words instead of cutting at an arbitrary source boundary |
| S-04 | A sentence remains too long | Split near a comma/conjunction and keep every cue at or below the readable ceiling |
| S-05 | Several one-to-three-word sentences occur consecutively | Group them into a short punctuated cue instead of flashing one word at a time |
| S-06 | Source text contains line breaks or non-breaking spaces | Normalize them before segmentation so the player receives clean single-line text |
| S-07 | Whisper returns word timestamps | Final cues begin/end on the actual first/last spoken word rather than the parent segment ratio |
| S-08 | JSON3 has `tOffsetMs` values separated by a long pause | The preceding cue ends after its spoken word and does not remain visible throughout the silence |
| S-09 | JSON3 lacks word offsets | Generate monotonic proportional word timing as a bounded fallback |
| S-10 | SRT cues overlap | End the earlier cue at the next cue start so the player has one deterministic active row |
| S-11 | Re-process an existing project | Reset the previous manual subtitle offset to `0.0s` and mark the project as word- or cue-aligned |
