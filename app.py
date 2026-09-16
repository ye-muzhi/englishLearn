"""
Video Subtitle Translator — MVP

Streamlit app that downloads/extracts video, transcribes with Whisper,
translates with LLM, and plays back with synced bilingual subtitles.
"""
import html as html_lib
import base64
import json
import logging
import mimetypes
import os
import re
import threading
import time
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

# `transformers` exposes optional vision modules lazily.  If an external
# Streamlit session enables its source watcher, probing those modules can emit
# hundreds of harmless missing-torchvision warnings even though this app uses
# only text translation.  Keep the optional-import noise out of the terminal.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

from englishlearn.i18n import t, TARGET_LANGUAGES, UI_LANGUAGES, LANG_FLAGS
from englishlearn.media.media_manager import (
    resolve_video_source, extract_subtitles, extract_audio,
    start_video_server, start_lookup_server, normalize_video_path, youtube_video_id,
)
from englishlearn.processing.asr_engine import ASREngine, MODEL_SIZES
from englishlearn.translation.llm_translator import translate_subtitles, translate_word, BATCH_SIZE, LINE_BATCH_SIZE, _detect_mode
from englishlearn.processing.sentence_segmenter import sentence_segment
from englishlearn.processing.ai_segmenter import ai_sentence_segment, can_use_ai_segment
from englishlearn.storage.project_store import (
    load_presets, save_presets,
    load_user_settings, save_user_settings,
    load_projects, save_projects,
    create_project, append_project, update_project, save_subtitles_to_project,
    load_project_subtitles, load_project_raw_subtitles,
    save_translated_subtitles,
    update_project_title, delete_project, delete_all_project_data,
    has_subtitles,
)
from englishlearn.storage.collections_store import CollectionStore
from englishlearn.storage.subtitle_editor import (
    history_status as subtitle_history_status,
    merge_with_next as merge_subtitle_with_next,
    redo as redo_subtitle_edit,
    split_cue as split_subtitle_cue,
    undo as undo_subtitle_edit,
    update_cue as update_subtitle_cue,
)
from englishlearn.paths import work_dir
from englishlearn.processing import pipeline_runner
from englishlearn.notes import note_agent, migrate_legacy_favorites

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EnglishLearn",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="auto",
)

WORK_DIR = str(work_dir())

wordbook_store = CollectionStore(os.path.join(WORK_DIR, "wordbook.json"))
favorites_store = CollectionStore(os.path.join(WORK_DIR, "favorites.json"))
notes_store = CollectionStore(os.path.join(WORK_DIR, "notes.json"))
migrate_legacy_favorites(favorites_store, notes_store)


def _apply_product_theme() -> None:
    """YouTube-inspired viewing hierarchy with an independent product identity."""
    st.markdown(
        """
        <style>
        :root {
            --brand: #8b7cf6; --brand-hover: #9d91f8; --brand-soft: rgba(139,124,246,.18);
            --ink: #f1f1f1; --muted: #aaa; --line: #303030;
            --surface: #272727; --surface-hover: #3f3f3f; --panel: #181818; --canvas: #0f0f0f;
        }
        html, body, [class*="css"] { color-scheme: dark !important; }
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background: var(--canvas); color: var(--ink);
            font-family: Roboto, Arial, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        /* Streamlit's native rail controls remain in the DOM as the state
           bridge, but the product-owned hamburger below is the visible UI. */
        [data-testid="stHeader"] {
            display: block; height: 0; min-height: 0; background: transparent;
            pointer-events: none; overflow: visible;
        }
        [data-testid="stHeader"] [data-testid="stToolbar"],
        [data-testid="stHeader"] [data-testid="stDecoration"] { display:none; }
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stExpandSidebarButton"] {
            opacity:0 !important; pointer-events:none !important;
        }
        .block-container {
            max-width: 1800px; padding: .85rem 1.5rem 3rem;
        }
        [data-testid="stSidebar"] {
            background: var(--canvas); border-right: 1px solid var(--line);
            min-width: 260px !important; width: 260px !important;
        }
        [data-testid="stSidebarContent"] { width:260px !important; padding: .2rem .75rem 1rem; }
        [data-testid="stSidebarNav"] { padding: .35rem 0 .6rem; border-bottom: 1px solid var(--line); }
        [data-testid="stSidebarNavItems"] { gap: .15rem; }
        [data-testid="stSidebarNavLink"] {
            min-height: 42px; padding: 0 .8rem; border-radius: 10px;
            color: var(--ink); font-weight: 560;
        }
        [data-testid="stSidebarNavLink"]:hover { background: var(--surface); }
        [data-testid="stSidebarNavLink"][aria-current="page"] {
            background: var(--surface); font-weight: 700;
        }
        [data-testid="stSidebarUserContent"] { padding-top: .45rem; }
        [data-testid="stSidebar"] hr { margin: .55rem 0 .7rem; }
        [data-testid="stSidebar"] h3 { font-size: 1rem; margin: 0; }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [class*="st-key-load_"]) {
            border: 0 !important; border-radius: 10px !important; padding: .28rem .35rem !important;
            gap: .1rem !important; background: transparent; transition: background .15s ease;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(> [class*="st-key-load_"]):hover {
            background: var(--surface);
        }
        [data-testid="stSidebar"] [class*="st-key-load_"] button {
            min-height: 38px !important; padding: .35rem .55rem !important;
            border: 0 !important; border-radius: 8px !important; box-shadow: none !important;
            text-align: left; justify-content: flex-start; line-height: 1.25;
        }
        [data-testid="stSidebar"] [class*="st-key-load_"] button[kind="secondary"] { background: transparent; }
        [data-testid="stSidebar"] [class*="st-key-load_"] button[kind="primary"] {
            background: var(--brand-soft); color: #d9d5ff;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--muted); font-size: .72rem; padding: 0 .55rem .2rem;
        }
        h1 { letter-spacing: -.035em; font-weight: 750; font-size: 2rem !important; }
        h2, h3 { letter-spacing: -.018em; }
        .stButton > button, .stDownloadButton > button, [data-testid="stPopoverButton"] {
            border-radius: 999px; font-weight: 650; min-height: 40px; border: 0;
            box-shadow: none; padding-left: 1rem; padding-right: 1rem;
        }
        .stButton > button[kind="primary"] {
            background: var(--brand); color: #fff;
        }
        .stButton > button[kind="primary"]:hover { background: var(--brand-hover); }
        .stButton > button[kind="secondary"], .stDownloadButton > button,
        [data-testid="stPopoverButton"] { background: var(--surface); color: var(--ink); }
        .stButton > button[kind="secondary"]:hover, .stDownloadButton > button:hover,
        [data-testid="stPopoverButton"]:hover { background: var(--surface-hover); color: var(--ink); }
        [data-testid="stMetric"] {
            background: var(--surface); color: var(--ink); border: 0;
            border-radius: 14px; padding: .8rem 1rem;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: var(--ink); }
        [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
            background: #121212 !important; color: var(--ink) !important;
        }
        [data-testid="stTextInputRootElement"] {
            border-radius: 999px !important; background:#121212 !important;
            border-color:var(--line) !important;
        }
        [data-testid="stTextInputRootElement"] input,
        [data-baseweb="textarea"] textarea { color:var(--ink) !important; caret-color:var(--ink); }
        [data-testid="stTextInputRootElement"] input::placeholder,
        [data-baseweb="textarea"] textarea::placeholder { color:#777 !important; opacity:1; }
        [data-testid="stExpander"], [data-testid="stForm"] {
            background: var(--panel); border: 1px solid var(--line); border-radius: 16px;
        }
        [data-testid="stFileUploaderDropzone"] {
            background:#121212; border-color:var(--line); color:var(--ink);
        }
        [data-baseweb="popover"], [data-baseweb="menu"], [role="dialog"] {
            background:var(--panel) !important; color:var(--ink) !important;
        }
        [data-testid="stAlert"] { border-color:var(--line); }
        [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] { color:var(--muted); }
        code { background:#222 !important; color:#d7d4ff !important; }
        iframe[title="st.iframe"] {
            border-radius: 14px; overflow: hidden; background: #000;
            box-shadow: 0 1px 2px rgba(0,0,0,.12);
        }
        .product-eyebrow { color: var(--muted); font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
        .app-masthead {
            display:flex; align-items:center; gap:.65rem; min-height:42px;
            font-weight:800; font-size:1.15rem; letter-spacing:-.025em;
        }
        [data-testid="stLayoutWrapper"]:has(> [class*="st-key-app_masthead"]) {
            position: sticky; top: 0; z-index: 40;
            background: rgba(15,15,15,.94); border-bottom: 1px solid rgba(48,48,48,.9);
            backdrop-filter: blur(12px);
        }
        [class*="st-key-app_masthead"] { padding: .25rem 0 .65rem; }
        .app-mark {
            width:34px; height:25px; border-radius:8px; background:var(--brand); color:#fff;
            display:inline-flex; align-items:center; justify-content:center; font-size:.78rem;
            box-shadow:0 4px 12px rgba(101,88,232,.2);
        }
        .home-intro { max-width: 700px; padding: 2.1rem 0 1rem; }
        .home-intro h1 { margin:.25rem 0 .45rem; font-size:2.45rem !important; line-height:1.08; }
        .home-intro p { color:var(--muted); font-size:1.02rem; margin:0; }
        .library-thumbnail {
            aspect-ratio:16/9; border-radius:14px; margin-bottom:.45rem;
            background:linear-gradient(135deg,#050505,#29263a); color:#fff;
            display:flex; align-items:center; justify-content:center; position:relative; overflow:hidden;
        }
        .library-thumbnail::before {
            content:""; width:54px; height:38px; border-radius:12px; background:rgba(255,255,255,.16);
            backdrop-filter:blur(6px);
        }
        .library-thumbnail::after { content:"▶"; position:absolute; margin-left:3px; font-size:1rem; }
        .library-thumbnail .mode-badge {
            position:absolute; right:9px; bottom:9px; background:rgba(0,0,0,.72); color:#fff;
            border-radius:5px; padding:.18rem .4rem; font-size:.68rem; font-weight:700;
        }
        .gallery-cover { display:block; position:relative; border-radius:14px; }
        .gallery-cover-image {
            display:block;
            width:100%; aspect-ratio:16/9; object-fit:cover; border-radius:14px;
            background:linear-gradient(135deg,#050505,#29263a);
        }
        .gallery-cover::after {
            content:"▶"; position:absolute; left:50%; top:50%; transform:translate(-50%,-50%) scale(.92);
            width:48px; height:34px; display:grid; place-items:center;
            border-radius:11px; background:rgba(15,15,15,.78); color:#fff;
            opacity:0; transition:opacity .16s ease, transform .16s ease;
        }
        [class*="st-key-cover_hit_"]:hover .gallery-cover::after,
        [class*="st-key-cover_hit_"]:has(button:focus-visible) .gallery-cover::after {
            opacity:1; transform:translate(-50%,-50%) scale(1);
        }
        [class*="st-key-cover_hit_"] { position:relative; }
        [class*="st-key-cover_hit_"] [data-testid="stButton"] {
            position:absolute; inset:0; z-index:3;
        }
        [class*="st-key-cover_hit_"] [data-testid="stButton"] button {
            width:100%; height:100%; min-height:0; padding:0; opacity:0; cursor:pointer;
        }
        [class*="st-key-cover_hit_"]:has(button:focus-visible) {
            outline:3px solid var(--brand); outline-offset:3px; border-radius:14px;
        }
        [class*="st-key-recent_card_"] { position:relative; }
        [class*="st-key-gallery_title_"] button {
            justify-content:flex-start; min-height:40px; padding:.35rem 0;
            border:0; background:transparent; color:var(--ink); font-weight:700;
            line-height:1.35; text-align:left;
        }
        .gallery-mode-label {
            display:inline-flex; align-items:center; margin:-.1rem 0 .2rem;
            border-radius:5px; padding:.16rem .4rem; color:#ddd;
            background:#272727; font-size:.68rem; font-weight:700;
        }
        [class*="st-key-gallery_"] button {
            border-radius:8px !important; background:transparent !important; padding:.2rem 0 !important;
            border:0 !important; min-height:34px !important;
            text-align:left; justify-content:flex-start; font-weight:700;
        }
        [class*="st-key-learning_note_composer"] {
            margin:.8rem 0 1.15rem; padding:.35rem .35rem .2rem;
            border:1px solid rgba(139,124,246,.42) !important;
            border-radius:16px !important;
            background:linear-gradient(145deg, rgba(139,124,246,.10), rgba(39,39,39,.72));
            box-shadow:0 14px 38px rgba(0,0,0,.24);
        }
        [class*="st-key-note_card_"] {
            border-color:var(--line) !important; border-radius:14px !important;
            background:var(--panel); margin-bottom:.65rem;
        }
        [class*="st-key-note_card_"] h3 {
            font-size:1.18rem !important; line-height:1.35 !important;
            letter-spacing:-.01em;
        }
        .watch-title { font-size:1.28rem; line-height:1.35; font-weight:750; letter-spacing:-.02em; margin:.75rem 0 .15rem; }
        .watch-meta { color:var(--muted); font-size:.82rem; margin-bottom:.35rem; }
        .transcript-sheet { display:flex; flex-direction:column; gap:.45rem; padding:.2rem 0 .35rem; }
        .transcript-head, .transcript-row {
            display:grid; grid-template-columns:88px 84px minmax(0,1fr) minmax(0,1fr);
            gap:1rem; align-items:start;
        }
        .transcript-head {
            padding:.25rem 1.1rem .4rem; color:#888; font-size:.72rem;
            font-weight:750; letter-spacing:.04em;
        }
        .transcript-row {
            padding:1rem 1.1rem; border-radius:12px; background:#202020;
            border:1px solid transparent; transition:background .15s ease, border-color .15s ease;
        }
        .transcript-row:hover { background:#242424; border-color:#383838; }
        .transcript-time {
            color:#aaa; font:650 .75rem/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;
            font-variant-numeric:tabular-nums; padding-top:.15rem; white-space:nowrap;
        }
        .transcript-speaker {
            display:inline-flex; width:max-content; max-width:100%; align-items:center;
            min-height:24px; padding:.15rem .5rem; border-radius:999px;
            color:#ddd; background:#303030; font-size:.75rem; font-weight:750;
            white-space:nowrap;
        }
        .transcript-source { color:#f1f1f1; font-size:1rem; line-height:1.65; overflow-wrap:anywhere; }
        .transcript-target { color:#c9c2ff; font-size:.94rem; line-height:1.65; overflow-wrap:anywhere; }
        .transcript-empty { color:#777; }
        .form-shell { max-width: 980px; padding-top: 1.5rem; }
        @media (max-width: 768px) {
            .block-container { padding: .75rem .85rem 2rem; }
            .home-intro h1 { font-size: 1.9rem !important; }
            h1 { font-size: 1.75rem !important; }
            [class*="st-key-app_masthead"] [data-testid="stHorizontalBlock"] { gap:.35rem; }
            [class*="st-key-app_masthead"] [data-testid="stColumn"]:nth-child(1) {
                flex:0 0 38px !important; min-width:38px !important; width:38px !important;
            }
            [class*="st-key-app_masthead"] [data-testid="stColumn"]:nth-child(2) {
                flex:1 1 auto !important; min-width:90px !important;
            }
            [class*="st-key-app_masthead"] [data-testid="stColumn"]:nth-child(3) {
                flex:0 0 98px !important; min-width:98px !important; width:98px !important;
            }
            .app-masthead span:last-child { display:none; }
            [class*="st-key-app_masthead"] button {
                white-space:nowrap; font-size:.78rem; padding-left:.55rem; padding-right:.55rem;
            }
            [class*="st-key-recent_gallery"] [data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
            [class*="st-key-recent_gallery"] [data-testid="stColumn"] {
                flex:1 1 220px !important; min-width:220px !important;
            }
            .transcript-head { display:none; }
            .transcript-row {
                grid-template-columns:88px minmax(0,1fr); gap:.45rem .75rem; padding:.85rem;
            }
            .transcript-time { padding:0; }
            .transcript-source, .transcript-target { grid-column:1 / -1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _install_sidebar_toggle(lang: str) -> None:
    """Install a persistent, app-owned hamburger outside Streamlit's rerun tree."""
    collapse_label = json.dumps(t("sidebar.collapse", lang))
    show_label = json.dumps(t("sidebar.show", lang))
    st.iframe(
        f"""
        <!doctype html><html><body><script>
        (() => {{
          const host = window.parent;
          const doc = host.document;
          const collapseLabel = {collapse_label};
          const showLabel = {show_label};
          const buttonId = "englishlearn-sidebar-toggle";
          const styleId = "englishlearn-sidebar-toggle-style";

          doc.getElementById(buttonId)?.remove();
          host.__englishLearnSidebarObserver?.disconnect();
          host.clearInterval(host.__englishLearnSidebarTimer);

          let style = doc.getElementById(styleId);
          if (!style) {{
            style = doc.createElement("style");
            style.id = styleId;
            style.textContent = `
              #${{buttonId}} {{
                position:fixed; top:12px; left:12px; z-index:2147483646;
                width:40px; height:40px; display:grid; place-items:center;
                padding:0; border:1px solid #3f3f3f; border-radius:999px;
                color:#f1f1f1; background:rgba(39,39,39,.96);
                box-shadow:0 4px 16px rgba(0,0,0,.36); cursor:pointer;
                font:600 22px/1 Arial,sans-serif; transition:background .15s ease;
              }}
              #${{buttonId}}:hover {{ background:#3f3f3f; }}
              #${{buttonId}}:focus-visible {{ outline:2px solid #8b7cf6; outline-offset:2px; }}
              body.englishlearn-sidebar-collapsed [class*="st-key-app_masthead"] {{
                padding-left:48px;
              }}
              body.englishlearn-sidebar-collapsed [data-testid="stAppViewContainer"]
                > [data-testid="stSidebar"] + div {{
                position:absolute !important; inset:0 !important;
                width:100% !important; min-width:0 !important;
              }}
            `;
            doc.head.appendChild(style);
          }}

          const button = doc.createElement("button");
          button.id = buttonId;
          button.type = "button";
          button.textContent = "☰";

          const sidebarIsOpen = () => {{
            const sidebar = doc.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) return false;
            const rect = sidebar.getBoundingClientRect();
            const css = host.getComputedStyle(sidebar);
            return rect.width > 40 && rect.right > 0 &&
              css.display !== "none" && css.visibility !== "hidden";
          }};

          let lastOpen = null;
          const sync = () => {{
            const open = sidebarIsOpen();
            if (open === lastOpen) return;
            lastOpen = open;
            const label = open ? collapseLabel : showLabel;
            button.title = label;
            button.setAttribute("aria-label", label);
            button.setAttribute("aria-expanded", String(open));
            doc.body.classList.toggle("englishlearn-sidebar-collapsed", !open);
          }};

          button.addEventListener("click", () => {{
            const open = sidebarIsOpen();
            const selector = open
              ? 'button[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapseButton"] button'
              : 'button[data-testid="stExpandSidebarButton"], [data-testid="stExpandSidebarButton"] button';
            const nativeButton = doc.querySelector(selector);
            if (nativeButton) nativeButton.click();
            host.setTimeout(() => {{ lastOpen = null; sync(); }}, 60);
            host.setTimeout(() => {{ lastOpen = null; sync(); }}, 300);
          }});

          const install = () => {{
            if (!doc.body || !doc.documentElement.isConnected) return false;
            doc.body.appendChild(button);
            // Streamlit can replace the host body during a reconnect. Polling
            // this tiny state keeps the state bridge attached safely.
            host.__englishLearnSidebarTimer = host.setInterval(sync, 500);
            sync();
            return true;
          }};
          if (!install()) host.setTimeout(install, 50);
        }})();
        </script></body></html>
        """,
        width=1,
        height=1,
        tab_index=-1,
    )


def _sync_browser_title(page_title: str) -> None:
    """Keep the browser tab title in sync with the active project."""
    normalized_title = " ".join(str(page_title or "").split()).strip()
    app_title = t("app.title", lang)
    st.set_page_config(
        page_title=f"{normalized_title} · {app_title}" if normalized_title else app_title,
    )


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
DEFAULTS = {
    "video_path": None,
    "audio_path": None,
    "subtitles": None,
    "server_url": None,
    "processed": False,
    "processing": False,
    # New keys
    "ui_lang": "zh",
    "target_lang": "zh",
    "asr_model": "base",
    "proxy": "",
    "presets": None,
    "selected_preset_index": 0,
    "projects": None,
    "selected_project_id": None,
    "current_project": None,
    "reprocess_project_id": None,   # set to project ID to re-run ASR+translation
    "retranslate_project_id": None,  # set to project ID to re-run translation only
    "model_override": "",            # quick model name override (bypasses preset)
    "max_workers": 6,                # concurrent API calls for translation/segmentation
    "show_new_task": False,          # show the new-task form in the main panel
    "pending_url": "",               # saved URL when processing a new task
    "pending_upload_path": None,     # saved upload path when processing a new task
    "pipeline_error": None,          # last pipeline error message (shown once)
    "collection_notice": None,       # success notice after a player collection action
    "collection_error": None,        # actionable collection failure shown after rerun
    "pending_ai_lookup": None,       # reopen a word card after a successful AI lookup
    "show_note_composer": False,     # project-scoped voice/manual note composer
    "note_draft": None,              # subtitle/time context captured by the player
    "note_job_id": None,             # background transcription/refinement task
    "note_result": None,             # editable result returned by the note agent
    "note_job_error": None,          # transcription/refinement failure shown inline
    "note_composer_nonce": 0,        # resets recorder/input widgets between notes
    "seen_task_completions": [],     # completion notices already shown in this tab
    "scroll_main_to_top": False,     # one-shot reset when entering a different workspace view
}
_persisted_settings = load_user_settings()
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = _persisted_settings.get(k, v)

# Lazy-load from disk
if st.session_state.presets is None:
    st.session_state.presets = load_presets()
if st.session_state.projects is None:
    st.session_state.projects = load_projects()

lang = st.session_state.ui_lang  # shorthand used throughout
presets = st.session_state.presets  # shorthand used throughout


def reset_state():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.presets = load_presets()
    st.session_state.projects = load_projects()
    save_user_settings({})


def _clear_all_learning_data() -> dict[str, int]:
    """Clear all learner-created content while retaining app configuration/model files."""
    note_agent.configure(WORK_DIR)
    if not note_agent.clear_finished_tasks():
        raise RuntimeError(t("projects.delete_all_busy", lang))
    result = delete_all_project_data()
    wordbook_store.save([])
    favorites_store.save([])
    notes_store.save([])
    st.session_state.projects = []
    st.session_state.selected_project_id = None
    st.session_state.current_project = None
    st.session_state.processed = False
    st.session_state.processing = False
    st.session_state.subtitles = None
    st.session_state.video_path = None
    st.session_state.audio_path = None
    st.session_state.server_url = None
    st.session_state.show_new_task = False
    st.session_state.collection_notice = None
    st.session_state.collection_error = None
    st.session_state.show_note_composer = False
    st.session_state.note_draft = None
    st.session_state.note_job_id = None
    st.session_state.note_result = None
    st.session_state.note_job_error = None
    return result


def _persist_settings() -> None:
    save_user_settings({key: st.session_state.get(key) for key in (
        "ui_lang", "target_lang", "asr_model", "proxy", "selected_preset_index",
        "model_override", "max_workers",
    )})


def _active_preset() -> Optional[dict]:
    presets = st.session_state.presets or []
    if not presets:
        return None
    index = min(max(st.session_state.selected_preset_index, 0), len(presets) - 1)
    st.session_state.selected_preset_index = index
    return presets[index]


def _effective_preset_api_key(preset: dict) -> str:
    """Resolve credentials once so every pipeline entry point behaves alike."""
    engine = str(preset.get("engine_type") or "").strip().lower()
    configured = str(preset.get("api_key") or "").strip()
    if engine == "openai":
        return configured or os.environ.get("OPENAI_API_KEY", "").strip() \
            or os.environ.get("OPENAI_ADMIN_KEY", "").strip()
    if engine == "ollama":
        # OpenAI-compatible Ollama endpoints still require a non-empty value
        # in the SDK constructor, although it is not sent as a cloud secret.
        return configured or "ollama"
    return configured


def _translation_setup_error(preset: Optional[dict]) -> Optional[str]:
    if not preset:
        return t("pipeline.invalid_model_config", lang)
    if preset.get("engine_type") == "hy_mt2_local":
        from englishlearn.translation.hy_mt2_local import model_status
        status = model_status(preset.get("api_base") or None)
        if not (status["downloaded"] and status["dependencies_ready"]):
            return t("pipeline.model_not_ready", lang, error=status["message"])
        return None
    if not preset.get("model_name", "").strip() or not preset.get("api_base", "").strip():
        return t("pipeline.invalid_model_config", lang)
    if preset.get("engine_type") == "openai" and not _effective_preset_api_key(preset):
        return t("pipeline.openai_key_missing", lang)
    return None


# ---------------------------------------------------------------------------
# Background pipeline runner — state lives in pipeline_runner module so it
# persists across Streamlit script reruns (app.py module-level code is
# re-executed on every rerun, which would reset any state defined here).
# ---------------------------------------------------------------------------
def _get_session_token() -> str:
    if "session_token" not in st.session_state:
        st.session_state.session_token = uuid.uuid4().hex
    return st.session_state.session_token


def _active_tasks_for_session() -> dict[str, dict]:
    pipeline_runner.configure_task_store(WORK_DIR)
    return pipeline_runner.active_tasks_for_session(_get_session_token())


def _active_tasks_for_app() -> dict[str, dict]:
    """Tasks are application-wide in this local, single-user product."""
    pipeline_runner.configure_task_store(WORK_DIR)
    return pipeline_runner.all_active_tasks()


def _start_pipeline_task(params: dict) -> str:
    params["_work_dir"] = WORK_DIR
    pipeline_runner.configure_task_store(WORK_DIR)
    return pipeline_runner.start_pipeline_task(_get_session_token(), params)


def _task_label(task: dict, lang: str) -> str:
    stage = task.get("stage", "")
    key = f"pipeline.status.{stage}"
    localized = t(key, lang)
    return localized if localized != key else task.get("step_label", "")


# ---------------------------------------------------------------------------
# Focused video player with custom controls + subtitle overlay
# ---------------------------------------------------------------------------

PLAYER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  :root {
    --accent: #7c6ee6;
    --accent-soft: #b9b0ff;
    --accent-dim: rgba(124,110,230,0.18);
    --gold: #f5b942;
    --bg: #111827;
    --surface: #18202d;
    --surface-2: #131a26;
    --border: #2b3648;
    --text: #f3f5f8;
    --text-2: #a8b2c1;
    --text-3: #778397;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  html, body { height:100%; overflow:hidden; }
  body {
    background: var(--bg); color: var(--text); overflow:hidden; height:100vh;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
  .container { display:flex; height:100vh; max-height:100vh; min-height:0; overflow:hidden; position:relative; background:#000; }
  .container:fullscreen { width:100vw; height:100vh; }

  /* ===== Video ===== */
  .video-section {
    position:relative; flex:1 1 auto; width:auto; height:100%; background:#000;
    display:flex; align-items:center; justify-content:center; min-width:0; min-height:0;
  }
  .player { position:relative; width:100%; height:100%; }
  .player video, .youtube-player { width:100%; height:100%; object-fit:contain; }
  .youtube-player iframe { width:100%; height:100%; border:0; }
  /* Online sources already provide a complete, policy-compliant YouTube
     control surface. Keep our learning overlay, but never stack a second
     timeline/playback toolbar on top of the official one. */
  .player.online .center-play,
  .player.online .controls { display:none !important; }
  .online-speed {
    display:none; position:absolute; top:12px; right:60px; z-index:14;
    border:1px solid rgba(255,255,255,.3); border-radius:7px;
    padding:6px 9px; color:#fff; background:rgba(0,0,0,.72);
    font:600 12px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    cursor:pointer;
  }
  .online-speed:hover { background:rgba(124,110,230,.85); }
  .player.online .online-speed { display:block; }
  .player.online .speed-menu { bottom:auto; top:48px; right:12px; }
  .online-note {
    display:none; position:absolute; top:12px; right:112px; z-index:14;
    width:34px; height:30px; border:1px solid rgba(255,255,255,.3); border-radius:7px;
    color:#fff; background:rgba(0,0,0,.72); cursor:pointer; font-size:16px;
  }
  .online-note:hover { background:rgba(124,110,230,.85); }
  .player.online .online-note { display:block; }
  .online-subtitle-settings {
    display:none; position:absolute; top:12px; right:12px; z-index:14;
    width:34px; height:30px; border:1px solid rgba(255,255,255,.3); border-radius:7px;
    color:#fff; background:rgba(0,0,0,.72); cursor:pointer; font-size:16px;
  }
  .online-subtitle-settings:hover { background:rgba(124,110,230,.85); }
  .player.online .online-subtitle-settings { display:block; }
  .sub-overlay {
    position:absolute; inset:0; pointer-events:none;
    opacity:0; transition:opacity 0.25s; z-index:5;
  }
  .sub-overlay.show { opacity:1; }
  .subtitle-layer {
    position:absolute; left:50%; top:78%; transform:translate(-50%,-50%);
    width:max-content; max-width:96%; text-align:center; pointer-events:none;
  }
  .subtitle-original-layer {
    left:var(--video-subtitle-original-x, 50%); top:var(--video-subtitle-original-y, 80%);
  }
  .subtitle-translation-layer {
    left:var(--video-subtitle-translation-x, 50%); top:var(--video-subtitle-translation-y, 70%);
  }
  .subtitle-layer:not(.has-content) { display:none; }
  .subtitle-drag-handle {
    display:block; margin:0 auto 4px; padding:2px 8px; border:0; border-radius:999px;
    color:rgba(255,255,255,.88); background:rgba(0,0,0,.58); cursor:grab;
    font:600 11px/1 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    letter-spacing:.03em; opacity:.5; touch-action:none; user-select:none; pointer-events:auto;
  }
  .subtitle-drag-handle:hover, .subtitle-drag-handle:focus-visible { opacity:1; background:rgba(31,25,67,.88); outline:1px solid var(--accent-soft); }
  .subtitle-drag-handle:active, .subtitle-layer.is-dragging .subtitle-drag-handle { cursor:grabbing; opacity:1; }
  .subtitle-layer .orig {
    color:#fff; font-size:calc(15px * var(--video-subtitle-scale, 1)); font-weight:500; line-height:1.3;
    text-shadow: 0 0 2px #000, 0 1px 4px rgba(0,0,0,0.9);
    background:rgba(0,0,0,0.6); padding:3px 12px; border-radius:5px;
    display:inline-block; max-width:100%; pointer-events:auto;
  }
  .subtitle-layer .trans {
    color:var(--accent-soft); font-size:calc(13px * var(--video-subtitle-scale, 1)); line-height:1.3; margin-top:3px;
    text-shadow: 0 0 2px #000, 0 1px 4px rgba(0,0,0,0.9);
    background:rgba(0,0,0,0.5); padding:2px 10px; border-radius:5px;
    display:inline-block; max-width:100%; pointer-events:auto;
  }
  .sub-word { cursor:text; border-radius:3px; transition:background .12s; }
  .sub-word.word-hover-ready { background:rgba(124,110,230,.28); }
  .learning-word-lookup-off .sub-word.word-hover-ready { background:transparent; }
  .learning-video-original-off .subtitle-original-layer,
  .learning-video-translation-off .subtitle-translation-layer,
  .learning-list-bilingual-off .sub-translation { display:none; }

  .center-play {
    position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
    width:68px; height:68px; border-radius:50%;
    background:rgba(0,0,0,0.65); cursor:pointer; z-index:10;
    display:flex; align-items:center; justify-content:center;
    opacity:0; transition:opacity 0.3s, transform 0.2s;
  }
  .center-play.show { opacity:1; }
  .center-play:hover { background:rgba(0,0,0,0.8); transform:translate(-50%,-50%) scale(1.08); }
  .center-play::before {
    content:''; width:0; height:0;
    border-left:20px solid #fff;
    border-top:13px solid transparent; border-bottom:13px solid transparent;
    margin-left:5px;
  }

  .controls {
    position:absolute; bottom:0; left:0; right:0; z-index:8;
    background:linear-gradient(transparent, rgba(0,0,0,0.85));
    padding:24px 14px 8px; opacity:0; transition:opacity 0.3s ease;
  }
  .player:hover .controls, .player.controls-pinned .controls { opacity:1; }

  .progress-bar {
    position:relative; height:4px; background:rgba(255,255,255,0.22);
    cursor:pointer; border-radius:2px; margin-bottom:8px; transition:height 0.15s;
  }
  .progress-bar:hover, .progress-bar.dragging { height:7px; }
  .progress-bar .buffered { position:absolute; height:100%; background:rgba(255,255,255,0.35); border-radius:2px; left:0; top:0; }
  .progress-bar .played { position:absolute; height:100%; background:var(--accent); border-radius:2px; left:0; top:0; }
  .progress-bar .thumb {
    position:absolute; width:13px; height:13px; background:#fff; border-radius:50%;
    top:50%; transform:translate(-50%,-50%); box-shadow:0 0 6px rgba(0,0,0,0.6);
    opacity:0; transition:opacity 0.15s; pointer-events:none;
  }
  .progress-bar:hover .thumb, .progress-bar.dragging .thumb { opacity:1; }

  .ctrl-row { display:flex; align-items:center; gap:8px; color:#fff; font-size:13px; }
  .ctrl-btn {
    cursor:pointer; background:none; border:none; color:#fff;
    padding:5px 7px; border-radius:5px; transition:all 0.15s;
    font-size:15px; display:flex; align-items:center; justify-content:center;
    min-width:30px; height:30px;
  }
  .ctrl-btn:hover { background:rgba(255,255,255,0.15); color:var(--accent-soft); }
  .ctrl-btn.active { color:var(--accent-soft); }
  .ctrl-speed { color:var(--accent-soft); font-weight:600; font-size:12px; }

  .time-display { font-variant-numeric:tabular-nums; font-size:12px; color:rgba(255,255,255,0.75); }
  .spacer { flex:1; }

  .vol-wrap { display:flex; align-items:center; gap:2px; }
  .vol-slider { width:0; overflow:hidden; transition:width 0.25s ease; }
  .vol-wrap:hover .vol-slider { width:64px; }
  .vol-slider input {
    width:64px; height:3px; -webkit-appearance:none; appearance:none;
    background:rgba(255,255,255,0.3); border-radius:2px; outline:none; cursor:pointer;
  }
  .vol-slider input::-webkit-slider-thumb { -webkit-appearance:none; width:11px; height:11px; background:#fff; border-radius:50%; cursor:pointer; }
  .vol-slider input::-moz-range-thumb { width:11px; height:11px; background:#fff; border-radius:50%; border:none; cursor:pointer; }

  .speed-menu {
    position:absolute; bottom:48px; right:60px; z-index:12;
    background:rgba(0,0,0,0.93); border-radius:10px; padding:6px 0;
    display:none; min-width:86px; box-shadow:0 4px 16px rgba(0,0,0,0.5);
  }
  .speed-menu.show { display:block; }
  .speed-item {
    padding:7px 18px; cursor:pointer; color:rgba(255,255,255,0.7);
    font-size:13px; text-align:center; transition:all 0.12s;
  }
  .speed-item:hover { color:#fff; background:rgba(255,255,255,0.1); }
  .speed-item.active { color:var(--accent-soft); font-weight:600; }

  /* Subtitle controls are deliberately collected in the player settings menu,
     not kept visible over the video or mixed into the reading panel. */
  .subtitle-settings-menu {
    position:absolute; right:10px; bottom:48px; z-index:20; display:none;
    width:276px; padding:8px; border:1px solid rgba(255,255,255,.18); border-radius:10px;
    color:#edf1ff; background:rgba(10,13,21,.95); backdrop-filter:blur(12px);
    box-shadow:0 8px 24px rgba(0,0,0,.42); font:500 12px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  }
  .subtitle-settings-menu.show { display:block; }
  .player.online .subtitle-settings-menu { top:48px; right:12px; bottom:auto; }
  .subtitle-settings-heading { padding:3px 5px 7px; color:#fff; font-weight:700; }
  .subtitle-setting-row {
    display:flex; align-items:center; justify-content:space-between; gap:12px; width:100%;
    margin:2px 0; padding:8px 7px; border:0; border-radius:7px; color:#e6e8f3;
    background:transparent; cursor:pointer; font:inherit; text-align:left;
  }
  .subtitle-setting-row:hover, .subtitle-setting-row:focus-visible { background:rgba(124,110,230,.2); outline:none; }
  .subtitle-setting-row .setting-state { color:#a9a2ff; font-weight:700; }
  .subtitle-setting-row:not(.active) .setting-state { color:#a9b0c3; }
  .subtitle-setting-divider { height:1px; margin:7px 3px; background:rgba(255,255,255,.12); }
  .subtitle-setting-control { display:grid; grid-template-columns:1fr auto; gap:6px 10px; padding:7px 5px; color:#d5d9e8; }
  .subtitle-setting-control label { grid-column:1 / -1; }
  .subtitle-setting-control input { width:100%; min-width:0; accent-color:var(--accent-soft); }
  .subtitle-setting-control output { min-width:42px; color:#fff; text-align:right; font-variant-numeric:tabular-nums; }
  .subtitle-timing-adjust { display:flex; align-items:center; gap:5px; }
  .player-timing-btn {
    border:0; border-radius:5px; cursor:pointer; padding:2px 5px;
    background:rgba(255,255,255,.11); color:#fff; font:600 12px inherit; line-height:1.25;
  }
  .player-timing-btn:hover { background:rgba(255,255,255,.22); }

  /* ===== Right Panel ===== */
  .right-panel {
    position:relative; z-index:18; flex:0 0 var(--panel-width, 42%);
    width:var(--panel-width, 42%); height:100%; background:#131a26; border-left:1px solid rgba(255,255,255,.14);
    display:flex; flex-direction:column; min-width:0; min-height:0; overflow:hidden;
    transition:opacity .18s ease, width .08s linear, flex-basis .08s linear;
  }
  .container.is-fullscreen .right-panel {
    width:var(--panel-width, 42%); flex-basis:var(--panel-width, 42%);
    border-left:1px solid rgba(255,255,255,.16);
  }
  .panel-resizer {
    display:none; position:absolute; top:0; bottom:0; right:calc(var(--panel-width, 42%) - 5px); width:10px; z-index:51;
    align-items:center; justify-content:center; cursor:col-resize;
    background:transparent; border:0; outline:none; touch-action:none;
  }
  .container:not(.panel-collapsed) .panel-resizer { display:flex; }
  .panel-resizer::before {
    content:''; width:3px; height:44px; border-radius:3px;
    background:#526078; box-shadow:0 0 0 1px rgba(255,255,255,.03);
    transition:background .15s, transform .15s;
  }
  .panel-resizer:hover::before,
  .panel-resizer:focus-visible::before,
  .container.is-resizing .panel-resizer::before {
    background:var(--accent-soft); transform:scaleX(1.35);
  }
  .panel-resizer:focus-visible {
    box-shadow:inset 0 0 0 2px var(--accent);
  }
  .container.is-resizing { user-select:none; }
  .container.is-resizing::after {
    content:''; position:absolute; inset:0; z-index:50; cursor:col-resize;
    background:transparent;
  }
  .container.panel-collapsed .right-panel {
    width:0; flex-basis:0; border-left:0; opacity:0; overflow:hidden;
    pointer-events:none;
  }
  .container.panel-collapsed .panel-resizer { display:none; }
  .panel-header {
    min-height:40px; padding:6px 8px 6px 12px; display:flex;
    align-items:center; gap:8px; background:var(--surface-2);
    border-bottom:1px solid var(--border);
  }
  .panel-title {
    flex:1; min-width:0; color:var(--text); font-size:13px; font-weight:650;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  }
  .panel-actions { display:flex; align-items:center; gap:4px; }
  .panel-action, .floating-action {
    border:1px solid var(--border); background:rgba(255,255,255,.04);
    color:var(--text-2); border-radius:7px; cursor:pointer; font:inherit;
    font-size:11px; line-height:1; padding:7px 9px; white-space:nowrap;
    transition:background .15s, color .15s, border-color .15s;
  }
  .panel-action:hover, .floating-action:hover {
    color:var(--text); background:rgba(255,255,255,.09); border-color:#47556b;
  }
  .panel-action.active { color:var(--accent-soft); border-color:var(--accent); }
  .panel-edge-restore {
    display:none; position:absolute; z-index:22; top:50%; right:0; transform:translateY(-50%);
    width:18px; height:72px; padding:0; border:1px solid rgba(255,255,255,.2); border-right:0;
    border-radius:10px 0 0 10px; color:#fff; background:rgba(19,26,38,.88);
    backdrop-filter:blur(10px); cursor:pointer; box-shadow:-8px 0 24px rgba(0,0,0,.3);
  }
  .panel-edge-restore:hover, .panel-edge-restore:focus-visible { width:24px; background:rgba(124,110,230,.82); outline:none; }
  .container.panel-collapsed .panel-edge-restore { display:block; }
  .tab-bar {
    display:flex; border-bottom:1px solid var(--border);
    background:var(--surface-2);
  }
  .subtitle-tools {
    display:none; grid-template-columns:minmax(0,1fr) auto; align-items:center; gap:7px 8px; padding:7px 12px;
    border-bottom:1px solid var(--border); background:var(--surface-2);
    color:var(--text-2); font-size:11px;
  }
  .subtitle-tools.show { display:grid; }
  .subtitle-search {
    grid-column:1 / -1; width:100%; min-width:0; padding:7px 10px;
    border:1px solid var(--border); border-radius:7px; outline:none;
    color:var(--text); background:#0f1520; font:inherit;
  }
  .subtitle-search:focus { border-color:var(--accent); box-shadow:0 0 0 2px var(--accent-dim); }
  .subtitle-display-control { display:grid; grid-template-columns:auto minmax(60px,1fr) 38px; align-items:center; gap:8px; grid-column:1 / -1; }
  .subtitle-display-control label { white-space:nowrap; }
  .subtitle-display-control input { accent-color:var(--accent); min-width:60px; }
  .subtitle-display-control output { min-width:38px; text-align:right; color:var(--accent-soft); font-variant-numeric:tabular-nums; }
  .sub-row.search-hidden { display:none; }
  .subtitle-search-empty { display:none; padding:22px 16px; text-align:center; color:var(--text-3); font-size:12px; }
  .subtitle-search-empty.show { display:block; }
  .tab {
    flex:1; padding:10px 8px; cursor:pointer; text-align:center;
    font-size:12px; color:var(--text-2); border:none; background:none;
    border-bottom:2px solid transparent; transition:all 0.15s;
    font-family:inherit;
  }
  .tab:hover { color:var(--text); background:rgba(255,255,255,0.03); }
  .tab.active { color:var(--accent-soft); border-bottom-color:var(--accent); font-weight:600; }

  .tab-content { flex:1 1 0; min-height:0; overflow-y:auto; overflow-x:hidden; padding:6px 4px; display:none; }
  .tab-content.active { display:block; }
  .tab-content::-webkit-scrollbar { width:5px; }
  .tab-content::-webkit-scrollbar-track { background:transparent; }
  .tab-content::-webkit-scrollbar-thumb { background:#3a3a5a; border-radius:4px; }

  .empty-hint { padding:32px 20px; text-align:center; color:var(--text-3); font-size:13px; line-height:1.6; }

  /* Subtitle rows */
  .sub-row {
    display:flex; gap:8px; padding:9px 12px; margin:2px 4px;
    border-radius:8px; cursor:pointer; transition:all 0.2s;
    border:1px solid transparent; align-items:flex-start;
  }
  .sub-row:hover { background:rgba(255,255,255,0.04); border-color:var(--border); }
  .sub-row.active {
    background:var(--accent-dim); border-color:var(--accent);
    box-shadow:0 0 12px rgba(124,110,230,0.16);
  }
  .sub-time {
    flex:0 0 42px; font-size:11px; color:var(--text-3);
    font-variant-numeric:tabular-nums; text-align:right; padding-top:3px;
    font-family:'SF Mono','Fira Code',monospace;
  }
  .sub-row.active .sub-time { color:var(--accent-soft); }
  .sub-content { flex:1; min-width:0; user-select:text; }
  .sub-original { font-size:calc(14px * var(--list-subtitle-scale, 1)); line-height:1.5; color:var(--text); word-break:break-word; }
  .sub-translation { font-size:calc(13px * var(--list-subtitle-scale, 1)); line-height:1.5; color:var(--accent-soft); margin-top:3px; word-break:break-word; }
  .sub-translation:empty::after { content:"__PENDING_TRANSLATION__"; color:#555; font-style:italic; }
  .note-row-btn {
    flex:0 0 26px; background:none; border:none; cursor:pointer;
    font-size:15px; color:var(--text-3); padding:2px; border-radius:4px;
    transition:all 0.15s; align-self:flex-start;
  }
  .note-row-btn:hover { color:var(--accent-soft); transform:scale(1.12); }

  /* Favorite / wordbook items */
  .item-row {
    display:flex; gap:8px; padding:9px 12px; margin:2px 4px;
    border-radius:8px; transition:all 0.2s; border:1px solid transparent;
    align-items:flex-start;
  }
  .item-row:hover { background:rgba(255,255,255,0.04); border-color:var(--border); }
  .item-main { flex:1; min-width:0; cursor:pointer; }
  .item-time { flex:0 0 42px; font-size:11px; color:var(--text-3); text-align:right; padding-top:3px; font-family:'SF Mono',monospace; font-variant-numeric:tabular-nums; }
  .item-text { font-size:14px; line-height:1.5; color:var(--text); word-break:break-word; }
  .item-trans { font-size:13px; line-height:1.5; color:var(--accent-soft); margin-top:2px; word-break:break-word; }
  .item-word { font-size:15px; font-weight:600; color:var(--text); }
  .item-word-trans { font-size:13px; color:var(--accent-soft); margin-top:2px; }
  .item-pos { font-size:11px; color:var(--text-3); font-style:italic; margin-left:6px; font-weight:normal; }
  .item-pron { font-size:12px; color:var(--text-2); margin-top:2px; }
  .item-other { font-size:11px; color:var(--text-3); margin-top:3px; line-height:1.4; }
  .item-context { font-size:12px; color:var(--text-3); margin-top:4px; font-style:italic; }
  .item-del {
    flex:0 0 26px; background:none; border:none; cursor:pointer;
    font-size:14px; color:var(--text-3); padding:2px; border-radius:4px;
    transition:all 0.15s; align-self:flex-start;
  }
  .item-del:hover { color:#ff4444; }

  /* Word popup */
  .word-popup {
    position:fixed; z-index:100; background:#101114;
    border:1px solid #343944; border-radius:12px;
    padding:16px 18px; width:min(390px, calc(100vw - 16px)); max-height:min(72vh, 560px);
    overflow-y:auto;
    box-shadow:0 16px 40px rgba(0,0,0,.55);
    display:none;
  }
  .word-popup.show { display:block; }
  .popup-header {
    display:flex; align-items:flex-start; gap:10px; padding-bottom:12px;
    border-bottom:1px solid #30343d;
  }
  .player-startup-error {
    display:none; position:absolute; inset:16px; z-index:80; place-items:center;
    padding:18px; border:1px solid #6b3540; border-radius:10px;
    color:#ffd4da; background:rgba(28,15,20,.95); font-size:13px; line-height:1.5;
    text-align:center;
  }
  .player-startup-error.show { display:grid; }
  .popup-word {
    color:#f4f6fb; font-size:24px; font-weight:650; line-height:1.2;
    min-width:0; flex:1; word-break:break-word;
  }
  .popup-icon-button {
    flex:0 0 32px; width:32px; height:32px; padding:0; border:0; border-radius:50%;
    color:#a9b0c3; background:transparent; cursor:pointer; font-size:22px; line-height:1;
    transition:color .15s, background .15s, transform .15s;
  }
  .popup-icon-button:hover:not(:disabled), .popup-icon-button:focus-visible {
    color:var(--gold); background:rgba(255,255,255,.08); outline:none; transform:scale(1.06);
  }
  .popup-icon-button.faved { color:var(--gold); }
  .popup-icon-button:disabled { cursor:wait; opacity:.5; }
  .lookup-phonetics { display:flex; flex-wrap:wrap; gap:6px 12px; margin-bottom:9px; color:#b8bfce; font-size:13px; }
  .lookup-phonetic { display:inline-flex; align-items:center; gap:4px; }
  .lookup-audio {
    width:24px; height:24px; padding:0; border:0; border-radius:50%; cursor:pointer;
    color:#cbc6ff; background:rgba(124,110,230,.15); font-size:12px;
  }
  .lookup-audio:hover, .lookup-audio.playing { color:#fff; background:rgba(124,110,230,.48); }
  .lookup-translations { margin-bottom:8px; color:#f0f2f8; }
  .lookup-translation { margin:3px 0; }
  .lookup-meaning { margin:7px 0; color:#cfd4df; font-size:13px; line-height:1.45; }
  .lookup-pos { margin-right:5px; color:var(--accent-soft); font-style:italic; }
  .lookup-example { margin-top:2px; color:#8f97a9; font-size:12px; }
  .lookup-source { margin-top:9px; color:#697184; font-size:10px; }
  }
  .popup-lookup {
    min-height:22px; color:#d9dee8; font-size:15px; line-height:1.65;
    margin-top:12px; white-space:pre-line; word-break:break-word;
  }
  .popup-ai-button {
    width:100%; margin-top:14px; padding:9px 10px; border:1px solid #5750a6;
    border-radius:8px; background:rgba(124,110,230,.16); color:var(--accent-soft);
    cursor:pointer; font:600 13px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    transition:background .15s, border-color .15s, opacity .15s;
  }
  .popup-ai-button:hover:not(:disabled) { background:rgba(124,110,230,.28); border-color:var(--accent-soft); }
  .popup-ai-button:disabled { cursor:wait; opacity:.6; }

  @media (max-width:640px) {
    .container { flex-direction:column; }
    .video-section { width:100%; height:auto; flex:1 1 auto; }
    .right-panel {
      width:100%; height:var(--panel-height, 50%); flex:0 0 var(--panel-height, 50%);
      border-left:none; border-top:1px solid var(--border);
    }
    .container.is-fullscreen .right-panel {
      width:100%; height:var(--panel-height, 50%); flex-basis:var(--panel-height, 50%);
      border-top:1px solid rgba(255,255,255,.16);
    }
    .panel-resizer {
      left:0; right:0; top:calc(100% - var(--panel-height, 50%) - 5px); bottom:auto;
      width:100%; height:10px; cursor:row-resize;
      border-left:0; border-right:0; border-top:1px solid var(--border);
      border-bottom:1px solid var(--border);
    }
    .panel-resizer::before { width:44px; height:3px; }
    .panel-resizer:hover::before,
    .panel-resizer:focus-visible::before,
    .container.is-resizing .panel-resizer::before {
      transform:scaleY(1.35);
    }
    .container.is-resizing::after { cursor:row-resize; }
    .container.panel-collapsed .right-panel { width:100%; height:0; flex-basis:0; }
    .panel-edge-restore { top:auto; bottom:0; right:50%; transform:translateX(50%); width:72px; height:18px; border-radius:10px 10px 0 0; border-right:1px solid rgba(255,255,255,.2); border-bottom:0; }
    .panel-edge-restore:hover, .panel-edge-restore:focus-visible { width:84px; height:24px; }
    .panel-action .action-label { display:none; }
    .subtitle-settings-menu { right:8px; bottom:48px; width:min(276px, calc(100% - 16px)); }
    .player.online .subtitle-settings-menu { top:48px; right:8px; bottom:auto; }
  }
</style>
</head>
<body>

<div class="container" id="learningShell">
  <!-- ===== VIDEO ===== -->
  <div class="video-section">
    <div class="player __PLAYER_CLASS__" id="player">
      __VIDEO_ELEMENT__
      <div class="center-play show" id="centerPlay"></div>
      <button class="online-note" id="onlineNoteBtn" title="__NOTE_CAPTURE__" aria-label="__NOTE_CAPTURE__">&#9998;</button>
      <button class="online-speed" id="onlineSpeedBtn" title="__SPEED__">1.0x</button>
      <button class="online-subtitle-settings" id="onlineSubtitleSettingsBtn" title="__SUBTITLE_SETTINGS__" aria-label="__SUBTITLE_SETTINGS__">&#9881;</button>
      <div class="sub-overlay" id="subOverlay">
        <div class="subtitle-layer subtitle-original-layer has-content" id="subOrigLayer">
          <button type="button" class="subtitle-drag-handle" id="subOrigDragHandle" title="__SUBTITLE_DRAG_HINT__" aria-label="__SUBTITLE_DRAG_ORIGINAL__">&#10303; __SUBTITLE_DRAG_ORIGINAL__</button>
          <div class="orig" id="subOrig"></div>
        </div>
        <div class="subtitle-layer subtitle-translation-layer has-content" id="subTransLayer">
          <button type="button" class="subtitle-drag-handle" id="subTransDragHandle" title="__SUBTITLE_DRAG_HINT__" aria-label="__SUBTITLE_DRAG_TRANSLATION__">&#10303; __SUBTITLE_DRAG_TRANSLATION__</button>
          <div class="trans" id="subTrans"></div>
        </div>
      </div>
      <div class="speed-menu" id="speedMenu">
        <div class="speed-item" data-s="0.5">0.5x</div>
        <div class="speed-item" data-s="0.75">0.75x</div>
        <div class="speed-item active" data-s="1">1.0x</div>
        <div class="speed-item" data-s="1.25">1.25x</div>
        <div class="speed-item" data-s="1.5">1.5x</div>
        <div class="speed-item" data-s="2">2.0x</div>
      </div>
      <div class="subtitle-settings-menu" id="subtitleSettingsMenu" role="dialog" aria-label="__SUBTITLE_SETTINGS__">
        <div class="subtitle-settings-heading">__SUBTITLE_SETTINGS__</div>
        <button type="button" class="subtitle-setting-row active" id="settingsSubtitlesToggle"><span>__SETTINGS_SUBTITLES__</span><span class="setting-state" id="settingsSubtitlesState">__SUBTITLE_ON_SHORT__</span></button>
        <button type="button" class="subtitle-setting-row active" id="settingsOriginalToggle"><span>__VIDEO_ORIGINAL_ON__</span><span class="setting-state" id="settingsOriginalState">__SUBTITLE_ON_SHORT__</span></button>
        <button type="button" class="subtitle-setting-row active" id="settingsTranslationToggle"><span>__VIDEO_TRANSLATION_ON__</span><span class="setting-state" id="settingsTranslationState">__SUBTITLE_ON_SHORT__</span></button>
        <button type="button" class="subtitle-setting-row active" id="settingsWordLookupToggle"><span>__WORD_LOOKUP_SETTING__</span><span class="setting-state" id="settingsWordLookupState">__SUBTITLE_ON_SHORT__</span></button>
        <div class="subtitle-setting-divider"></div>
        <div class="subtitle-setting-control">
          <label for="videoSubFontSize">__VIDEO_FONT_SIZE__</label>
          <input id="videoSubFontSize" type="range" min="80" max="200" step="5" value="100" aria-label="__VIDEO_FONT_SIZE__">
          <output id="videoSubFontValue">100%</output>
        </div>
        <button type="button" class="subtitle-setting-row" id="settingsPositionReset"><span>__SUBTITLE_POSITION_RESET__</span><span class="setting-state">↺</span></button>
        <div class="subtitle-setting-divider"></div>
        <div class="subtitle-setting-control" role="group" aria-label="__SUBTITLE_TIMING_CONTROLS__" title="__SUBTITLE_TIMING_HINT__">
          <label for="subtitleOffset">__SUBTITLE_TIMING__</label>
          <div class="subtitle-timing-adjust">
            <button type="button" class="player-timing-btn" id="subtitleOffsetEarlier" title="__SUBTITLE_OFFSET_EARLIER__" aria-label="__SUBTITLE_OFFSET_EARLIER__">−</button>
            <input id="subtitleOffset" type="range" min="-5" max="5" step="0.1" value="0" aria-label="__SUBTITLE_TIMING__" title="__SUBTITLE_TIMING_HINT__">
            <button type="button" class="player-timing-btn" id="subtitleOffsetLater" title="__SUBTITLE_OFFSET_LATER__" aria-label="__SUBTITLE_OFFSET_LATER__">+</button>
          </div>
          <span><output id="subtitleOffsetValue" aria-live="polite">0.0s</output> <button type="button" class="player-timing-btn" id="subtitleOffsetReset" title="__SUBTITLE_OFFSET_RESET__" aria-label="__SUBTITLE_OFFSET_RESET__">↺</button></span>
        </div>
      </div>
      <div class="controls" id="controls">
        <div class="progress-bar" id="progressBar">
          <div class="buffered" id="buffered"></div>
          <div class="played" id="played"></div>
          <div class="thumb" id="thumb"></div>
        </div>
        <div class="ctrl-row">
          <button class="ctrl-btn" id="playBtn" title="Play / Pause">&#9654;</button>
          <div class="vol-wrap">
            <button class="ctrl-btn" id="muteBtn">&#128266;</button>
            <div class="vol-slider"><input type="range" id="volume" min="0" max="1" step="0.05" value="1"></div>
          </div>
          <span class="time-display"><span id="current">0:00</span> / <span id="duration">0:00</span></span>
          <span class="spacer"></span>
          <button class="ctrl-btn" id="noteBtn" title="__NOTE_CAPTURE__" aria-label="__NOTE_CAPTURE__">&#9998;</button>
          <button class="ctrl-btn active" id="subToggle" style="font-weight:bold;" title="__SETTINGS_SUBTITLES__">CC</button>
          <button class="ctrl-btn" id="subtitleSettingsBtn" title="__SUBTITLE_SETTINGS__" aria-label="__SUBTITLE_SETTINGS__">&#9881;</button>
          <button class="ctrl-btn" id="speedBtn" title="Speed">1.0x</button>
          <button class="ctrl-btn" id="fsBtn" title="__FULLSCREEN__">&#9974;</button>
        </div>
      </div>
    </div>
  </div>

  <div class="panel-resizer" id="panelResizer" role="separator"
       aria-label="__PANEL_RESIZE__" aria-orientation="vertical"
       aria-valuemin="24" aria-valuemax="55" aria-valuenow="42"
       tabindex="0" title="__PANEL_RESIZE_HINT__"></div>

  <!-- ===== RIGHT PANEL ===== -->
  <div class="right-panel" id="rightPanel">
    <div class="panel-header">
      <div class="panel-title">__SUB_PANEL_TITLE__</div>
      <div class="panel-actions">
        <button class="panel-action active" id="followBtn" aria-pressed="true" title="__FOLLOW_ON__">&#9678; <span class="action-label" id="followLabel">__FOLLOW_ON__</span></button>
        <button class="panel-action active" id="listBilingualPanelBtn" title="__LIST_BILINGUAL_TITLE__">__LIST_BILINGUAL_ON__</button>
        <button class="panel-action" id="subtitleToolsBtn" aria-expanded="false" title="__SUBTITLE_TOOLS__" aria-label="__SUBTITLE_TOOLS__">&#128269;</button>
      </div>
    </div>
    <div class="tab-bar">
      <button class="tab active" data-tab="subs">&#128196; __TAB_SUBS__</button>
      <button class="tab" data-tab="notes">&#128221; __TAB_NOTES__ <span class="tab-count" id="noteCount"></span></button>
      <button class="tab" data-tab="words">&#128214; __TAB_WORDS__ <span class="tab-count" id="wordCount"></span></button>
    </div>
    <div class="subtitle-tools" id="subtitleTools">
      <input class="subtitle-search" id="subtitleSearch" type="search" placeholder="__SUBTITLE_SEARCH_PLACEHOLDER__" aria-label="__SUBTITLE_SEARCH__">
      <div class="subtitle-display-control">
        <label for="listSubFontSize">__LIST_FONT_SIZE__</label>
        <input id="listSubFontSize" type="range" min="80" max="180" step="5" value="100" aria-label="__LIST_FONT_SIZE__">
        <output id="listSubFontValue">100%</output>
      </div>
    </div>

    <div class="tab-content active" id="content-subs" role="tabpanel" aria-label="__TAB_SUBS__" tabindex="0"></div>
    <div class="subtitle-search-empty" id="subtitleSearchEmpty">__SUBTITLE_SEARCH_EMPTY__</div>
    <div class="tab-content" id="content-notes" role="tabpanel" aria-label="__TAB_NOTES__"></div>
    <div class="tab-content" id="content-words" role="tabpanel" aria-label="__TAB_WORDS__"></div>
  </div>
  <button class="panel-edge-restore" id="panelRestoreBtn" title="__PANEL_RESTORE__" aria-label="__PANEL_RESTORE__">&#8249;</button>
  <div class="player-startup-error" id="playerStartupError" role="alert"></div>
</div>

<!-- Word popup -->
<div class="word-popup" id="wordPopup">
  <div class="popup-header">
    <div class="popup-word" id="popupWord"></div>
    <button type="button" class="popup-icon-button" id="popupWordSave" aria-label="__WORD_SAVE__" title="__WORD_SAVE__">&#9734;</button>
  </div>
  <div class="popup-lookup" id="popupLookup"></div>
  <button type="button" class="popup-ai-button" id="popupAiButton">__AI_TRANSLATE__</button>
</div>

<script type="application/json" id="subtitle-data">__SUBTITLE_JSON__</script>
<script type="application/json" id="notes-data">__NOTES_JSON__</script>
<script type="application/json" id="wordbook-data">__WORDBOOK_JSON__</script>
<script type="application/json" id="ai-lookup-data">__AI_LOOKUP_JSON__</script>

<script>
(function() {
  function showPlayerStartupError(error) {
    var el = document.getElementById('playerStartupError');
    if (!el) return;
    el.textContent = '__PLAYER_INIT_FAILED__ ' + (error && error.message ? error.message : '');
    el.classList.add('show');
  }
  window.addEventListener('error', function(event) { showPlayerStartupError(event.error || event); });
  window.addEventListener('unhandledrejection', function(event) { showPlayerStartupError(event.reason || event); });
})();
</script>

<script>
(function() {
  var PROJECT_ID = "__PROJECT_ID__";
  var INITIAL_SEEK = __INITIAL_SEEK__;
  var PLAYBACK_MODE = "__PLAYBACK_MODE__";
  var YOUTUBE_VIDEO_ID = "__YOUTUBE_VIDEO_ID__";
  var FREE_TRANSLATE_TARGET = "__FREE_TRANSLATE_TARGET__";
  var LOOKUP_API_BASE = "__LOOKUP_API_BASE__";
  var INITIAL_SUBTITLE_OFFSET = __INITIAL_SUBTITLE_OFFSET__;

  var subtitles = JSON.parse(document.getElementById('subtitle-data').textContent);
  var notes = JSON.parse(document.getElementById('notes-data').textContent);
  var wordbook = JSON.parse(document.getElementById('wordbook-data').textContent);
  var initialAiLookup = JSON.parse(document.getElementById('ai-lookup-data').textContent || 'null');

  var player = document.getElementById('player');
  var learningShell = document.getElementById('learningShell');
  var rightPanel = document.getElementById('rightPanel');
  var panelResizer = document.getElementById('panelResizer');
  var $ = function(id){ return document.getElementById(id); };
  var nativeVideo = document.getElementById('video');
  var video = nativeVideo;

  // A small HTMLMediaElement-compatible adapter keeps every existing learning
  // interaction (timeline, shortcuts, subtitle sync and click-to-seek) working
  // with YouTube's official IFrame Player API.
  if (PLAYBACK_MODE === 'online') {
    var ytPlayer = null, ytReady = false, ytTime = 0, ytDuration = 0;
    var ytPaused = true, ytMuted = false, ytVolume = 1, ytRate = 1;
    var ytHandlers = {};
    function ytEmit(name) {
      (ytHandlers[name] || []).forEach(function(fn){ try { fn(); } catch (_) {} });
    }
    video = {
      buffered: { length: 0, end: function(){ return 0; } },
      addEventListener: function(name, fn) { (ytHandlers[name] = ytHandlers[name] || []).push(fn); },
      play: function() { if (ytReady) ytPlayer.playVideo(); return Promise.resolve(); },
      pause: function() { if (ytReady) ytPlayer.pauseVideo(); },
    };
    Object.defineProperties(video, {
      currentTime: {
        get: function(){ return ytReady ? (ytPlayer.getCurrentTime() || 0) : ytTime; },
        set: function(value){ ytTime = Number(value) || 0; if (ytReady) { ytPlayer.seekTo(ytTime, true); setTimeout(function(){ ytEmit('seeked'); ytEmit('timeupdate'); }, 80); } }
      },
      duration: { get: function(){ return ytReady ? (ytPlayer.getDuration() || ytDuration) : ytDuration; } },
      paused: { get: function(){ return ytPaused; } },
      muted: {
        get: function(){ return ytMuted; },
        set: function(value){ ytMuted = !!value; if (ytReady) { if (ytMuted) ytPlayer.mute(); else ytPlayer.unMute(); } }
      },
      volume: {
        get: function(){ return ytVolume; },
        set: function(value){ ytVolume = Math.max(0, Math.min(1, Number(value))); if (ytReady) ytPlayer.setVolume(ytVolume * 100); }
      },
      playbackRate: {
        get: function(){ return ytRate; },
        set: function(value){ ytRate = Number(value) || 1; if (ytReady) ytPlayer.setPlaybackRate(ytRate); }
      }
    });
    window.onYouTubeIframeAPIReady = function() {
      ytPlayer = new YT.Player('youtubePlayer', {
        videoId: YOUTUBE_VIDEO_ID,
        playerVars: { playsinline: 1, rel: 0, controls: 1 },
        events: {
          onReady: function(event) {
            ytReady = true;
            ytDuration = event.target.getDuration() || 0;
            event.target.setVolume(ytVolume * 100);
            event.target.setPlaybackRate(ytRate);
            if (ytTime > 0) event.target.seekTo(ytTime, true);
            ytEmit('loadedmetadata'); ytEmit('progress'); ytEmit('timeupdate');
            setInterval(function(){ ytEmit('timeupdate'); ytEmit('progress'); }, 250);
          },
          onStateChange: function(event) {
            if (event.data === YT.PlayerState.PLAYING) { ytPaused = false; ytEmit('play'); }
            else if (event.data === YT.PlayerState.PAUSED) { ytPaused = true; ytEmit('pause'); }
            else if (event.data === YT.PlayerState.ENDED) { ytPaused = true; ytEmit('ended'); }
          }
        }
      });
    };
    var ytScript = document.createElement('script');
    ytScript.src = 'https://www.youtube.com/iframe_api';
    document.head.appendChild(ytScript);
  }

  var playBtn = $('playBtn'), muteBtn = $('muteBtn'), volumeInput = $('volume');
  var progressBar = $('progressBar'), buffered = $('buffered');
  var played = $('played'), thumb = $('thumb');
  var currentEl = $('current'), durationEl = $('duration');
  var centerPlay = $('centerPlay');
  var subOverlay = $('subOverlay'), subOrig = $('subOrig'), subTrans = $('subTrans');
  var subOrigLayer = $('subOrigLayer'), subTransLayer = $('subTransLayer');
  var subOrigDragHandle = $('subOrigDragHandle'), subTransDragHandle = $('subTransDragHandle');
  var noteBtn = $('noteBtn'), onlineNoteBtn = $('onlineNoteBtn');
  var subToggle = $('subToggle'), subtitleSettingsBtn = $('subtitleSettingsBtn');
  var onlineSubtitleSettingsBtn = $('onlineSubtitleSettingsBtn'), subtitleSettingsMenu = $('subtitleSettingsMenu');
  var settingsSubtitlesToggle = $('settingsSubtitlesToggle'), settingsSubtitlesState = $('settingsSubtitlesState');
  var settingsOriginalToggle = $('settingsOriginalToggle'), settingsOriginalState = $('settingsOriginalState');
  var settingsTranslationToggle = $('settingsTranslationToggle'), settingsTranslationState = $('settingsTranslationState');
  var settingsWordLookupToggle = $('settingsWordLookupToggle'), settingsWordLookupState = $('settingsWordLookupState');
  var settingsPositionReset = $('settingsPositionReset');
  var listBilingualPanelBtn = $('listBilingualPanelBtn');
  var speedBtn = $('speedBtn'), onlineSpeedBtn = $('onlineSpeedBtn'), speedMenu = $('speedMenu');
  var fsBtn = $('fsBtn');
  var panelRestoreBtn = $('panelRestoreBtn');
  var followBtn = $('followBtn'), followLabel = $('followLabel');
  var subtitleToolsBtn = $('subtitleToolsBtn'), subtitleTools = $('subtitleTools');
  var subtitleSearch = $('subtitleSearch'), subtitleSearchEmpty = $('subtitleSearchEmpty');
  var wordPopup = $('wordPopup'), popupWord = $('popupWord'), popupWordSave = $('popupWordSave');
  var popupLookup = $('popupLookup'), popupAiButton = $('popupAiButton');
  var subtitleOffsetInput = $('subtitleOffset'), subtitleOffsetValue = $('subtitleOffsetValue');
  var subtitleOffsetEarlier = $('subtitleOffsetEarlier'), subtitleOffsetLater = $('subtitleOffsetLater');
  var subtitleOffsetReset = $('subtitleOffsetReset');

  var subsOn = true, videoOriginalOn = true, videoTranslationOn = true, listBilingualOn = true;
  var WORD_LOOKUP_ENABLED_KEY = 'englishLearn.wordLookupEnabled';
  var wordLookupEnabled = true;
  var lastActiveId = -1, ctrlTimer, currentRate = 1, autoFollow = true;
  var hoverDictCache = {};
  var hoverIntentTimer = null, hoverIntentWord = null;
  var popupMode = 'lookup';

  function resetActiveHighlight() {
    // Seeking and timing changes invalidate the cached ID. Clear the visual
    // state at the same time so the next update cannot accumulate highlights.
    var highlightedRows = document.querySelectorAll('.sub-row.active');
    for (var i = 0; i < highlightedRows.length; i++) {
      highlightedRows[i].classList.remove('active');
    }
    lastActiveId = -1;
  }

  try { wordLookupEnabled = localStorage.getItem(WORD_LOOKUP_ENABLED_KEY) !== '0'; } catch (_) {}

  function setWordLookupEnabled(enabled, persist) {
    wordLookupEnabled = !!enabled;
    learningShell.classList.toggle('learning-word-lookup-off', !wordLookupEnabled);
    if (settingsWordLookupToggle) {
      settingsWordLookupToggle.classList.toggle('active', wordLookupEnabled);
      settingsWordLookupState.textContent = wordLookupEnabled ? '__SUBTITLE_ON_SHORT__' : '__SUBTITLE_OFF_SHORT__';
      settingsWordLookupToggle.setAttribute('aria-pressed', wordLookupEnabled ? 'true' : 'false');
    }
    if (!wordLookupEnabled) {
      clearTimeout(hoverIntentTimer);
      hoverIntentWord = null;
      document.querySelectorAll('.sub-word.word-hover-ready').forEach(function(el){ el.classList.remove('word-hover-ready'); });
      if (typeof hidePopup === 'function') hidePopup();
    }
    if (persist) { try { localStorage.setItem(WORD_LOOKUP_ENABLED_KEY, wordLookupEnabled ? '1' : '0'); } catch (_) {} }
  }
  setWordLookupEnabled(wordLookupEnabled, false);
  if (settingsWordLookupToggle) settingsWordLookupToggle.addEventListener('click', function(){ setWordLookupEnabled(!wordLookupEnabled, true); });

  // ---- Fullscreen split resizing ----
  var PANEL_DESKTOP_KEY = 'englishLearn.panelRatio.desktop';
  var PANEL_MOBILE_KEY = 'englishLearn.panelRatio.mobile';
  var desktopPanelRatio = readPanelRatio(PANEL_DESKTOP_KEY, 42, 24, 55);
  var mobilePanelRatio = readPanelRatio(PANEL_MOBILE_KEY, 50, 28, 65);
  var resizingPanel = false, resizePointerId = null;

  function isNarrowLayout() { return window.matchMedia('(max-width: 640px)').matches; }
  function clampPanelRatio(value, min, max) { return Math.max(min, Math.min(max, value)); }
  function readPanelRatio(key, fallback, min, max) {
    try {
      var value = parseFloat(localStorage.getItem(key));
      return Number.isFinite(value) ? clampPanelRatio(value, min, max) : fallback;
    } catch (_) { return fallback; }
  }
  function currentPanelRatio() { return isNarrowLayout() ? mobilePanelRatio : desktopPanelRatio; }
  function applyPanelRatio(value, persist) {
    var narrow = isNarrowLayout();
    var min = narrow ? 28 : 24, max = narrow ? 65 : 55;
    var ratio = Math.round(clampPanelRatio(value, min, max) * 10) / 10;
    if (narrow) {
      mobilePanelRatio = ratio;
      learningShell.style.setProperty('--panel-height', ratio + '%');
      panelResizer.setAttribute('aria-orientation', 'horizontal');
    } else {
      desktopPanelRatio = ratio;
      learningShell.style.setProperty('--panel-width', ratio + '%');
      panelResizer.setAttribute('aria-orientation', 'vertical');
    }
    panelResizer.setAttribute('aria-valuemin', String(min));
    panelResizer.setAttribute('aria-valuemax', String(max));
    panelResizer.setAttribute('aria-valuenow', String(Math.round(ratio)));
    panelResizer.setAttribute('aria-valuetext', '__PANEL_RESIZE_VALUE__ ' + Math.round(ratio) + '%');
    if (persist) {
      try { localStorage.setItem(narrow ? PANEL_MOBILE_KEY : PANEL_DESKTOP_KEY, String(ratio)); } catch (_) {}
    }
  }
  applyPanelRatio(currentPanelRatio(), false);

  // Video overlay and subtitle list are deliberately independent learner
  // preferences. The former shared value is only a migration fallback.
  var videoSubFontSize = $('videoSubFontSize'), videoSubFontValue = $('videoSubFontValue');
  var listSubFontSize = $('listSubFontSize'), listSubFontValue = $('listSubFontValue');
  var LEGACY_SUB_FONT_KEY = 'englishLearn.subtitleScale';
  var VIDEO_SUB_FONT_KEY = 'englishLearn.videoSubtitleScale';
  var LIST_SUB_FONT_KEY = 'englishLearn.listSubtitleScale';
  function readSubtitleScale(key, maximum) {
    try {
      var saved = localStorage.getItem(key);
      var fallback = saved === null ? parseFloat(localStorage.getItem(LEGACY_SUB_FONT_KEY)) : parseFloat(saved);
      return Number.isFinite(fallback) ? Math.max(.8, Math.min(maximum, fallback)) : 1;
    } catch (_) { return 1; }
  }
  var videoSubtitleScale = readSubtitleScale(VIDEO_SUB_FONT_KEY, 2);
  var listSubtitleScale = readSubtitleScale(LIST_SUB_FONT_KEY, 1.8);
  function applyVideoSubtitleScale(value, persist) {
    videoSubtitleScale = Math.max(.8, Math.min(2, Number(value) || 1));
    learningShell.style.setProperty('--video-subtitle-scale', String(videoSubtitleScale));
    if (videoSubFontSize) videoSubFontSize.value = String(Math.round(videoSubtitleScale * 100));
    if (videoSubFontValue) videoSubFontValue.textContent = Math.round(videoSubtitleScale * 100) + '%';
    if (persist) { try { localStorage.setItem(VIDEO_SUB_FONT_KEY, String(videoSubtitleScale)); } catch (_) {} }
  }
  function applyListSubtitleScale(value, persist) {
    listSubtitleScale = Math.max(.8, Math.min(1.8, Number(value) || 1));
    learningShell.style.setProperty('--list-subtitle-scale', String(listSubtitleScale));
    if (listSubFontSize) listSubFontSize.value = String(Math.round(listSubtitleScale * 100));
    if (listSubFontValue) listSubFontValue.textContent = Math.round(listSubtitleScale * 100) + '%';
    if (persist) { try { localStorage.setItem(LIST_SUB_FONT_KEY, String(listSubtitleScale)); } catch (_) {} }
  }
  applyVideoSubtitleScale(videoSubtitleScale, false);
  applyListSubtitleScale(listSubtitleScale, false);
  if (videoSubFontSize) videoSubFontSize.addEventListener('input', function(){ applyVideoSubtitleScale(Number(this.value) / 100, true); });
  if (listSubFontSize) listSubFontSize.addEventListener('input', function(){ applyListSubtitleScale(Number(this.value) / 100, true); });

  // Subtitle timing is project-specific. Positive values delay the subtitle;
  // negative values make it appear earlier. The slider updates immediately so
  // a learner can line it up against speech without pausing or reloading.
  var SUBTITLE_OFFSET_KEY = 'englishLearn.subtitleOffset.' + PROJECT_ID;
  var subtitleOffsetSaveTimer = null;
  function readSubtitleOffset() {
    try {
      var saved = localStorage.getItem(SUBTITLE_OFFSET_KEY);
      var value = saved === null ? Number(INITIAL_SUBTITLE_OFFSET) : parseFloat(saved);
      return Number.isFinite(value) ? Math.max(-5, Math.min(5, value)) : 0;
    } catch (_) {
      var initial = Number(INITIAL_SUBTITLE_OFFSET);
      return Number.isFinite(initial) ? Math.max(-5, Math.min(5, initial)) : 0;
    }
  }
  var subtitleOffset = readSubtitleOffset();
  function subtitleVideoTime(sourceTime) { return Math.max(0, Number(sourceTime || 0) + subtitleOffset); }
  function saveSubtitleOffset() {
    if (!PROJECT_ID || !LOOKUP_API_BASE) return;
    clearTimeout(subtitleOffsetSaveTimer);
    subtitleOffsetSaveTimer = setTimeout(function() {
      var url = LOOKUP_API_BASE + '/api/subtitle-offset?project_id=' + encodeURIComponent(PROJECT_ID) +
        '&offset=' + encodeURIComponent(subtitleOffset.toFixed(1));
      fetchJsonWithin(url, 1200);
    }, 350);
  }
  function applySubtitleOffset(value, persist) {
    subtitleOffset = Math.round(Math.max(-5, Math.min(5, Number(value) || 0)) * 10) / 10;
    if (subtitleOffsetInput) {
      subtitleOffsetInput.value = subtitleOffset.toFixed(1);
      subtitleOffsetInput.setAttribute('aria-valuetext', (subtitleOffset > 0 ? '+' : '') + subtitleOffset.toFixed(1) + 's');
    }
    if (subtitleOffsetValue) subtitleOffsetValue.textContent = (subtitleOffset > 0 ? '+' : '') + subtitleOffset.toFixed(1) + 's';
    if (persist) {
      try { localStorage.setItem(SUBTITLE_OFFSET_KEY, subtitleOffset.toFixed(1)); } catch (_) {}
      saveSubtitleOffset();
    }
    // A forced refresh changes both the video overlay and the active list row
    // at the current video frame, without moving the video playhead.
    resetActiveHighlight();
    if (typeof subsContainer !== 'undefined' && subsContainer) updateActive();
  }
  applySubtitleOffset(subtitleOffset, false);
  if (subtitleOffsetInput) subtitleOffsetInput.addEventListener('input', function(){ applySubtitleOffset(this.value, true); });
  if (subtitleOffsetEarlier) subtitleOffsetEarlier.addEventListener('click', function(){ applySubtitleOffset(subtitleOffset - .1, true); });
  if (subtitleOffsetLater) subtitleOffsetLater.addEventListener('click', function(){ applySubtitleOffset(subtitleOffset + .1, true); });
  if (subtitleOffsetReset) subtitleOffsetReset.addEventListener('click', function(){ applySubtitleOffset(0, true); });

  // Original and translation have separate, persistent layers. A learner can
  // keep the English line at eye level and move the translation elsewhere.
  // The former shared vertical value is used only to migrate existing setups.
  var LEGACY_SUB_POSITION_KEY = 'englishLearn.videoSubtitleY';
  var VIDEO_ORIGINAL_X_KEY = 'englishLearn.videoSubtitleOriginalX';
  var VIDEO_ORIGINAL_Y_KEY = 'englishLearn.videoSubtitleOriginalY';
  var VIDEO_TRANSLATION_X_KEY = 'englishLearn.videoSubtitleTranslationX';
  var VIDEO_TRANSLATION_Y_KEY = 'englishLearn.videoSubtitleTranslationY';
  var legacySubtitleY = readPanelRatio(LEGACY_SUB_POSITION_KEY, 80, 8, 92);
  var videoOriginalPosition = {
    x: readPanelRatio(VIDEO_ORIGINAL_X_KEY, 50, 5, 95),
    y: readPanelRatio(VIDEO_ORIGINAL_Y_KEY, legacySubtitleY, 8, 92)
  };
  var videoTranslationPosition = {
    x: readPanelRatio(VIDEO_TRANSLATION_X_KEY, 50, 5, 95),
    y: readPanelRatio(VIDEO_TRANSLATION_Y_KEY, Math.max(8, legacySubtitleY - 10), 8, 92)
  };
  function resetVideoSubtitlePositions(persist) {
    videoOriginalPosition.x = 50; videoOriginalPosition.y = 80;
    videoTranslationPosition.x = 50; videoTranslationPosition.y = 70;
    applyVideoSubtitlePosition('original', videoOriginalPosition, persist);
    applyVideoSubtitlePosition('translation', videoTranslationPosition, persist);
  }
  var draggingSubtitle = null, subtitlePointerId = null;
  function applyVideoSubtitlePosition(kind, position, persist) {
    var target = kind === 'original' ? videoOriginalPosition : videoTranslationPosition;
    target.x = Math.round(clampPanelRatio(position.x, 5, 95) * 10) / 10;
    target.y = Math.round(clampPanelRatio(position.y, 8, 92) * 10) / 10;
    var prefix = kind === 'original' ? '--video-subtitle-original-' : '--video-subtitle-translation-';
    learningShell.style.setProperty(prefix + 'x', target.x + '%');
    learningShell.style.setProperty(prefix + 'y', target.y + '%');
    if (persist) {
      var keys = kind === 'original' ? [VIDEO_ORIGINAL_X_KEY, VIDEO_ORIGINAL_Y_KEY] : [VIDEO_TRANSLATION_X_KEY, VIDEO_TRANSLATION_Y_KEY];
      try { localStorage.setItem(keys[0], String(target.x)); localStorage.setItem(keys[1], String(target.y)); } catch (_) {}
    }
  }
  function updateVideoSubtitlePosition(e) {
    if (!draggingSubtitle) return;
    var rect = player.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    applyVideoSubtitlePosition(draggingSubtitle, {
      x: (e.clientX - rect.left) / rect.width * 100,
      y: (e.clientY - rect.top) / rect.height * 100
    }, true);
  }
  function attachSubtitleDrag(kind, handle, layer, position) {
    if (!handle) return;
    handle.addEventListener('pointerdown', function(e) {
      draggingSubtitle = kind; subtitlePointerId = e.pointerId;
      layer.classList.add('is-dragging');
      handle.setPointerCapture(e.pointerId);
      updateVideoSubtitlePosition(e);
      e.preventDefault();
    });
    handle.addEventListener('keydown', function(e) {
      var next = { x: position.x, y: position.y }, handled = true;
      if (e.key === 'ArrowUp') next.y -= 3;
      else if (e.key === 'ArrowDown') next.y += 3;
      else if (e.key === 'ArrowLeft') next.x -= 3;
      else if (e.key === 'ArrowRight') next.x += 3;
      else if (e.key === 'Home') next.y = 8;
      else if (e.key === 'End') next.y = 92;
      else handled = false;
      if (handled) { applyVideoSubtitlePosition(kind, next, true); e.preventDefault(); }
    });
  }
  applyVideoSubtitlePosition('original', videoOriginalPosition, false);
  applyVideoSubtitlePosition('translation', videoTranslationPosition, false);
  attachSubtitleDrag('original', subOrigDragHandle, subOrigLayer, videoOriginalPosition);
  attachSubtitleDrag('translation', subTransDragHandle, subTransLayer, videoTranslationPosition);
  if (settingsPositionReset) settingsPositionReset.addEventListener('click', function(){
    resetVideoSubtitlePositions(true);
  });
  document.addEventListener('pointermove', function(e) {
    if (draggingSubtitle && e.pointerId === subtitlePointerId) updateVideoSubtitlePosition(e);
  });
  function finishSubtitleDrag(e) {
    if (!draggingSubtitle || (e.pointerId !== undefined && e.pointerId !== subtitlePointerId)) return;
    var handle = draggingSubtitle === 'original' ? subOrigDragHandle : subTransDragHandle;
    var layer = draggingSubtitle === 'original' ? subOrigLayer : subTransLayer;
    layer.classList.remove('is-dragging');
    try { handle.releasePointerCapture(subtitlePointerId); } catch (_) {}
    draggingSubtitle = null; subtitlePointerId = null;
  }
  document.addEventListener('pointerup', finishSubtitleDrag);
  document.addEventListener('pointercancel', finishSubtitleDrag);

  function updatePanelRatioFromPointer(e) {
    var rect = learningShell.getBoundingClientRect();
    var ratio = isNarrowLayout()
      ? ((rect.bottom - e.clientY) / rect.height * 100)
      : ((rect.right - e.clientX) / rect.width * 100);
    if (ratio <= 10) {
      togglePanel(true);
      return;
    }
    togglePanel(false);
    applyPanelRatio(ratio, true);
  }
  panelResizer.addEventListener('pointerdown', function(e) {
    if (learningShell.classList.contains('panel-collapsed')) return;
    resizingPanel = true; resizePointerId = e.pointerId;
    learningShell.classList.add('is-resizing');
    panelResizer.setPointerCapture(e.pointerId);
    updatePanelRatioFromPointer(e);
    e.preventDefault();
  });
  document.addEventListener('pointermove', function(e) {
    if (resizingPanel && e.pointerId === resizePointerId) updatePanelRatioFromPointer(e);
  });
  function finishPanelResize(e) {
    if (!resizingPanel || (e.pointerId !== undefined && e.pointerId !== resizePointerId)) return;
    resizingPanel = false;
    learningShell.classList.remove('is-resizing');
    try { panelResizer.releasePointerCapture(resizePointerId); } catch (_) {}
    resizePointerId = null;
  }
  document.addEventListener('pointerup', finishPanelResize);
  document.addEventListener('pointercancel', finishPanelResize);
  panelResizer.addEventListener('dblclick', function() {
    applyPanelRatio(isNarrowLayout() ? 50 : 42, true);
  });
  panelResizer.addEventListener('keydown', function(e) {
    var narrow = isNarrowLayout();
    var value = currentPanelRatio(), minimum = narrow ? 28 : 24, handled = true;
    if (e.key === 'Home') value = narrow ? 28 : 24;
    else if (e.key === 'End') value = narrow ? 65 : 55;
    else if ((!narrow && e.key === 'ArrowLeft') || (narrow && e.key === 'ArrowUp')) value += 2;
    else if ((!narrow && e.key === 'ArrowRight') || (narrow && e.key === 'ArrowDown')) {
      if (value <= minimum) {
        togglePanel(true);
        e.preventDefault();
        return;
      }
      value -= 2;
    }
    else handled = false;
    if (handled) { applyPanelRatio(value, true); e.preventDefault(); }
  });
  window.addEventListener('resize', function(){ applyPanelRatio(currentPanelRatio(), false); });

  // ---- Action sender (iframe -> Streamlit via URL query params) ----
  function sendAction(action, data) {
    var url = new URL(window.top.location.href);
    url.searchParams.set('action', action);
    for (var key in data) {
      if (data[key] !== undefined && data[key] !== null)
        url.searchParams.set(key, String(data[key]));
    }
    // Streamlit's iframe sandbox intentionally omits allow-top-navigation.
    // Updating the same-origin parent history first keeps star/word actions
    // working in the real player; the direct navigation remains a fallback
    // for hosts that expose a less restrictive iframe.
    try {
      var host = window.parent;
      host.history.pushState({}, '', url.toString());
      host.location.reload();
      return;
    } catch (_) {}
    try { window.top.location.href = url.toString(); } catch (_) { window.location.href = url.toString(); }
  }

  // ---- Format time ----
  function fmt(t) {
    t = Math.max(0, t);
    var h = Math.floor(t/3600), m = Math.floor((t%3600)/60), s = Math.floor(t%60);
    var str = m + ':' + (s<10?'0':'') + s;
    return h > 0 ? h + ':' + (m<10?'0':'') + str : str;
  }

  function esc(str) { var d = document.createElement('div'); d.appendChild(document.createTextNode(str||'')); return d.innerHTML; }

  // ===========================
  // RENDER: Subtitles tab
  // ===========================
  var subsContainer = $('content-subs');
  function setVideoOriginal(enabled) {
    videoOriginalOn = !!enabled;
    learningShell.classList.toggle('learning-video-original-off', !videoOriginalOn);
    if (settingsOriginalToggle) {
      settingsOriginalToggle.classList.toggle('active', videoOriginalOn);
      settingsOriginalState.textContent = videoOriginalOn ? '__SUBTITLE_ON_SHORT__' : '__SUBTITLE_OFF_SHORT__';
      settingsOriginalToggle.title = videoOriginalOn ? '__VIDEO_ORIGINAL_TITLE__' : '__VIDEO_ORIGINAL_OFF_TITLE__';
    }
  }
  function setVideoTranslation(enabled) {
    videoTranslationOn = !!enabled;
    learningShell.classList.toggle('learning-video-translation-off', !videoTranslationOn);
    if (settingsTranslationToggle) {
      settingsTranslationToggle.classList.toggle('active', videoTranslationOn);
      settingsTranslationState.textContent = videoTranslationOn ? '__SUBTITLE_ON_SHORT__' : '__SUBTITLE_OFF_SHORT__';
      settingsTranslationToggle.title = videoTranslationOn ? '__VIDEO_TRANSLATION_TITLE__' : '__VIDEO_TRANSLATION_OFF_TITLE__';
    }
  }
  function setListBilingual(enabled) {
    listBilingualOn = !!enabled;
    learningShell.classList.toggle('learning-list-bilingual-off', !listBilingualOn);
    if (listBilingualPanelBtn) {
      listBilingualPanelBtn.classList.toggle('active', listBilingualOn);
      listBilingualPanelBtn.textContent = listBilingualOn ? '__LIST_BILINGUAL_ON__' : '__LIST_BILINGUAL_OFF__';
      listBilingualPanelBtn.title = listBilingualOn ? '__LIST_BILINGUAL_TITLE__' : '__LIST_BILINGUAL_OFF_TITLE__';
    }
  }
  setVideoOriginal(true);
  setVideoTranslation(true);
  setListBilingual(true);
  if (settingsOriginalToggle) settingsOriginalToggle.addEventListener('click', function(){ setVideoOriginal(!videoOriginalOn); });
  if (settingsTranslationToggle) settingsTranslationToggle.addEventListener('click', function(){ setVideoTranslation(!videoTranslationOn); });
  if (listBilingualPanelBtn) listBilingualPanelBtn.addEventListener('click', function(){ setListBilingual(!listBilingualOn); });

  function hoverMarkup(text) {
    var raw = String(text || '');
    var out = '', last = 0, re = /[A-Za-z][A-Za-z'-]*/g, match;
    while ((match = re.exec(raw))) {
      out += esc(raw.substring(last, match.index));
      out += '<span class="sub-word" data-word="' + esc(match[0]) + '">' + esc(match[0]) + '</span>';
      last = match.index + match[0].length;
    }
    return out + esc(raw.substring(last));
  }

  function showQuickLookup(message) {
    popupLookup.textContent = message || '';
  }

  function subtitleForWord(wordEl) {
    var row = wordEl.closest('.sub-row');
    var sub = row && subtitles.find(function(item){ return String(item.id) === String(row.dataset.subId); });
    if (!sub && wordEl.closest('.sub-overlay')) {
      sub = subtitles.find(function(item){ return String(item.id) === String(lastActiveId); });
    }
    return sub || null;
  }

  function normaliseAudioUrl(value) {
    var url = String(value || '').trim();
    if (url.startsWith('//')) url = 'https:' + url;
    return url.toLowerCase().indexOf('https://') === 0 ? url : '';
  }

  function localLookupModel(entry) {
    return {
      translations: [entry.translation].concat(entry.other_meanings || []).filter(Boolean),
      phonetics: entry.pronunciation ? [{
        text: entry.pronunciation,
        audio: normaliseAudioUrl(entry.audio),
        speech: true,
        label: ''
      }] : [{ text: '', audio: '', speech: true, label: '' }],
      meanings: [],
      source: '__WORDBOOK_SOURCE__'
    };
  }

  function dictionaryLookupModel(result) {
    var model = { translations: [], phonetics: [], meanings: [], source: '' };
    var local = result && result.local;
    if (local) {
      model.translations = (local.translations || []).slice(0, 12);
      (local.definitions || []).slice(0, 8).forEach(function(definition) {
        model.meanings.push({ pos: local.pos || '', definition: definition, example: '' });
      });
      if (local.phonetic) model.phonetics.push({ text: local.phonetic, audio: '', speech: true, label: '' });
      model.source = local.source || 'ECDICT';
    }
    if (result && result.translation && model.translations.indexOf(result.translation) < 0) {
      model.translations.unshift(result.translation);
    }
    var entries = Array.isArray(result && result.dictionary) ? result.dictionary : [];
    entries.forEach(function(entry) {
      (entry.phonetics || []).forEach(function(item) {
        if (!item || (!item.text && !item.audio)) return;
        var audio = normaliseAudioUrl(item.audio);
        var label = /(?:_gb_|uk|british)/i.test(audio) ? '__PHONETIC_UK__' :
          (/(?:_us_|american)/i.test(audio) ? '__PHONETIC_US__' : '');
        var candidate = {
          text: String(item.text || entry.phonetic || '').replace(/^[/]+|[/]+$/g, ''),
          audio: audio,
          speech: true,
          label: label
        };
        var sameText = model.phonetics.find(function(existing) {
          return existing.text && existing.text === candidate.text;
        });
        if (sameText && candidate.audio && !sameText.audio) {
          sameText.audio = candidate.audio;
          sameText.label = candidate.label;
        } else {
          var duplicate = model.phonetics.some(function(existing) {
            return existing.text === candidate.text && existing.audio === candidate.audio;
          });
          if (!duplicate) model.phonetics.push(candidate);
        }
      });
      if (!model.phonetics.length && entry.phonetic) {
        model.phonetics.push({
          text: String(entry.phonetic).replace(/^[/]+|[/]+$/g, ''),
          audio: '',
          speech: true,
          label: ''
        });
      }
      (entry.meanings || []).forEach(function(meaning) {
        (meaning.definitions || []).slice(0, 4).forEach(function(item) {
          if (item && item.definition && model.meanings.length < 12) {
            model.meanings.push({
              pos: meaning.partOfSpeech || '',
              definition: item.definition,
              example: item.example || ''
            });
          }
        });
      });
    });
    if (!model.phonetics.length) {
      model.phonetics.push({ text: '', audio: '', speech: true, label: '' });
    }
    if (!model.source && entries.length) model.source = 'Free Dictionary API';
    return model;
  }

  function renderLookupModel(model) {
    if (!model) { showQuickLookup('__HOVER_NO_RESULT__'); return; }
    var html = '';
    if (model.phonetics && model.phonetics.length) {
      html += '<div class="lookup-phonetics">';
      model.phonetics.slice(0, 4).forEach(function(item, index) {
        var label = item.label || (index === 0 ? '__PHONETIC__' : '');
        html += '<span class="lookup-phonetic">' +
          (label ? '<span>' + esc(label) + '</span>' : '') +
          (item.text ? '<span>/' + esc(item.text) + '/</span>' : '') +
          ((item.audio || item.speech) ? '<button type="button" class="lookup-audio" data-audio="' + esc(item.audio || '') +
            '" data-speech="' + esc((popupData && popupData.text) || '') +
            '" data-locale="' + esc(item.label === '__PHONETIC_UK__' ? 'en-GB' : 'en-US') +
            '" aria-label="__PLAY_PRONUNCIATION__">&#128266;</button>' : '') +
          '</span>';
      });
      html += '</div>';
    }
    if (model.translations && model.translations.length) {
      html += '<div class="lookup-translations">';
      model.translations.slice(0, 8).forEach(function(item) {
        html += '<div class="lookup-translation">' + esc(item) + '</div>';
      });
      html += '</div>';
    }
    (model.meanings || []).slice(0, 8).forEach(function(item) {
      html += '<div class="lookup-meaning">' +
        (item.pos ? '<span class="lookup-pos">' + esc(item.pos) + '.</span>' : '') +
        esc(item.definition || '') +
        (item.example ? '<div class="lookup-example">' + esc(item.example) + '</div>' : '') +
        '</div>';
    });
    if (model.source) html += '<div class="lookup-source">' + esc(model.source) + '</div>';
    popupLookup.innerHTML = html || esc('__HOVER_NO_RESULT__');
  }

  function wordSavePayload(model, sub, word) {
    var translations = (model && model.translations || []).filter(Boolean);
    var meanings = (model && model.meanings || []).map(function(item) {
      return (item.pos ? item.pos + '. ' : '') + item.definition;
    }).filter(Boolean);
    var phonetic = model && model.phonetics && model.phonetics.find(function(item){ return item.text; });
    var audio = model && model.phonetics && model.phonetics.find(function(item){ return item.audio; });
    var pos = [];
    (model && model.meanings || []).forEach(function(item) {
      if (item.pos && pos.indexOf(item.pos) < 0) pos.push(item.pos);
    });
    return {
      word: word,
      translation: translations[0] || '',
      pos: pos.join('/'),
      pronunciation: phonetic ? phonetic.text : '',
      other_meanings: translations.slice(1).concat(meanings).slice(0, 12),
      audio: audio ? audio.audio : '',
      context: sub ? (sub.text || '') : '',
      time: sub ? sub.start : 0,
      subtitle_id: sub ? sub.id : '',
      project_id: PROJECT_ID
    };
  }

  // Public APIs are accessed through a localhost bridge. Direct iframe calls
  // are unreliable because browser CORS policies can reject them even when the
  // same machine can reach the services.
  function fetchJsonWithin(url, timeoutMs, options) {
    var controller = new AbortController();
    var timer = setTimeout(function(){ controller.abort(); }, timeoutMs);
    var requestOptions = Object.assign({ credentials:'omit' }, options || {});
    requestOptions.signal = controller.signal;
    return fetch(url, requestOptions)
      .then(function(resp){ return resp.ok ? resp.json() : null; })
      .catch(function(){ return null; })
      .finally(function(){ clearTimeout(timer); });
  }

  function postCollection(path, payload) {
    if (!LOOKUP_API_BASE) return Promise.resolve(null);
    // A Streamlit srcdoc iframe has an opaque origin. Some embedded browsers
    // block fetch/form/script writes to a random loopback port. Image resources
    // use the same permitted path as the video stream, so a one-pixel response
    // can confirm a local write without reading cross-origin response data.
    return new Promise(function(resolve) {
      var requestId = 'collection-' + Date.now() + '-' + Math.random().toString(36).slice(2);
      var timer = setTimeout(function() {
        delete collectionPending[requestId];
        resolve(null);
      }, 1800);
      var image = new Image(1, 1);
      collectionPending[requestId] = { resolve: resolve, timer: timer, image: image };
      image.onload = function() {
        var pending = collectionPending[requestId];
        if (!pending) return;
        clearTimeout(pending.timer);
        delete collectionPending[requestId];
        if (path === '/api/word/save') {
          pending.resolve({ ok: true, saved: true, entry: Object.assign({}, payload) });
        } else {
          pending.resolve({ ok: true });
        }
      };
      image.onerror = function() {
        var pending = collectionPending[requestId];
        if (!pending) return;
        clearTimeout(pending.timer);
        delete collectionPending[requestId];
        pending.resolve(null);
      };
      image.src = LOOKUP_API_BASE + '/api/collection-pixel?action=' +
        encodeURIComponent(path) + '&request_id=' + encodeURIComponent(requestId) +
        '&payload=' + encodeURIComponent(JSON.stringify(payload || {})) +
        '&cache=' + encodeURIComponent(Date.now());
    });
  }

  var collectionPending = {};

  function fetchFreeTranslation(text) {
    var source = String(text || '').trim().slice(0, 900);
    if (!source || !LOOKUP_API_BASE) return Promise.resolve('');
    var url = LOOKUP_API_BASE + '/api/free-translate?text=' + encodeURIComponent(source) +
      '&target=' + encodeURIComponent(FREE_TRANSLATE_TARGET);
    return fetchJsonWithin(url, 1900).then(function(data) {
      var translated = data && data.translation;
      return translated && String(translated).trim() ? String(translated).trim() : '';
    });
  }

  function setPopupWordSaved(entry) {
    if (!popupWordSave) return;
    popupWordSave.disabled = false;
    popupWordSave.classList.toggle('faved', !!entry);
    popupWordSave.textContent = entry ? '★' : '☆';
    popupWordSave.title = entry ? '__WORD_REMOVE__' : '__WORD_SAVE__';
    popupWordSave.setAttribute('aria-label', entry ? '__WORD_REMOVE__' : '__WORD_SAVE__');
  }

  var pronunciationAudio = null;
  popupLookup.addEventListener('click', function(e) {
    var button = e.target.closest && e.target.closest('.lookup-audio');
    if (!button) return;
    var audioUrl = normaliseAudioUrl(button.dataset.audio);
    if (!audioUrl) {
      var speechText = String(button.dataset.speech || '').trim();
      if (!speechText || !window.speechSynthesis || !window.SpeechSynthesisUtterance) return;
      window.speechSynthesis.cancel();
      var utterance = new SpeechSynthesisUtterance(speechText);
      utterance.lang = button.dataset.locale || 'en-US';
      button.classList.add('playing');
      utterance.addEventListener('end', function(){ button.classList.remove('playing'); });
      utterance.addEventListener('error', function(){ button.classList.remove('playing'); });
      window.speechSynthesis.speak(utterance);
      return;
    }
    if (pronunciationAudio) {
      pronunciationAudio.pause();
      document.querySelectorAll('.lookup-audio.playing').forEach(function(item){ item.classList.remove('playing'); });
    }
    pronunciationAudio = new Audio(audioUrl);
    button.classList.add('playing');
    pronunciationAudio.addEventListener('ended', function(){ button.classList.remove('playing'); });
    pronunciationAudio.addEventListener('error', function(){ button.classList.remove('playing'); });
    pronunciationAudio.play().catch(function(){ button.classList.remove('playing'); });
  });

  function openDictionaryLookup(word, sub, anchorOrRect) {
    var normalized = String(word || '').toLowerCase();
    if (!normalized) return;
    var local = wordbook.find(function(item) {
      return String(item.word || '').toLowerCase() === normalized &&
        String(item.project_id || '') === String(PROJECT_ID);
    });
    popupMode = 'lookup';
    popupData = { sub: sub, text: word, lookupKey: normalized, savedEntry: local || null, lookupModel: null };
    popupWord.textContent = word;
    setPopupWordSaved(local || null);
    popupWordSave.disabled = !local;
    popupAiButton.disabled = false;
    popupAiButton.textContent = '__AI_TRANSLATE__';
    showQuickLookup('__HOVER_LOOKUP__');
    var anchorRect = anchorOrRect && anchorOrRect.getBoundingClientRect
      ? anchorOrRect.getBoundingClientRect() : anchorOrRect;
    if (!anchorRect) {
      anchorRect = { top: window.innerHeight * 0.38, bottom: window.innerHeight * 0.38, left: window.innerWidth / 2, width: 0 };
    }
    showPopupNearRect(anchorRect);
    if (initialAiLookup && String(initialAiLookup.word || '').toLowerCase() === normalized && initialAiLookup.translation) {
      var restoredModel = localLookupModel(initialAiLookup);
      hoverDictCache[normalized] = restoredModel;
      popupData.lookupModel = restoredModel;
      initialAiLookup = null;
      renderLookupModel(restoredModel);
    }
    var cached = hoverDictCache[normalized];
    if (cached) {
      popupData.lookupModel = cached;
      renderLookupModel(cached);
      popupWordSave.disabled = false;
      return;
    }
    if (local) {
      popupData.lookupModel = localLookupModel(local);
      renderLookupModel(popupData.lookupModel);
    }
    if (!/^[a-z][a-z'-]{1,40}$/i.test(word)) { showQuickLookup('__HOVER_AI_HINT__'); return; }
    if (!LOOKUP_API_BASE) { showQuickLookup('__FREE_TRANSLATION_FAILED__'); return; }
    var lookupUrl = LOOKUP_API_BASE + '/api/word-lookup?word=' + encodeURIComponent(word) +
      '&target=' + encodeURIComponent(FREE_TRANSLATE_TARGET);
    fetchJsonWithin(lookupUrl, 1900).then(function(result) {
      var model = dictionaryLookupModel(result || {});
      var hasContent = model.translations.length || model.phonetics.length || model.meanings.length;
      if (!hasContent && local) model = localLookupModel(local);
      hoverDictCache[normalized] = model;
      if (popupData && popupData.lookupKey === normalized) {
        popupData.lookupModel = model;
        renderLookupModel(model);
        popupWordSave.disabled = false;
      }
    });
  }

  popupWordSave.addEventListener('click', function() {
    if (!popupData || popupWordSave.disabled) return;
    popupWordSave.disabled = true;
    var savedEntry = popupData.savedEntry;
    if (savedEntry) {
    postCollection('/api/word/delete', {
      id: savedEntry.id,
      project_id: savedEntry.project_id || PROJECT_ID,
      word: savedEntry.word || popupData.word
    }).then(function(result) {
        if (!result || !result.ok || !popupData) return;
        wordbook = wordbook.filter(function(item){ return item.id !== savedEntry.id; });
        popupData.savedEntry = null;
        setPopupWordSaved(null);
        renderWordbook();
      }).finally(function(){ if (popupData) popupWordSave.disabled = false; });
      return;
    }
    var payload = wordSavePayload(popupData.lookupModel, popupData.sub, popupData.text);
    postCollection('/api/word/save', payload).then(function(result) {
      if (!result || !result.ok || !result.entry || !popupData) return;
      wordbook = wordbook.filter(function(item) {
        return !(String(item.project_id) === String(result.entry.project_id) &&
          String(item.word || '').toLowerCase() === String(result.entry.word || '').toLowerCase());
      });
      wordbook.push(result.entry);
      popupData.savedEntry = result.entry;
      setPopupWordSaved(result.entry);
      renderWordbook();
    }).finally(function(){ if (popupData) popupWordSave.disabled = false; });
  });

  popupAiButton.addEventListener('click', function() {
    if (!popupData || !popupData.sub || popupAiButton.disabled) return;
    popupAiButton.disabled = true;
    popupAiButton.textContent = '__POPUP_TRANSLATING__';
    showQuickLookup('__POPUP_TRANSLATING__');
    var sub = popupData.sub;
    // Keep contextual translation inside the learning iframe. The old query
    // parameter bridge reloaded Streamlit and therefore exited immersive mode.
    fetchFreeTranslation(sub.text || popupData.text).then(function(translation) {
      if (!popupData || popupData.sub !== sub) return;
      popupAiButton.disabled = false;
      popupAiButton.textContent = '__AI_TRANSLATE__';
      showQuickLookup(translation ? '__FREE_TRANSLATION_LABEL__ ' + translation : '__FREE_TRANSLATION_FAILED__');
    });
  });

  var HOVER_INTENT_DELAY = 200;
  function handleWordOver(e) {
    if (!wordLookupEnabled) return;
    var wordEl = e.target.closest && e.target.closest('.sub-word');
    if (wordEl && (!e.relatedTarget || !wordEl.contains(e.relatedTarget))) {
      clearTimeout(hoverIntentTimer);
      hoverIntentWord = wordEl;
      hoverIntentTimer = setTimeout(function(){
        if (hoverIntentWord === wordEl) wordEl.classList.add('word-hover-ready');
      }, HOVER_INTENT_DELAY);
    }
  }
  document.addEventListener('pointerover', handleWordOver);
  function handleWordOut(e) {
    var wordEl = e.target.closest && e.target.closest('.sub-word');
    if (wordEl && (!e.relatedTarget || !wordEl.contains(e.relatedTarget))) {
      if (hoverIntentWord === wordEl) hoverIntentWord = null;
      clearTimeout(hoverIntentTimer);
      wordEl.classList.remove('word-hover-ready');
    }
  }
  document.addEventListener('pointerout', handleWordOut);
  document.addEventListener('click', function(e) {
    if (!wordLookupEnabled) return;
    var wordEl = e.target.closest && e.target.closest('.sub-word');
    if (!wordEl || window.getSelection().toString().trim()) return;
    var sub = subtitleForWord(wordEl);
    if (!sub) return;
    e.preventDefault();
    e.stopPropagation();
    wordEl.classList.remove('word-hover-ready');
    openDictionaryLookup(wordEl.dataset.word, sub, wordEl);
  });

  function setAutoFollow(enabled, centerNow) {
    autoFollow = !!enabled;
    followBtn.classList.toggle('active', autoFollow);
    followBtn.setAttribute('aria-pressed', autoFollow ? 'true' : 'false');
    followBtn.title = autoFollow ? '__FOLLOW_ON__' : '__FOLLOW_RESUME__';
    followLabel.textContent = autoFollow ? '__FOLLOW_ON__' : '__FOLLOW_RESUME__';
    if (autoFollow && centerNow) {
      resetActiveHighlight();
      updateActive();
    }
  }

  function pauseAutoFollow() { setAutoFollow(false, false); }
  subsContainer.addEventListener('wheel', pauseAutoFollow, {passive:true});
  subsContainer.addEventListener('touchstart', pauseAutoFollow, {passive:true});
  subsContainer.addEventListener('pointerdown', pauseAutoFollow);
  subsContainer.addEventListener('keydown', function(e) {
    if (['PageDown','PageUp','Home','End','ArrowDown','ArrowUp'].indexOf(e.key) >= 0) pauseAutoFollow();
  });
  followBtn.addEventListener('click', function(){ setAutoFollow(true, true); });

  function setSubtitleToolsOpen(open, focusSearch) {
    open = !!open;
    subtitleTools.classList.toggle('show', open);
    subtitleToolsBtn.classList.toggle('active', open);
    subtitleToolsBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open && focusSearch) setTimeout(function(){ subtitleSearch.focus(); }, 0);
  }
  function filterSubtitleRows() {
    var query = subtitleSearch.value.trim().toLocaleLowerCase();
    var visible = 0;
    document.querySelectorAll('#content-subs .sub-row').forEach(function(row) {
      var match = !query || (row.dataset.searchText || '').indexOf(query) >= 0;
      row.classList.toggle('search-hidden', !match);
      if (match) visible += 1;
    });
    subtitleSearchEmpty.classList.toggle('show', Boolean(query) && visible === 0);
  }
  subtitleToolsBtn.addEventListener('click', function(){
    switchTab('subs');
    setSubtitleToolsOpen(!subtitleTools.classList.contains('show'), true);
  });
  subtitleSearch.addEventListener('input', filterSubtitleRows);
  subtitleSearch.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      subtitleSearch.value = '';
      filterSubtitleRows();
      setSubtitleToolsOpen(false, false);
      subsContainer.focus();
    }
  });
  document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key.toLocaleLowerCase() === 'f') {
      e.preventDefault();
      switchTab('subs');
      setSubtitleToolsOpen(true, true);
    }
  });
  if (!subtitles.length) {
    subsContainer.innerHTML = '<div class="empty-hint">__NO_SUBTITLES__</div>';
  } else {
    subtitles.forEach(function(sub) {
      var row = document.createElement('div');
      row.className = 'sub-row';
      row.id = 'sub-' + sub.id;
      row.dataset.start = sub.start;
      row.dataset.subId = sub.id;
      row.dataset.searchText = ((sub.text || '') + ' ' + (sub.translation || '')).toLocaleLowerCase();

      var secs = Math.floor(sub.start), mins = Math.floor(secs/60), s = secs%60;
      row.innerHTML =
        '<div class="sub-time">' + mins + ':' + (s<10?'0':'') + s + '</div>' +
        '<div class="sub-content">' +
          '<div class="sub-original">' + hoverMarkup(sub.text) + '</div>' +
          '<div class="sub-translation">' + esc(sub.translation||'') + '</div>' +
        '</div>' +
        '<button type="button" class="note-row-btn" aria-label="__NOTE_FROM_SUBTITLE__" title="__NOTE_FROM_SUBTITLE__" data-sub-id="' + sub.id + '">&#9998;</button>';

      row.addEventListener('click', function(e) {
        if (e.target.classList.contains('note-row-btn')) return;
        if (wordLookupEnabled && e.target.closest && e.target.closest('.sub-word')) return;
        if (window.getSelection().toString().trim()) return; // don't seek when selecting text
        setAutoFollow(true, false);
        video.currentTime = subtitleVideoTime(sub.start);
      });

      var noteRowBtn = row.querySelector('.note-row-btn');
      noteRowBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        video.pause();
        sendAction('open_note', {
          subtitle_id: sub.id,
          text: sub.text,
          translation: sub.translation || '',
          time: sub.start,
          project_id: PROJECT_ID
        });
      });

      subsContainer.appendChild(row);
    });
    subsContainer.appendChild(subtitleSearchEmpty);
  }

  // ===========================
  // RENDER: Project notes tab
  // ===========================
  function renderNotes() {
    var container = $('content-notes');
    var projectNotes = notes.filter(function(note){ return note.project_id === PROJECT_ID; });
    $('noteCount').textContent = projectNotes.length ? '(' + projectNotes.length + ')' : '';

    if (!projectNotes.length) {
      container.innerHTML = '<div class="empty-hint">__NOTE_EMPTY__</div>';
      return;
    }
    container.innerHTML = '';
    projectNotes.slice().reverse().forEach(function(note) {
      var row = document.createElement('div');
      row.className = 'item-row';
      var secs = Math.floor(note.time||0), mins = Math.floor(secs/60), s = secs%60;
      row.innerHTML =
        '<div class="item-time">' + mins + ':' + (s<10?'0':'') + s + '</div>' +
        '<div class="item-main">' +
          '<div class="item-word">' + esc(note.title || '__UNTITLED_NOTE__') + '</div>' +
          '<div class="item-text">' + esc(note.summary || note.body || '') + '</div>' +
        '</div>';
      row.querySelector('.item-main').addEventListener('click', function() {
        video.currentTime = subtitleVideoTime(note.time || 0);
        switchTab('subs');
      });
      container.appendChild(row);
    });
  }

  // ===========================
  // RENDER: Wordbook tab
  // ===========================
  function renderWordbook() {
    var container = $('content-words');
    var projWords = wordbook.filter(function(w){ return w.project_id === PROJECT_ID; });
    $('wordCount').textContent = projWords.length ? '(' + projWords.length + ')' : '';

    if (!projWords.length) {
      container.innerHTML = '<div class="empty-hint">__WORD_EMPTY__</div>';
      return;
    }
    container.innerHTML = '';
    projWords.forEach(function(entry) {
      var row = document.createElement('div');
      row.className = 'item-row';
      var html = '<div class="item-main">';
      html += '<div class="item-word">' + esc(entry.word);
      if (entry.pos) html += '<span class="item-pos">' + esc(entry.pos) + '</span>';
      if (entry.translation) html += ' <span class="item-word-trans">= ' + esc(entry.translation) + '</span>';
      html += '</div>';
      if (entry.pronunciation) html += '<div class="item-pron">/' + esc(entry.pronunciation) + '/</div>';
      if (entry.other_meanings && entry.other_meanings.length)
        html += '<div class="item-other">other: ' + entry.other_meanings.map(esc).join('; ') + '</div>';
      if (entry.context) html += '<div class="item-context">&#128279; ' + esc(entry.context) + '</div>';
      html += '</div>';
      html += '<button class="item-del" title="Delete">&times;</button>';
      row.innerHTML = html;
      if (entry.time) {
        var ctx = row.querySelector('.item-context');
        if (ctx) {
          ctx.style.cursor = 'pointer';
          ctx.addEventListener('click', function() {
            video.currentTime = subtitleVideoTime(entry.time);
            video.play().catch(function(){});
            switchTab('subs');
          });
        }
      }
      row.querySelector('.item-del').addEventListener('click', function() {
        postCollection('/api/word/delete', {
          id: entry.id,
          project_id: entry.project_id || PROJECT_ID,
          word: entry.word
        }).then(function(result) {
          if (!result || !result.ok) return;
          wordbook = wordbook.filter(function(item){ return item.id !== entry.id; });
          renderWordbook();
        });
      });
      container.appendChild(row);
    });
  }

  renderNotes();
  renderWordbook();

  // ===========================
  // Tab switching
  // ===========================
  function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(function(t){ t.classList.toggle('active', t.dataset.tab === tabName); });
    document.querySelectorAll('.tab-content').forEach(function(c){ c.classList.toggle('active', c.id === 'content-' + tabName); });
    if (tabName !== 'subs') setSubtitleToolsOpen(false, false);
  }
  document.querySelectorAll('.tab').forEach(function(t){
    t.addEventListener('click', function(){ switchTab(this.dataset.tab); });
  });

  // ===========================
  // Video controls
  // ===========================
  function togglePlay() {
    if (video.paused) video.play().catch(function(){}); else video.pause();
  }
  function openNoteComposer() {
    video.pause();
    var now = Number(video.currentTime || 0);
    var active = subtitles.find(function(item){ return String(item.id) === String(lastActiveId); });
    if (!active) {
      active = subtitles.find(function(item) {
        return now >= subtitleVideoTime(item.start) && now < subtitleVideoTime(item.end);
      });
    }
    sendAction('open_note', {
      project_id: PROJECT_ID,
      time: now,
      subtitle_id: active ? (active.cue_id || active.id) : '',
      text: active ? (active.text || '') : '',
      translation: active ? (active.translation || '') : ''
    });
  }
  if (noteBtn) noteBtn.addEventListener('click', openNoteComposer);
  if (onlineNoteBtn) onlineNoteBtn.addEventListener('click', openNoteComposer);
  playBtn.addEventListener('click', togglePlay);
  centerPlay.addEventListener('click', togglePlay);
  if (nativeVideo) nativeVideo.addEventListener('click', togglePlay);
  video.addEventListener('play', function(){ playBtn.innerHTML='&#10074;&#10074;'; centerPlay.classList.remove('show'); });
  video.addEventListener('pause', function(){ playBtn.innerHTML='&#9654;'; centerPlay.classList.add('show'); });
  video.addEventListener('ended', function(){ playBtn.innerHTML='&#9654;'; centerPlay.classList.add('show'); });

  video.addEventListener('loadedmetadata', function() {
    durationEl.textContent = fmt(video.duration);
    if (INITIAL_SEEK > 0 && INITIAL_SEEK < video.duration) {
      video.currentTime = INITIAL_SEEK;
    }
  });
  video.addEventListener('timeupdate', function() {
    var pct = video.duration ? (video.currentTime/video.duration*100) : 0;
    played.style.width = pct + '%';
    thumb.style.left = pct + '%';
    currentEl.textContent = fmt(video.currentTime);
    updateActive();
  });
  video.addEventListener('progress', function() {
    if (video.buffered.length && video.duration)
      buffered.style.width = (video.buffered.end(video.buffered.length-1)/video.duration*100) + '%';
  });

  // Progress bar
  var dragging = false;
  function seekFromEvent(e) {
    var r = progressBar.getBoundingClientRect();
    var pct = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    video.currentTime = pct * video.duration;
  }
  progressBar.addEventListener('mousedown', function(e){ dragging = true; progressBar.classList.add('dragging'); seekFromEvent(e); });
  document.addEventListener('mousemove', function(e){ if (dragging) seekFromEvent(e); });
  document.addEventListener('mouseup', function(){ if (dragging){ dragging = false; progressBar.classList.remove('dragging'); } });

  // Volume
  volumeInput.addEventListener('input', function() {
    video.volume = parseFloat(this.value);
    video.muted = false;
    muteBtn.innerHTML = video.volume === 0 ? '&#128263;' : '&#128266;';
  });
  muteBtn.addEventListener('click', function() {
    video.muted = !video.muted;
    muteBtn.innerHTML = video.muted ? '&#128263;' : '&#128266;';
  });

  // The master CC switch and the subtitle settings menu always control the
  // learning overlay together. Online playback gets the same menu as a small
  // floating gear rather than a second playback timeline.
  function setSubtitles(enabled) {
    subsOn = !!enabled;
    subToggle.classList.toggle('active', subsOn);
    if (settingsSubtitlesToggle) {
      settingsSubtitlesToggle.classList.toggle('active', subsOn);
      settingsSubtitlesState.textContent = subsOn ? '__SUBTITLE_ON_SHORT__' : '__SUBTITLE_OFF_SHORT__';
    }
    if (!subsOn) subOverlay.classList.remove('show');
    else { resetActiveHighlight(); updateActive(); }
  }
  subToggle.addEventListener('click', function() { setSubtitles(!subsOn); });
  if (settingsSubtitlesToggle) settingsSubtitlesToggle.addEventListener('click', function() { setSubtitles(!subsOn); });

  function setSubtitleSettingsOpen(open) {
    subtitleSettingsMenu.classList.toggle('show', open);
    if (subtitleSettingsBtn) subtitleSettingsBtn.classList.toggle('active', open);
    if (onlineSubtitleSettingsBtn) onlineSubtitleSettingsBtn.classList.toggle('active', open);
  }
  function toggleSubtitleSettings(e) {
    e.stopPropagation();
    setSubtitleSettingsOpen(!subtitleSettingsMenu.classList.contains('show'));
  }
  if (subtitleSettingsBtn) subtitleSettingsBtn.addEventListener('click', toggleSubtitleSettings);
  if (onlineSubtitleSettingsBtn) onlineSubtitleSettingsBtn.addEventListener('click', toggleSubtitleSettings);
  document.addEventListener('click', function(e) {
    if (!subtitleSettingsMenu.contains(e.target) && e.target !== subtitleSettingsBtn && e.target !== onlineSubtitleSettingsBtn) setSubtitleSettingsOpen(false);
  });

  // Speed menu
  function toggleSpeedMenu(e) { e.stopPropagation(); speedMenu.classList.toggle('show'); }
  speedBtn.addEventListener('click', toggleSpeedMenu);
  if (onlineSpeedBtn) onlineSpeedBtn.addEventListener('click', toggleSpeedMenu);
  document.addEventListener('click', function(e) {
    if (!speedMenu.contains(e.target) && e.target !== speedBtn && e.target !== onlineSpeedBtn) speedMenu.classList.remove('show');
  });
  var speedItems = document.querySelectorAll('.speed-item');
  function setRate(rate) {
    currentRate = rate;
    video.playbackRate = rate;
    speedBtn.textContent = rate + 'x';
    if (onlineSpeedBtn) onlineSpeedBtn.textContent = rate + 'x';
  }
  for (var i = 0; i < speedItems.length; i++) {
    speedItems[i].addEventListener('click', function() {
      var sp = parseFloat(this.dataset.s);
      setRate(sp);
      for (var j = 0; j < speedItems.length; j++) speedItems[j].classList.remove('active');
      this.classList.add('active');
      speedMenu.classList.remove('show');
    });
  }

  // Layout + immersive fullscreen. The complete learning shell is the target,
  // so video and subtitles remain together instead of fullscreening video only.
  function togglePanel(collapsed) {
    learningShell.classList.toggle('panel-collapsed', !!collapsed);
    panelRestoreBtn.setAttribute('aria-hidden', collapsed ? 'false' : 'true');
  }
  panelRestoreBtn.addEventListener('click', function(){ togglePanel(false); });

  var fullscreenEventDoc = document;
  var fullscreenEventTarget = learningShell;
  try {
    if (window.frameElement && window.top.document) {
      fullscreenEventDoc = window.top.document;
      fullscreenEventTarget = window.frameElement;
    }
  } catch (_) {}
  var pseudoFullscreen = false;
  var originalHostStyle = null, originalBodyOverflow = '', originalHtmlOverflow = '';

  function syncFullscreenUi(active) {
    learningShell.classList.toggle('is-fullscreen', active);
    if (active) applyPanelRatio(currentPanelRatio(), false);
  }

  function enterPseudoFullscreen() {
    try {
      var hostFrame = window.frameElement;
      var hostDoc = window.top.document;
      if (!hostFrame || !hostDoc) return;
      originalHostStyle = hostFrame.getAttribute('style');
      originalBodyOverflow = hostDoc.body.style.overflow;
      originalHtmlOverflow = hostDoc.documentElement.style.overflow;
      hostFrame.style.setProperty('position', 'fixed', 'important');
      hostFrame.style.setProperty('inset', '0', 'important');
      hostFrame.style.setProperty('width', '100vw', 'important');
      hostFrame.style.setProperty('height', '100vh', 'important');
      hostFrame.style.setProperty('z-index', '2147483647', 'important');
      hostFrame.style.setProperty('border', '0', 'important');
      hostDoc.body.style.overflow = 'hidden';
      hostDoc.documentElement.style.overflow = 'hidden';
      pseudoFullscreen = true;
      syncFullscreenUi(true);
    } catch (_) {
      learningShell.classList.add('is-fullscreen');
    }
  }

  function exitPseudoFullscreen() {
    try {
      var hostFrame = window.frameElement;
      var hostDoc = window.top.document;
      if (hostFrame) {
        if (originalHostStyle === null) hostFrame.removeAttribute('style');
        else hostFrame.setAttribute('style', originalHostStyle);
      }
      if (hostDoc) {
        hostDoc.body.style.overflow = originalBodyOverflow;
        hostDoc.documentElement.style.overflow = originalHtmlOverflow;
      }
    } catch (_) {}
    pseudoFullscreen = false;
    syncFullscreenUi(false);
  }

  function toggleImmersiveFullscreen() {
    if (pseudoFullscreen) { exitPseudoFullscreen(); return; }
    if (fullscreenEventDoc.fullscreenElement) {
      fullscreenEventDoc.exitFullscreen();
      return;
    }
    try {
      Promise.resolve(fullscreenEventTarget.requestFullscreen()).then(function() {
        setTimeout(function() {
          if (!fullscreenEventDoc.fullscreenElement) enterPseudoFullscreen();
        }, 160);
      }).catch(enterPseudoFullscreen);
    } catch (_) {
      enterPseudoFullscreen();
    }
  }
  fsBtn.addEventListener('click', toggleImmersiveFullscreen);
  fullscreenEventDoc.addEventListener('fullscreenchange', function() {
    var active = fullscreenEventDoc.fullscreenElement === fullscreenEventTarget;
    syncFullscreenUi(active || pseudoFullscreen);
  });
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && pseudoFullscreen) exitPseudoFullscreen();
  });

  // Active subtitle tracking
  function updateActive() {
    var t = video.currentTime - subtitleOffset, activeId = -1;
    var lo = 0, hi = subtitles.length - 1;
    while (lo <= hi) {
      var mid = (lo+hi) >> 1, sub = subtitles[mid];
      if (t >= sub.start && t < sub.end) { activeId = sub.id; break; }
      else if (t < sub.start) hi = mid - 1; else lo = mid + 1;
    }
    if (activeId === lastActiveId) return;
    var newEl = $('sub-' + activeId);
    // Defensive invariant: one timeline position can own at most one
    // highlighted row, even after rapid seeks or programmatic timing changes.
    var highlightedRows = document.querySelectorAll('.sub-row.active');
    for (var i = 0; i < highlightedRows.length; i++) {
      if (highlightedRows[i] !== newEl) highlightedRows[i].classList.remove('active');
    }
    if (newEl) {
      newEl.classList.add('active');
      if (autoFollow && subsContainer.classList.contains('active')) {
        var targetTop = newEl.offsetTop - (subsContainer.clientHeight - newEl.offsetHeight) / 2;
        subsContainer.scrollTo({top:Math.max(0, targetTop), behavior:'smooth'});
      }
    }
    if (activeId >= 0 && subsOn) {
      var cur = subtitles[activeId];
      subOrig.innerHTML = hoverMarkup(cur.text);
      subTrans.textContent = cur.translation || '';
      subOrigLayer.classList.toggle('has-content', Boolean(cur.text));
      subTransLayer.classList.toggle('has-content', Boolean(cur.translation));
      subOverlay.classList.add('show');
    } else {
      subOverlay.classList.remove('show');
    }
    lastActiveId = activeId;
  }
  video.addEventListener('seeked', function() {
    resetActiveHighlight();
    updateActive();
  });

  // ===========================
  // Keyboard shortcuts
  // ===========================
  var rightKeyDown = false, speedFwdTimer = null;
  document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT') return;

    if (e.key === 'ArrowRight') {
      if (!rightKeyDown) {
        rightKeyDown = true;
        video.currentTime = Math.min(video.duration, video.currentTime + 5);
        speedFwdTimer = setTimeout(function() {
          if (rightKeyDown) video.playbackRate = 3.0;
        }, 400);
      }
      e.preventDefault();
      return;
    }

    switch (e.key) {
      case ' ':
        e.preventDefault(); togglePlay(); break;
      case 'ArrowLeft':
        video.currentTime = Math.max(0, video.currentTime - 5); break;
      case 'ArrowUp':
        e.preventDefault(); video.volume = Math.min(1, video.volume + 0.1); volumeInput.value = video.volume; break;
      case 'ArrowDown':
        e.preventDefault(); video.volume = Math.max(0, video.volume - 0.1); volumeInput.value = video.volume; break;
    }
  });
  document.addEventListener('keyup', function(e) {
    if (e.key === 'ArrowRight') {
      rightKeyDown = false;
      if (speedFwdTimer) { clearTimeout(speedFwdTimer); speedFwdTimer = null; }
      video.playbackRate = currentRate;
    }
  });

  // Auto-hide controls
  player.addEventListener('mousemove', function() {
    $('controls').style.opacity = '1';
    clearTimeout(ctrlTimer);
    ctrlTimer = setTimeout(function() { if (!video.paused) $('controls').style.opacity = '0'; }, 3000);
  });
  player.addEventListener('mouseleave', function() { if (!video.paused) $('controls').style.opacity = '0'; });

  // ===========================
  // Dictionary card lifecycle
  // ===========================
  var popupData = null;

  function showPopupNearRect(rect) {
    wordPopup.classList.add('show');
    wordPopup.style.visibility = 'hidden';
    var popupRect = wordPopup.getBoundingClientRect();
    var top = rect.bottom + 6;
    var left = rect.left + (rect.width / 2) - (popupRect.width / 2);

    left = Math.max(8, Math.min(left, window.innerWidth - popupRect.width - 8));
    if (top + popupRect.height > window.innerHeight - 8) {
      top = rect.top - popupRect.height - 6;
    }

    wordPopup.style.top = top + 'px';
    wordPopup.style.left = left + 'px';
    wordPopup.style.visibility = 'visible';
  }

  function hidePopup() {
    wordPopup.classList.remove('show');
    wordPopup.style.visibility = '';
    popupData = null;
    popupMode = 'lookup';
    showQuickLookup('');
  }

  document.addEventListener('mousedown', function(e) {
    if (wordPopup.classList.contains('show') && !wordPopup.contains(e.target)) hidePopup();
  });

  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && wordPopup.classList.contains('show')) hidePopup();
  });

  // An AI lookup crosses Streamlit's iframe boundary and reloads the player.
  // Reopen the same word card with its newly saved result so the learner never
  // has to click the word a second time just to see the answer.
  if (wordLookupEnabled && initialAiLookup && initialAiLookup.word && initialAiLookup.project_id === PROJECT_ID) {
    var initialSub = subtitles.find(function(item) {
      return String(item.id) === String(initialAiLookup.subtitle_id);
    });
    if (initialSub) {
      setTimeout(function() { openDictionaryLookup(initialAiLookup.word, initialSub, null); }, 80);
    }
  }
})();
</script>
</body>
</html>"""


def build_player_html(
    video_url: str,
    subtitles: list[dict],
    lang: str = "en",
    project_id: str = "",
    notes: list[dict] = None,
    wordbook: list[dict] = None,
    initial_seek: float = 0.0,
    playback_mode: str = "local",
    external_video_id: str = "",
    ai_lookup: dict | None = None,
    target_lang_code: str = "zh",
    lookup_api_base: str = "",
    subtitle_offset: float = 0.0,
) -> str:
    notes = notes or []
    wordbook = wordbook or []
    subtitle_json = json.dumps(subtitles, ensure_ascii=False).replace("</", "<\\/")
    notes_json = json.dumps(notes, ensure_ascii=False).replace("</", "<\\/")
    wordbook_json = json.dumps(wordbook, ensure_ascii=False).replace("</", "<\\/")
    ai_lookup_json = json.dumps(ai_lookup or {}, ensure_ascii=False).replace("</", "<\\/")
    is_online = playback_mode == "online" and bool(external_video_id)
    free_translate_target = {
        "zh": "zh-CN", "pt": "pt-PT",
    }.get(target_lang_code, target_lang_code or "zh-CN")
    try:
        safe_subtitle_offset = round(max(-5.0, min(5.0, float(subtitle_offset))) * 10) / 10
    except (TypeError, ValueError):
        safe_subtitle_offset = 0.0
    video_element = (
        '<div id="youtubePlayer" class="youtube-player"></div>'
        if is_online else
        f'<video id="video" playsinline preload="metadata" crossorigin="anonymous" src="{video_url}"></video>'
    )
    return (
        PLAYER_HTML
        .replace("__VIDEO_ELEMENT__", video_element)
        .replace("__PLAYBACK_MODE__", "online" if is_online else "local")
        .replace("__PLAYER_CLASS__", "online" if is_online else "local")
        .replace("__YOUTUBE_VIDEO_ID__", external_video_id if is_online else "")
        .replace("__FREE_TRANSLATE_TARGET__", free_translate_target)
        .replace("__LOOKUP_API_BASE__", lookup_api_base.rstrip("/"))
        .replace("__INITIAL_SUBTITLE_OFFSET__", str(safe_subtitle_offset))
        .replace("__SUBTITLE_JSON__", subtitle_json)
        .replace("__NOTES_JSON__", notes_json)
        .replace("__WORDBOOK_JSON__", wordbook_json)
        .replace("__AI_LOOKUP_JSON__", ai_lookup_json)
        .replace("__PROJECT_ID__", project_id or "")
        .replace("__INITIAL_SEEK__", str(initial_seek or 0.0))
        .replace("__NO_SUBTITLES__", t("player.no_subtitles", lang))
        .replace("__PENDING_TRANSLATION__", t("player.pending_translation", lang))
        .replace("__SUB_PANEL_TITLE__", t("player.sub_panel_title", lang))
        .replace("__FULLSCREEN__", t("player.fullscreen", lang))
        .replace("__PANEL_COLLAPSE__", t("player.panel_collapse", lang))
        .replace("__PANEL_RESTORE__", t("player.panel_restore", lang))
        .replace("__FOLLOW_ON__", t("player.follow_on", lang))
        .replace("__FOLLOW_RESUME__", t("player.follow_resume", lang))
        .replace("__PANEL_RESIZE__", t("player.panel_resize", lang))
        .replace("__PANEL_RESIZE_HINT__", t("player.panel_resize_hint", lang))
        .replace("__PANEL_RESIZE_VALUE__", t("player.panel_resize_value", lang))
        .replace("__TAB_SUBS__", t("player.tab_subtitles", lang))
        .replace("__TAB_NOTES__", t("notes.title", lang))
        .replace("__NOTE_CAPTURE__", t("notes.capture", lang))
        .replace("__NOTE_FROM_SUBTITLE__", t("notes.from_subtitle", lang))
        .replace("__NOTE_EMPTY__", t("notes.project_empty", lang))
        .replace("__UNTITLED_NOTE__", t("notes.untitled", lang))
        .replace("__TAB_WORDS__", t("player.tab_wordbook", lang))
        .replace("__POPUP_PLACEHOLDER__", t("player.popup_translation_placeholder", lang))
        .replace("__POPUP_SAVE__", t("player.popup_save", lang))
        .replace("__POPUP_CANCEL__", t("player.popup_cancel", lang))
        .replace("__SPEED__", t("player.speed", lang))
        .replace("__SUBTITLE_SETTINGS__", t("player.subtitle_settings", lang))
        .replace("__SETTINGS_SUBTITLES__", t("player.settings_subtitles", lang))
        .replace("__WORD_LOOKUP_SETTING__", t("player.word_lookup_setting", lang))
        .replace("__SUBTITLE_ON_SHORT__", t("player.subtitle_on_short", lang))
        .replace("__SUBTITLE_OFF_SHORT__", t("player.subtitle_off_short", lang))
        .replace("__VIDEO_ORIGINAL_TITLE__", t("player.video_original_title", lang))
        .replace("__VIDEO_ORIGINAL_OFF_TITLE__", t("player.video_original_off_title", lang))
        .replace("__VIDEO_ORIGINAL_ON__", t("player.video_original_on", lang))
        .replace("__VIDEO_ORIGINAL_OFF__", t("player.video_original_off", lang))
        .replace("__VIDEO_TRANSLATION_TITLE__", t("player.video_translation_title", lang))
        .replace("__VIDEO_TRANSLATION_OFF_TITLE__", t("player.video_translation_off_title", lang))
        .replace("__VIDEO_TRANSLATION_ON__", t("player.video_translation_on", lang))
        .replace("__VIDEO_TRANSLATION_OFF__", t("player.video_translation_off", lang))
        .replace("__VIDEO_TRANSLATION_ON_SHORT__", t("player.video_translation_on_short", lang))
        .replace("__VIDEO_TRANSLATION_OFF_SHORT__", t("player.video_translation_off_short", lang))
        .replace("__LIST_BILINGUAL_TITLE__", t("player.list_bilingual_title", lang))
        .replace("__LIST_BILINGUAL_OFF_TITLE__", t("player.list_bilingual_off_title", lang))
        .replace("__LIST_BILINGUAL_ON__", t("player.list_bilingual_on", lang))
        .replace("__LIST_BILINGUAL_OFF__", t("player.list_bilingual_off", lang))
        .replace("__VIDEO_SUBTITLES_ON__", t("player.video_subtitles_on", lang))
        .replace("__VIDEO_SUBTITLES_OFF__", t("player.video_subtitles_off", lang))
        .replace("__VIDEO_FONT_SIZE__", t("player.video_font_size", lang))
        .replace("__LIST_FONT_SIZE__", t("player.list_font_size", lang))
        .replace("__SUBTITLE_TOOLS__", t("player.subtitle_tools", lang))
        .replace("__SUBTITLE_SEARCH__", t("player.subtitle_search", lang))
        .replace("__SUBTITLE_SEARCH_PLACEHOLDER__", t("player.subtitle_search_placeholder", lang))
        .replace("__SUBTITLE_SEARCH_EMPTY__", t("player.subtitle_search_empty", lang))
        .replace("__SUBTITLE_TIMING__", t("player.subtitle_timing", lang))
        .replace("__SUBTITLE_TIMING_SHORT__", t("player.subtitle_timing_short", lang))
        .replace("__SUBTITLE_TIMING_CONTROLS__", t("player.subtitle_timing_controls", lang))
        .replace("__SUBTITLE_TIMING_HINT__", t("player.subtitle_timing_hint", lang))
        .replace("__SUBTITLE_OFFSET_EARLIER__", t("player.subtitle_offset_earlier", lang))
        .replace("__SUBTITLE_OFFSET_LATER__", t("player.subtitle_offset_later", lang))
        .replace("__SUBTITLE_OFFSET_RESET__", t("player.subtitle_offset_reset", lang))
        .replace("__SUBTITLE_DRAG_ORIGINAL__", t("player.subtitle_drag_original", lang))
        .replace("__SUBTITLE_DRAG_TRANSLATION__", t("player.subtitle_drag_translation", lang))
        .replace("__SUBTITLE_DRAG_HINT__", t("player.subtitle_drag_hint", lang))
        .replace("__SUBTITLE_POSITION_RESET__", t("player.subtitle_position_reset", lang))
        .replace("__HOVER_LOOKUP__", t("player.hover_lookup", lang))
        .replace("__HOVER_NO_RESULT__", t("player.hover_no_result", lang))
        .replace("__HOVER_AI_HINT__", t("player.hover_ai_hint", lang))
        .replace("__ACTION_TIMEOUT__", t("player.action_timeout", lang))
        .replace("__WORD_SAVE__", t("player.word_save", lang))
        .replace("__WORD_REMOVE__", t("player.word_remove", lang))
        .replace("__PLAY_PRONUNCIATION__", t("player.play_pronunciation", lang))
        .replace("__PHONETIC_UK__", t("player.phonetic_uk", lang))
        .replace("__PHONETIC_US__", t("player.phonetic_us", lang))
        .replace("__PHONETIC__", t("player.phonetic", lang))
        .replace("__WORDBOOK_SOURCE__", t("player.wordbook_source", lang))
        .replace("__POPUP_TRANSLATING__", t("player.popup_translating", lang))
        .replace("__POPUP_SAVING__", t("player.popup_saving", lang))
        .replace("__AI_TRANSLATE__", t("player.ai_translate", lang))
        .replace("__FREE_TRANSLATION_LABEL__", t("player.free_translation_label", lang))
        .replace("__FREE_TRANSLATION_FAILED__", t("player.free_translation_failed", lang))
        .replace("__WORD_EMPTY__", t("player.word_empty", lang))
        .replace("__PLAYER_INIT_FAILED__", t("player.init_failed", lang))
    )


# ===========================================================================
# Settings page (full-width via st.navigation)
# ===========================================================================

def settings_page():
    """Full-width settings page."""
    _apply_product_theme()
    _install_sidebar_toggle(lang)
    st.title(t("sidebar.header", lang))
    st.caption(t("app.eyebrow", lang))

    # ---- UI Language ----
    st.subheader(t("sidebar.ui_language", lang))
    ui_lang_new = st.selectbox(
        t("sidebar.ui_language", lang),
        options=list(UI_LANGUAGES.keys()),
        format_func=lambda x: UI_LANGUAGES[x],
        index=list(UI_LANGUAGES.keys()).index(st.session_state.ui_lang),
        label_visibility="collapsed",
    )
    if ui_lang_new != st.session_state.ui_lang:
        st.session_state.ui_lang = ui_lang_new
        st.rerun()

    st.divider()

    # ---- ASR ----
    st.subheader(t("sidebar.section_asr", lang))
    c1, c2 = st.columns([1, 2])
    with c1:
        asr_model = st.selectbox(
            t("sidebar.asr_model", lang),
            options=list(MODEL_SIZES.keys()),
            format_func=lambda x: MODEL_SIZES[x],
            index=list(MODEL_SIZES.keys()).index(st.session_state.asr_model)
            if st.session_state.asr_model in MODEL_SIZES else 1,
            help=t("sidebar.asr_model_help", lang),
        )
        st.session_state.asr_model = asr_model

    st.divider()

    # ---- Translation ----
    st.subheader(t("sidebar.section_translation", lang))
    c1, c2 = st.columns(2)
    with c1:
        target_lang = st.selectbox(
            t("sidebar.target_language", lang),
            options=list(TARGET_LANGUAGES.keys()),
            format_func=lambda x: f"{LANG_FLAGS.get(x, '')} {TARGET_LANGUAGES[x]}",
            index=list(TARGET_LANGUAGES.keys()).index(st.session_state.target_lang)
            if st.session_state.target_lang in TARGET_LANGUAGES else 0,
        )
        st.session_state.target_lang = target_lang
    with c2:
        presets = st.session_state.presets
        preset_names = [p["name"] for p in presets]
        if not preset_names:
            preset_names = [t("sidebar.preset_none", lang)]
        sel_idx = st.session_state.selected_preset_index
        if sel_idx >= len(presets):
            sel_idx = 0
        selected_name = st.selectbox(
            t("sidebar.model_preset", lang),
            options=preset_names,
            index=sel_idx,
        )
        for i, p in enumerate(presets):
            if p["name"] == selected_name:
                st.session_state.selected_preset_index = i
                break

    active_preset = presets[st.session_state.selected_preset_index] if presets else None
    if active_preset and active_preset.get("engine_type") == "hy_mt2_local":
        _render_hy_mt2_status(active_preset)

    # ---- Concurrency ----
    st.divider()
    st.subheader(t("sidebar.section_concurrency", lang))
    if active_preset and active_preset.get("engine_type") == "hy_mt2_local":
        st.caption(t("settings.local_concurrency", lang))
    else:
        workers = st.slider(
            t("sidebar.max_workers", lang),
            min_value=1, max_value=16,
            value=st.session_state.max_workers,
            help=t("sidebar.max_workers_help", lang),
        )
        st.session_state.max_workers = workers

    with st.expander(t("settings.advanced", lang), expanded=False):
        # Model-name overrides are meaningful for OpenAI-compatible engines,
        # but the on-device engine always uses its verified local directory.
        if not active_preset or active_preset.get("engine_type") != "hy_mt2_local":
            current_model = presets[st.session_state.selected_preset_index]["model_name"] if presets else ""
            override = st.session_state.get("model_override", "")
            c_m, c_clr = st.columns([4, 1])
            with c_m:
                model_input = st.text_input(
                    t("sidebar.model_override", lang),
                    value=override or current_model,
                    placeholder="kaelri/hy-mt2:7b",
                    help=t("sidebar.model_override_help", lang),
                )
            with c_clr:
                if st.button(t("sidebar.model_reset", lang), width="stretch"):
                    st.session_state.model_override = ""
                    st.rerun()
            if model_input.strip() and model_input.strip() != current_model:
                st.session_state.model_override = model_input.strip()
            elif model_input.strip() == current_model:
                st.session_state.model_override = ""

        st.subheader(t("sidebar.manage_presets", lang))
        _render_preset_editor()

        st.divider()
        st.subheader(t("sidebar.section_network", lang))
        proxy_env = os.environ.get("https_proxy") or os.environ.get("http_proxy") or ""
        proxy = st.text_input(
            t("sidebar.proxy", lang),
            value=st.session_state.proxy or proxy_env,
            placeholder=t("sidebar.proxy_placeholder", lang),
            help=t("sidebar.proxy_help", lang),
        )
        st.session_state.proxy = proxy
    _persist_settings()

    st.divider()

    # ---- Actions ----
    st.subheader(t("sidebar.section_actions", lang))
    if st.button(t("sidebar.clear_reset", lang)):
        reset_state()
        st.rerun()


def _render_hy_mt2_status(preset: dict) -> None:
    """Show actionable local-model state before a user starts a long task."""
    from englishlearn.translation.hy_mt2_local import download_model, model_status

    status = model_status(preset.get("api_base") or None)
    with st.container(border=True):
        st.subheader(t("settings.local_runtime", lang))
        st.caption(t("settings.model_download_hint", lang))
        st.code(status["path"], language=None)
        if status["downloaded"] and status["dependencies_ready"]:
            st.success(t("settings.model_ready", lang))
            return

        st.warning(status["message"] or t("settings.model_setup", lang))
        if not status["dependencies_ready"]:
            st.caption(t("settings.model_setup", lang))
            return

        if st.button(t("settings.model_download", lang), type="primary", key="download_hy_mt2"):
            try:
                with st.spinner(t("settings.model_download", lang)):
                    download_model(preset.get("api_base") or None)
                st.success(t("settings.model_downloaded", lang))
                st.rerun()
            except Exception as exc:
                st.error(t("settings.model_download_failed", lang, error=str(exc)))


def _render_preset_editor():
    """Render the preset editor as cards with Use / Edit / Save / Delete."""
    presets = st.session_state.presets

    # Show currently active model at top
    override = st.session_state.get("model_override", "")
    active_preset = presets[st.session_state.selected_preset_index] if presets else None
    if active_preset:
        active_model = override or active_preset["model_name"]
        badge = f"🔀 `{active_model}`" if override else f"✨ `{active_model}`"
        st.info(f"{t('sidebar.current_model', lang)}{badge}", icon="🧠")

    for idx, p in enumerate(presets):
        is_active = idx == st.session_state.selected_preset_index and not override
        with st.container(border=True):
            # Row 1: name + Use / Edit buttons
            c_name, c_use, c_edit = st.columns([4, 1.2, 1])
            with c_name:
                icon = "🟢" if is_active else "⚪"
                engine_icon = {"ollama": "🦙", "openai": "☁️", "hy_mt2_local": "💻"}.get(p["engine_type"], "⚙️")
                engine_label = t(f"engine.{p['engine_type']}", lang)
                st.caption(f"{icon} **{p['name']}**  ·  {engine_icon} {engine_label}  ·  `{p['model_name']}`")
            with c_use:
                use_label = t("sidebar.preset_using", lang) if is_active else t("sidebar.preset_use", lang)
                use_disabled = is_active
                if st.button(
                    use_label, key=f"use_{idx}",
                    width="stretch",
                    disabled=use_disabled,
                ):
                    st.session_state.selected_preset_index = idx
                    st.session_state.model_override = ""
                    st.rerun()
            with c_edit:
                edit_open = st.session_state.get(f"edit_preset_{idx}", False)
                edit_label = t("sidebar.preset_hide", lang) if edit_open else "✏️"
                if st.button(edit_label, key=f"edit_{idx}", width="stretch"):
                    st.session_state[f"edit_preset_{idx}"] = not edit_open
                    st.rerun()

            # Row 2: Edit form (shown when toggled)
            if st.session_state.get(f"edit_preset_{idx}", False):
                p["name"] = st.text_input(t("sidebar.preset_name", lang), value=p["name"], key=f"pn_{idx}")
                p["engine_type"] = st.selectbox(
                    t("sidebar.preset_engine", lang),
                    options=["hy_mt2_local", "ollama", "openai"],
                    format_func=lambda x: t(f"engine.{x}", lang),
                    index=["hy_mt2_local", "ollama", "openai"].index(
                        p.get("engine_type", "openai")
                        if p.get("engine_type") in {"hy_mt2_local", "ollama", "openai"} else "openai"
                    ),
                    key=f"pe_{idx}",
                )
                p["model_name"] = st.text_input(t("sidebar.preset_model", lang), value=p.get("model_name", ""), key=f"pm_{idx}")
                p["api_base"] = st.text_input(t("sidebar.preset_api_base", lang), value=p.get("api_base", ""), key=f"pb_{idx}")
                if p["engine_type"] != "hy_mt2_local":
                    p["api_key"] = st.text_input(t("sidebar.preset_api_key", lang), value=p.get("api_key", ""), type="password", key=f"pk_{idx}")
                else:
                    p["api_key"] = ""
                p["max_tokens"] = st.number_input("Max Tokens", value=p.get("max_tokens", 32768), min_value=256, step=256, key=f"pmt_{idx}")

                c_save, c_del = st.columns([1, 1])
                with c_save:
                    if st.button(t("sidebar.preset_save_btn", lang), key=f"apply_{idx}", width="stretch"):
                        save_presets(presets)
                        st.session_state.presets = presets
                        st.session_state[f"edit_preset_{idx}"] = False
                        st.rerun()
                with c_del:
                    if len(presets) > 1 and st.button(
                        t("sidebar.preset_delete", lang),
                        key=f"del_preset_{idx}", width="stretch",
                    ):
                        presets.pop(idx)
                        save_presets(presets)
                        st.session_state.presets = presets
                        st.session_state.selected_preset_index = 0
                        st.rerun()

    # Add new preset
    if st.button(f"+ {t('sidebar.preset_add', lang)}", width="stretch"):
        st.session_state.show_add_preset = not st.session_state.get("show_add_preset", False)
        st.rerun()

    if st.session_state.get("show_add_preset", False):
        with st.container(border=True):
            st.caption(t("sidebar.preset_add", lang))
            p_name = st.text_input(t("sidebar.preset_name", lang), key="new_preset_name")
            p_engine = st.selectbox(
                t("sidebar.preset_engine", lang),
                options=["hy_mt2_local", "ollama", "openai"],
                format_func=lambda x: t(f"engine.{x}", lang),
                key="new_preset_engine",
            )
            p_model = st.text_input(t("sidebar.preset_model", lang), key="new_preset_model")
            p_base = st.text_input(t("sidebar.preset_api_base", lang), key="new_preset_base")
            p_key = st.text_input(t("sidebar.preset_api_key", lang), type="password", key="new_preset_key")
            p_max_tokens = st.number_input("Max Tokens", value=32768, min_value=256, step=256, key="new_preset_max_tokens")
            if st.button(t("sidebar.preset_add_btn", lang), width="stretch") and p_name.strip():
                presets.append({
                    "name": p_name.strip(),
                    "engine_type": p_engine,
                    "model_name": p_model.strip(),
                    "api_base": p_base.strip(),
                    "api_key": p_key.strip(),
                    "max_tokens": p_max_tokens,
                })
                save_presets(presets)
                st.session_state.presets = presets
                st.session_state.show_add_preset = False
                st.rerun()

    st.caption(t("sidebar.api_key_note", lang))

# ===========================================================================
# Home page
# ===========================================================================

def _open_new_task() -> None:
    """Enter the creation flow from any primary entry point."""
    st.session_state.show_new_task = True
    st.session_state.processed = False
    st.session_state.selected_project_id = None
    st.session_state.current_project = None
    st.session_state.subtitles = None
    st.session_state.scroll_main_to_top = True


def _scroll_main_to_top_once() -> None:
    """Reset Streamlit's scroll container only after an explicit view change."""
    if not st.session_state.get("scroll_main_to_top"):
        return
    st.session_state.scroll_main_to_top = False
    st.iframe(
        """
        <script>
        const main = window.parent.document.querySelector('[data-testid="stMain"]');
        if (main) {
          const reset = () => { main.scrollTop = 0; };
          reset();
          window.requestAnimationFrame(reset);
          window.setTimeout(reset, 60);
        }
        </script>
        """,
        width=1,
        height=1,
        tab_index=-1,
    )


def _render_home_masthead(lang: str) -> None:
    """Persistent top bar: product identity, project search, primary action."""
    with st.container(key="app_masthead"):
        brand_col, search_col, action_col = st.columns(
            [1.1, 2.4, .8], vertical_alignment="center",
        )
        with brand_col:
            st.markdown(
                '<div class="app-masthead"><span class="app-mark">▶</span><span>EnglishLearn</span></div>',
                unsafe_allow_html=True,
            )
        with search_col:
            st.text_input(
                t("projects.search", lang), key="project_search",
                placeholder=f"⌕ {t('projects.search_placeholder', lang)}",
                label_visibility="collapsed",
            )
        with action_col:
            if st.button(
                f"＋ {t('projects.new_task', lang)}", key="new_task_btn",
                type="primary", width="stretch",
            ):
                _open_new_task()
                st.rerun()

def home_page():
    """Main page: project library + pipeline + player."""
    global lang, presets
    _apply_product_theme()
    _install_sidebar_toggle(lang)
    # Do not wake and reconcile the UI every two seconds while idle. Mount the
    # polling fragment only when work is active or has just completed.
    _tasks = _active_tasks_for_app()
    _seen = set(st.session_state.get("seen_task_completions", []))
    if any(not task["completed"] or task_id not in _seen for task_id, task in _tasks.items()):
        _poll_tasks_fragment()
    _render_home_masthead(lang)
    with st.sidebar:
        _render_project_list(lang)
    _render_home(lang, presets)


@st.fragment(run_every=2)
def _poll_tasks_fragment():
    """Poll background pipeline tasks: refresh UI while running, update on finish."""
    tasks = _active_tasks_for_app()
    seen = set(st.session_state.get("seen_task_completions", []))
    just_finished = [
        (task_key, task) for task_key, task in tasks.items()
        if task["completed"] and task_key not in seen
    ]
    if just_finished:
        st.session_state.projects = load_projects()
        current_pid = st.session_state.get("selected_project_id")
        for task_key, task in just_finished:
            seen.add(task_key)
            fname = task.get("filename", "")[:40]
            if task.get("error"):
                st.session_state.pipeline_error = task["error"]
                st.toast(f"❌ {fname}: {task['error'][:60]}")
            else:
                st.toast(f"✅ {fname}", icon="✅")
            if current_pid and task.get("project_id") == current_pid:
                updated = next(
                    (p for p in st.session_state.projects if p["id"] == current_pid), None,
                )
                if updated:
                    st.session_state.current_project = updated
                    subs = load_project_subtitles(current_pid)
                    if subs:
                        st.session_state.subtitles = subs
                        video_path = updated.get("video_path")
                        if video_path and os.path.exists(video_path):
                            st.session_state.video_path = video_path
                            st.session_state.audio_path = updated.get("audio_path")
                            try:
                                st.session_state.server_url = start_video_server(
                                    os.path.dirname(video_path), Path(video_path).name,
                                )
                            except OSError as exc:
                                st.session_state.pipeline_error = str(exc)
                            else:
                                st.session_state.processed = True
                                st.session_state.processing = False
        st.session_state.seen_task_completions = sorted(seen)

    if just_finished:
        st.rerun(scope="app")


# ---------------------------------------------------------------------------
# Player action dispatcher (query params -> data store)
# ---------------------------------------------------------------------------

def _handle_player_actions():
    """Process actions sent from the player iframe via URL query params.

    Actions:
        open_note        — pause/capture context and open the project note composer
        lookup_word_ai   — translate a word in context via LLM, show in the popup
        translate_word   — legacy wordbook-save action
        delete_word       — remove word by id
        toggle_favorite   — legacy sentence-favorite compatibility action
        delete_favorite   — legacy sentence-favorite compatibility action
        seek              — set seek_to for next player render
    """
    qp = st.query_params
    raw_params = {key: qp.get(key) for key in list(qp.keys())}

    def _param(name: str, default: str = "") -> str:
        value = raw_params.get(name, default)
        if isinstance(value, list):
            value = value[-1] if value else default
        return str(value if value is not None else default)

    action = _param("action")
    if not action:
        return
    valid_actions = {"open_note", "lookup_word_ai", "translate_word", "delete_word", "toggle_favorite", "delete_favorite", "seek"}

    # Consume the URL action before model or disk work so a failure/rerun cannot
    # replay the same write.
    for key in list(qp.keys()):
        del qp[key]

    if action not in valid_actions:
        st.rerun()

    projects_list = st.session_state.projects or load_projects()
    project_ids = {p.get("id") for p in projects_list}

    # Full-page reload is the only reliable bridge out of Streamlit's sandboxed
    # iframe. Restore the source project from the action payload before the
    # final rerun so a star/word action never drops the learner back at Welcome.
    action_project_id = _param("project_id")
    if action_project_id in project_ids:
        action_project = next(p for p in projects_list if p.get("id") == action_project_id)
        st.session_state.current_project = action_project
        st.session_state.selected_project_id = action_project_id
        action_subtitles = load_project_subtitles(action_project_id)
        if action_subtitles:
            st.session_state.subtitles = action_subtitles

    if action == "open_note":
        try:
            time_val = max(0.0, float(_param("time", "0")))
        except ValueError:
            time_val = 0.0
        st.session_state.seek_to = time_val
        st.session_state.show_note_composer = True
        st.session_state.note_draft = {
            "project_id": action_project_id,
            "time": time_val,
            "subtitle_id": _param("subtitle_id"),
            "source_text": _param("text").strip()[:4000],
            "source_translation": _param("translation").strip()[:4000],
        }
        st.session_state.note_result = None
        st.session_state.note_job_id = None
        st.session_state.note_job_error = None
        st.session_state.note_composer_nonce += 1

    elif action in {"lookup_word_ai", "translate_word"}:
        word = _param("word").strip()[:160]
        context = _param("context").strip()[:2000]
        project_id = _param("project_id")
        if not word or not context or project_id not in project_ids:
            st.session_state.collection_error = t("collections.action_invalid", lang)
            st.rerun()
        try:
            time_val = max(0.0, float(_param("time", "0")))
        except ValueError:
            time_val = 0.0
        st.session_state.seek_to = time_val

        preset = _active_preset()
        setup_error = _translation_setup_error(preset)
        if setup_error:
            st.session_state.collection_error = setup_error
            st.rerun()
        assert preset is not None
        _model = st.session_state.model_override or preset["model_name"]
        target_lang_name = TARGET_LANGUAGES.get(st.session_state.target_lang, st.session_state.target_lang)

        try:
            with st.spinner(t("collections.translating_word", lang, word=word)):
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(
                    translate_word,
                    word=word,
                    context=context,
                    target_lang=target_lang_name,
                    model=_model,
                    api_base=preset["api_base"],
                    api_key=_effective_preset_api_key(preset),
                    max_tokens=preset.get("max_tokens", 1024),
                    engine=preset["engine_type"],
                )
                try:
                    result = future.result(timeout=2.0)
                except FutureTimeout as exc:
                    future.cancel()
                    raise RuntimeError(t("collections.action_timeout", lang)) from exc
                finally:
                    # Do not hold the Streamlit rerun open behind a slow model.
                    executor.shutdown(wait=False, cancel_futures=True)
            translation = str((result or {}).get("translation", "")).strip()
            if not translation:
                raise RuntimeError(t("collections.word_translation_empty", lang))

            payload = {
                "word": word,
                "translation": translation,
                "pos": str((result or {}).get("pos", "")).strip(),
                "pronunciation": str((result or {}).get("pronunciation", "")).strip(),
                "other_meanings": [
                    str(item).strip()
                    for item in ((result or {}).get("other_meanings") or [])
                    if str(item).strip()
                ],
                "context": context,
                "time": time_val,
                "subtitle_id": _param("subtitle_id"),
                "project_id": project_id,
            }
            if action == "translate_word":
                existing = wordbook_store.find(word=word, context=context, project_id=project_id)
                if existing:
                    wordbook_store.update(existing["id"], **payload)
                    st.session_state.collection_notice = t("collections.word_updated", lang, word=word)
                else:
                    wordbook_store.add(payload)
                    st.session_state.collection_notice = t("collections.word_saved", lang, word=word)
            else:
                # A lookup should behave like a dictionary, not silently add a
                # word to the learner's review queue.  Keep the result only
                # long enough to reopen the originating popup after rerun.
                st.session_state.pending_ai_lookup = payload
        except Exception as exc:
            error_key = "collections.word_lookup_failed" if action == "lookup_word_ai" else "collections.action_failed"
            st.session_state.collection_error = t(error_key, lang, error=str(exc))

    elif action == "delete_word":
        wid = _param("id")
        if wid:
            wordbook_store.remove(wid)
            st.session_state.collection_notice = t("collections.word_removed", lang)

    elif action == "toggle_favorite":
        sub_id = _param("subtitle_id")
        project_id = _param("project_id")
        if project_id not in project_ids or not sub_id:
            st.session_state.collection_error = t("collections.action_invalid", lang)
            st.rerun()
        try:
            time_val = max(0.0, float(_param("time", "0")))
        except ValueError:
            time_val = 0.0
        st.session_state.seek_to = time_val
        existing = next(
            (
                item for item in favorites_store.load()
                if str(item.get("subtitle_id")) == str(sub_id)
                and str(item.get("project_id")) == str(project_id)
            ),
            None,
        )
        if existing:
            favorites_store.remove(existing["id"])
            st.session_state.collection_notice = t("collections.favorite_removed", lang)
        else:
            favorites_store.add({
                "subtitle_id": sub_id,
                "text": _param("text").strip()[:4000],
                "translation": _param("translation").strip()[:4000],
                "time": time_val,
                "project_id": project_id,
            })
            st.session_state.collection_notice = t("collections.favorite_saved", lang)

    elif action == "delete_favorite":
        fid = _param("id")
        if fid:
            favorites_store.remove(fid)
            st.session_state.collection_notice = t("collections.favorite_removed", lang)

    elif action == "seek":
        try:
            st.session_state.seek_to = max(0.0, float(_param("time", "0")))
        except ValueError:
            pass

    st.rerun()


def _render_home(lang, presets):
    _handle_project_open_query(lang)
    _handle_player_actions()
    _scroll_main_to_top_once()
    _collection_notice = st.session_state.get("collection_notice")
    if _collection_notice:
        st.toast(_collection_notice, icon="✅")
        st.session_state.collection_notice = None
    _collection_error = st.session_state.get("collection_error")
    if _collection_error:
        st.error(_collection_error)
        st.session_state.collection_error = None
    _err = st.session_state.get("pipeline_error")
    if _err:
        st.error(t("pipeline.failed", lang, error=_err))
        st.session_state.pipeline_error = None

    if st.session_state.current_project:
        _render_project_detail(lang, presets)
    elif st.session_state.show_new_task:
        _render_new_task_form(lang, presets)
    else:
        _render_welcome(lang)


# ---------------------------------------------------------------------------
# Project selection
# ---------------------------------------------------------------------------

_STATUS_ICONS = {"completed": "✅", "processing": "⏳", "failed": "❌", "legacy": "📦"}


def _format_project_created_at(value: str) -> str:
    """Present a compact, local creation timestamp for a project card."""
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(value)[:16]


_SPEAKER_PREFIX = re.compile(
    r"^([A-Za-z\u3400-\u9fff][A-Za-z0-9\u3400-\u9fff ._'’·・-]{0,31})\s*[:：]\s*(.+)$",
    re.DOTALL,
)
_INLINE_SPEAKER_PREFIX = re.compile(
    r"(?:^|(?<=[.!?。！？]\s))"
    r"([A-Za-z\u3400-\u9fff][A-Za-z0-9\u3400-\u9fff ._'’·・-]{0,31})\s*[:：]"
)


def _transcript_parts(
    subtitle: dict, field: str, unknown_speaker: str,
) -> tuple[str, str]:
    """Return a speaker label and flat readable text for one transcript cell."""
    raw = str(subtitle.get(field) or "").replace("\xa0", " ")
    text = " ".join(part.strip() for part in raw.splitlines() if part.strip()).strip()
    speaker = str(
        subtitle.get("speaker") or subtitle.get("speaker_name") or ""
    ).strip()
    if text.startswith(">>"):
        text = text[2:].lstrip(" -–—:：")
        speaker = speaker or unknown_speaker
    match = _SPEAKER_PREFIX.match(text)
    if match:
        speaker, text = match.group(1).strip(), match.group(2).strip()
    return speaker, text


def _transcript_sheet_html(subtitles: list[dict], lang: str) -> str:
    """Build a stable time/speaker/source/target transcript grid."""
    unknown_speaker = t("results.speaker_unknown", lang)
    speaker_numbers: dict[str, int] = {}
    current_speaker = ""
    rows: list[str] = []
    for subtitle in subtitles:
        source_speaker, source_text = _transcript_parts(
            subtitle, "text", unknown_speaker,
        )
        target_speaker, target_text = _transcript_parts(
            subtitle, "translation", unknown_speaker,
        )
        if target_speaker == unknown_speaker and source_speaker:
            target_speaker = source_speaker

        explicit_speaker = source_speaker
        if explicit_speaker == unknown_speaker:
            explicit_speaker = ""
        if not explicit_speaker and not source_text:
            explicit_speaker = target_speaker
            if explicit_speaker == unknown_speaker:
                explicit_speaker = ""
        if explicit_speaker:
            current_speaker = explicit_speaker
        elif not current_speaker:
            current_speaker = "__unknown__"
        if current_speaker not in speaker_numbers:
            speaker_numbers[current_speaker] = len(speaker_numbers) + 1
        speaker_label = t(
            "results.speaker_number", lang,
            number=speaker_numbers[current_speaker],
        )

        def text_html(text: str, css_class: str) -> str:
            value = html_lib.escape(text) if text else "—"
            empty = " transcript-empty" if not text else ""
            return f'<div class="{css_class}{empty}">{value}</div>'

        start = float(subtitle.get("start", 0.0) or 0.0)
        end = max(start, float(subtitle.get("end", start) or start))
        rows.append(
            '<article class="transcript-row">'
            f'<div class="transcript-time">{start:.1f}s – {end:.1f}s</div>'
            f'<div class="transcript-speaker">{html_lib.escape(speaker_label)}</div>'
            f'{text_html(source_text, "transcript-source")}'
            f'{text_html(target_text, "transcript-target")}'
            '</article>'
        )

        # A cue can finish with the next speaker (for example, "... Sarah:").
        # Carry that identity into the following row without changing this
        # row's leading speaker label.
        raw_source = str(subtitle.get("text") or "").replace("\xa0", " ")
        mentioned_speakers = _INLINE_SPEAKER_PREFIX.findall(raw_source)
        if mentioned_speakers:
            current_speaker = mentioned_speakers[-1].strip()

    header = (
        '<div class="transcript-head" aria-hidden="true">'
        f'<span>{html_lib.escape(t("results.transcript_time", lang))}</span>'
        f'<span>{html_lib.escape(t("results.transcript_speaker", lang))}</span>'
        f'<span>{html_lib.escape(t("results.transcript_source", lang))}</span>'
        f'<span>{html_lib.escape(t("results.transcript_target", lang))}</span>'
        '</div>'
    )
    return '<div class="transcript-sheet">' + header + "".join(rows) + '</div>'


def _render_transcript_sheet(subtitles: list[dict], lang: str) -> None:
    """Render subtitle data as a reading-first bilingual transcript."""
    st.markdown(
        _transcript_sheet_html(subtitles, lang),
        unsafe_allow_html=True,
    )


def _refresh_edited_subtitles(project_id: str) -> None:
    refreshed = load_project_subtitles(project_id) or []
    st.session_state.subtitles = refreshed


def _render_subtitle_editor(project_id: str, subtitles: list[dict], lang: str) -> None:
    """Render one focused editor instead of hundreds of expensive widgets."""
    if not subtitles:
        return
    labels = {
        str(cue.get("cue_id")): (
            f"{float(cue.get('start', 0)):.1f}s · "
            f"{str(cue.get('text') or '')[:72]}"
        )
        for cue in subtitles
    }
    cue_ids = list(labels)
    selected_id = st.selectbox(
        t("editor.select", lang), cue_ids,
        format_func=lambda value: labels[value], key=f"subtitle_editor_cue_{project_id}",
    )
    cue = next(item for item in subtitles if str(item.get("cue_id")) == selected_id)
    history = subtitle_history_status(project_id)
    undo_col, redo_col, status_col = st.columns([1, 1, 3], vertical_alignment="center")
    with undo_col:
        if st.button(t("editor.undo", lang), disabled=history["undo"] == 0, key=f"editor_undo_{project_id}"):
            if undo_subtitle_edit(project_id) is not None:
                _refresh_edited_subtitles(project_id)
                st.rerun()
    with redo_col:
        if st.button(t("editor.redo", lang), disabled=history["redo"] == 0, key=f"editor_redo_{project_id}"):
            if redo_subtitle_edit(project_id) is not None:
                _refresh_edited_subtitles(project_id)
                st.rerun()
    with status_col:
        st.caption(t("editor.history", lang, undo=history["undo"], redo=history["redo"]))

    with st.form(f"subtitle_edit_form_{project_id}_{selected_id}", border=True):
        source = st.text_area(t("editor.source", lang), value=str(cue.get("text") or ""), height=100)
        target = st.text_area(t("editor.target", lang), value=str(cue.get("translation") or ""), height=100)
        start_col, end_col = st.columns(2)
        with start_col:
            start = st.number_input(t("editor.start", lang), min_value=0.0, value=float(cue.get("start", 0)), step=0.1, format="%.3f")
        with end_col:
            end = st.number_input(t("editor.end", lang), min_value=0.0, value=float(cue.get("end", 0)), step=0.1, format="%.3f")
        if st.form_submit_button(t("editor.save", lang), type="primary"):
            if update_subtitle_cue(
                project_id, selected_id, text=source, translation=target,
                start=start, end=end,
            ) is not None:
                _refresh_edited_subtitles(project_id)
                st.toast(t("editor.saved", lang), icon="✅")
                st.rerun()

    split_default = max(1, min(len(str(cue.get("text") or "")) - 1, len(str(cue.get("text") or "")) // 2))
    op_a, op_b = st.columns(2)
    with op_a:
        split_at = st.number_input(
            t("editor.split_position", lang), min_value=1,
            max_value=max(1, len(str(cue.get("text") or "")) - 1),
            value=split_default, step=1, key=f"split_at_{project_id}_{selected_id}",
        )
        if st.button(t("editor.split", lang), key=f"split_{project_id}_{selected_id}", width="stretch"):
            if split_subtitle_cue(project_id, selected_id, int(split_at)) is not None:
                _refresh_edited_subtitles(project_id)
                st.rerun()
            st.error(t("editor.split_invalid", lang))
    with op_b:
        st.caption(t("editor.merge_hint", lang))
        if st.button(t("editor.merge_next", lang), key=f"merge_{project_id}_{selected_id}", width="stretch"):
            if merge_subtitle_with_next(project_id, selected_id) is not None:
                _refresh_edited_subtitles(project_id)
                st.rerun()
            st.error(t("editor.merge_invalid", lang))
    project = next(
        (item for item in (st.session_state.projects or []) if item.get("id") == project_id),
        None,
    )
    if project and st.button(
        t("editor.retranslate_selected", lang),
        key=f"retranslate_selected_{project_id}_{selected_id}", width="stretch",
    ):
        title = project.get("custom_title") or project.get("title") or project_id
        if _queue_project_task(
            "partial_retranslate", project, title, cue_ids=[selected_id],
        ):
            st.toast(t("editor.retranslate_queued", lang), icon="🔄")
            st.rerun()


def _resolve_project_playback(proj: dict, lang: str) -> tuple[Optional[str], Optional[str]]:
    """Resolve a project into (video_path, server_url).

    Online projects intentionally return ``(None, None)`` and must never fall
    through to local-file validation or the local HTTP server.
    """
    if proj.get("playback_mode") == "online":
        if not proj.get("external_video_id"):
            raise RuntimeError(t("projects.online_source_missing", lang))
        return None, None

    video_path = proj.get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise RuntimeError(t("projects.file_missing", lang))
    normalized_video_path = normalize_video_path(video_path)
    if normalized_video_path != video_path:
        video_path = normalized_video_path
        proj["video_path"] = video_path
        update_project(proj)
    server_url = start_video_server(os.path.dirname(video_path), Path(video_path).name)
    return video_path, server_url


def _handle_project_open_query(lang: str) -> None:
    """Fallback for cover links when the client-side relay is unavailable."""
    raw_project_id = st.query_params.get("open_project")
    if not raw_project_id:
        return
    if isinstance(raw_project_id, list):
        raw_project_id = raw_project_id[-1] if raw_project_id else ""
    project_id = str(raw_project_id)
    del st.query_params["open_project"]
    project = next(
        (item for item in (st.session_state.projects or []) if item.get("id") == project_id),
        None,
    )
    if project is not None:
        _select_project(project, lang, rerun=False)


def _select_project(proj, lang, *, rerun: bool = True):
    pid = proj["id"]
    if has_subtitles(proj):
        subtitles = load_project_subtitles(pid)
        if subtitles is None:
            st.warning(t("projects.load_failed", lang))
            return
        try:
            video_path, server_url = _resolve_project_playback(proj, lang)
        except RuntimeError as exc:
            st.warning(str(exc))
            return
        except OSError as exc:
            st.error(t("projects.player_start_failed", lang, error=str(exc)))
            return
        st.session_state.subtitles = subtitles
        st.session_state.video_path = video_path
        st.session_state.audio_path = proj.get("audio_path")
        st.session_state.server_url = server_url
        st.session_state.processed = True
        st.session_state.processing = False
        st.session_state.current_project = proj
        st.session_state.selected_project_id = pid
        st.session_state.show_new_task = False
        st.session_state.scroll_main_to_top = True
        if rerun:
            st.rerun()
    else:
        st.session_state.current_project = proj
        st.session_state.selected_project_id = pid
        st.session_state.processed = False
        st.session_state.show_new_task = False
        st.session_state.scroll_main_to_top = True
        if rerun:
            st.rerun()


# ---------------------------------------------------------------------------
# LEFT COLUMN: compact project list
# ---------------------------------------------------------------------------

def _render_project_list(lang):
    # Deletion is intentionally disabled while a pipeline owns files. Removing
    # them mid-import would let the background worker recreate partial data.
    tasks = _active_tasks_for_app()
    running = [tsk for tsk in tasks.values() if not tsk["completed"]]
    note_agent.configure(WORK_DIR)
    note_running = note_agent.has_running_tasks()
    delete_busy = bool(running) or note_running
    title_col, clear_col = st.columns([5, 1], vertical_alignment="center")
    with title_col:
        st.subheader(t("projects.header", lang))
    with clear_col:
        with st.popover("🗑️", help=t("projects.delete_all", lang), width="stretch"):
            st.warning(t("projects.delete_all_confirm", lang))
            st.caption(t("projects.delete_all_note", lang))
            if delete_busy:
                st.info(t("projects.delete_all_busy", lang))
            acknowledged = st.checkbox(
                t("projects.delete_all_ack", lang), key="delete_all_acknowledged",
            )
            if st.button(
                t("projects.delete_all", lang), key="delete_all_learning_data",
                type="primary", width="stretch",
                disabled=delete_busy or not acknowledged,
            ):
                result = _clear_all_learning_data()
                st.toast(t("projects.delete_all_done", lang, n=result["projects"]), icon="🗑️")
                st.rerun()
    with st.expander(
        t("tasks.center", lang, n=len(tasks)),
        expanded=any(not task.get("completed") for task in tasks.values()),
    ):
        _render_task_center(tasks, lang)
    # ---- Active background pipeline tasks ----
    # Evict old consumed tasks (>60s after completion)
    now = time.time()
    for tid in list(tasks.keys()):
        tsk = tasks[tid]
        if tsk["completed"] and tsk.get("consumed") and now - tsk.get("started_at", now) > 60:
            del tasks[tid]

    running = [tsk for tsk in tasks.values() if not tsk["completed"]]
    # Show new (project-less) tasks as virtual cells at top
    for task in running:
        if task.get("project_id"):
            continue
        with st.container(border=True):
            label = task.get("filename") or task.get("step_label", "Processing")
            st.caption(f"🔄 **{label[:35]}** — {_task_label(task, lang)}")
            st.progress(min(0.99, task.get("progress", 0.0) or 0.0))

    search_query = str(st.session_state.get("project_search", "")).strip().lower()

    projects = st.session_state.projects or []
    if search_query:
        filtered = [
            p for p in projects
            if search_query in p["title"].lower()
            or (p.get("custom_title") or "").lower().find(search_query) >= 0
            or search_query in p.get("target_lang", "")
        ]
    else:
        filtered = projects

    if not filtered and not running:
        st.caption(t("projects.empty", lang))
        return

    active_project_ids = {
        tsk["project_id"] for tsk in running if tsk.get("project_id")
    }

    for proj in sorted(filtered, key=lambda x: x.get("created_at", ""), reverse=True):
        pid = proj["id"]
        display_title = proj.get("custom_title") or proj["title"]
        if pid in active_project_ids:
            status_icon = "🔄"
        elif proj.get("status") == "processing":
            status_icon = "⚠️"
        else:
            status_icon = _STATUS_ICONS.get(proj.get("status", "legacy"), "❓")
        created_at = _format_project_created_at(proj.get("created_at", ""))
        is_selected = st.session_state.selected_project_id == pid

        # Everything belonging to a project lives in one visual cell. This
        # keeps several simultaneous imports readable instead of detaching
        # progress bars from their project title.
        with st.container(border=True):
            if st.button(
                f"{status_icon}  {display_title[:34]}",
                key=f"load_{pid}",
                width="stretch",
                type="primary" if is_selected else "secondary",
                help=display_title,
            ):
                _select_project(proj, lang)
            st.caption(f"🕒 {t('projects.created_at', lang)} · {created_at}")
            if pid in active_project_ids:
                _render_project_card_progress_fragment(pid, lang)


@st.fragment(run_every=2)
def _render_project_card_progress_fragment(project_id: str, lang: str) -> None:
    """Refresh only one active project's card, never the learning player."""
    task = next(
        (item for item in _active_tasks_for_app().values()
         if item.get("project_id") == project_id and not item["completed"]),
        None,
    )
    if task is None:
        return
    st.caption(_task_label(task, lang))
    st.progress(min(0.99, task.get("progress", 0.0) or 0.0))


def _render_task_center(tasks: dict[str, dict], lang: str) -> None:
    """Compact durable task history with safe cancellation and retry."""
    if not tasks:
        st.caption(t("tasks.empty", lang))
        return
    ordered = sorted(
        tasks.items(), key=lambda item: item[1].get("started_at", 0), reverse=True,
    )
    for task_key, task in ordered[:12]:
        stage = task.get("stage") or "starting"
        if stage == "complete":
            icon = "✅"
        elif stage in {"failed", "interrupted"}:
            icon = "❌"
        elif stage == "cancelled":
            icon = "⏹"
        else:
            icon = "🔄"
        name = str(task.get("filename") or task.get("project_id") or task["task_id"])
        st.markdown(f"{icon} **{html_lib.escape(name[:34])}**")
        st.caption(_task_label(task, lang))
        if not task.get("completed"):
            st.progress(min(0.99, float(task.get("progress", 0) or 0)))
            if st.button(
                t("tasks.cancel", lang), key=f"cancel_task_{task_key}",
                width="stretch", disabled=bool(task.get("cancel_requested")),
            ):
                if pipeline_runner.cancel_task(task_key):
                    st.toast(t("tasks.cancel_requested", lang))
                    st.rerun()
        elif stage in {"failed", "cancelled", "interrupted"}:
            if task.get("error"):
                st.caption(str(task["error"])[:160])
            can_retry = pipeline_runner.can_retry_task(task_key)
            if st.button(
                t("tasks.retry", lang), key=f"retry_task_{task_key}",
                width="stretch", disabled=not can_retry,
                help=None if can_retry else t("tasks.retry_after_restart", lang),
            ):
                if pipeline_runner.retry_task(task_key, _get_session_token()):
                    st.toast(t("tasks.retry_started", lang), icon="🔄")
                    st.rerun()
        st.divider()


# ---------------------------------------------------------------------------
# RIGHT COLUMN views
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _local_thumbnail_data_uri(path: str, modified_ns: int) -> str:
    """Encode a local cover once so its real HTML image can be a link."""
    del modified_ns  # Included in the cache key to invalidate replaced covers.
    mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _render_welcome(lang):
    projects = st.session_state.projects or []
    st.markdown(
        '<div class="home-intro">'
        f'<div class="product-eyebrow">{html_lib.escape(t("app.eyebrow", lang))}</div>'
        f'<h1>{html_lib.escape(t("app.start_title", lang))}</h1>'
        f'<p>{html_lib.escape(t("app.start_body", lang))}</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not projects:
        st.info(t("projects.welcome_hint", lang))
        return

    st.subheader(t("app.continue_learning", lang))
    st.caption(t("app.recent_projects", lang))
    query = str(st.session_state.get("project_search", "")).strip().lower()
    visible_projects = [
        project for project in projects
        if not query or query in (project.get("custom_title") or project.get("title") or "").lower()
    ]
    visible_projects = sorted(
        visible_projects, key=lambda item: item.get("created_at", ""), reverse=True,
    )[:9]
    if not visible_projects:
        st.caption(t("projects.empty", lang))
        return

    with st.container(key="recent_gallery"):
        columns = st.columns(3, gap="medium")
        for index, project in enumerate(visible_projects):
            with columns[index % 3]:
                pid = project["id"]
                title = project.get("custom_title") or project["title"]
                mode_key = "online" if project.get("playback_mode") == "online" else "local"
                mode_label = t(f"input.playback_mode_{mode_key}", lang).replace("（默认）", "").replace(" (default)", "")
                thumbnail_path = project.get("thumbnail_path")
                thumbnail_source = (
                    thumbnail_path
                    if thumbnail_path and os.path.isfile(thumbnail_path)
                    else None
                )
                if thumbnail_source is None:
                    video_id = youtube_video_id(project.get("source_url") or "")
                    if video_id:
                        thumbnail_source = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
                with st.container(key=f"recent_card_{pid}"):
                    open_label = t("projects.open_project", lang, title=title)
                    with st.container(key=f"cover_hit_{pid}"):
                        if thumbnail_source:
                            if os.path.isfile(thumbnail_source):
                                thumbnail_source = _local_thumbnail_data_uri(
                                    thumbnail_source, os.stat(thumbnail_source).st_mtime_ns,
                                )
                            st.markdown(
                                '<div class="gallery-cover">'
                                f'<img class="gallery-cover-image" src="{html_lib.escape(thumbnail_source)}" '
                                f'alt="{html_lib.escape(title)}"></div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                '<div class="library-thumbnail"></div>',
                                unsafe_allow_html=True,
                            )
                        if st.button(open_label, key=f"cover_button_{pid}", width="stretch"):
                            _select_project(project, lang)
                    st.markdown(
                        f'<span class="gallery-mode-label">{html_lib.escape(mode_label)}</span>',
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        title, key=f"gallery_title_{pid}", type="tertiary", width="stretch",
                    ):
                        _select_project(project, lang)
                    created_at = _format_project_created_at(project.get("created_at", ""))
                    status_key = project.get("status", "legacy")
                    status_text = t(f"projects.status_{status_key}", lang)
                    if status_text == f"projects.status_{status_key}":
                        status_text = status_key
                    st.caption(f"{status_text} · {created_at}")


def _render_new_task_form(lang, presets):
    st.title(t("input.source_title", lang))
    st.caption(t("input.source_hint", lang))

    with st.form("new_task_form", border=True):
        url = st.text_input(
            t("input.url_label", lang),
            placeholder=t("input.url_placeholder", lang),
            key="url_input",
        )
        playback_mode = st.radio(
            t("input.playback_mode", lang),
            # Keep the safest, fully featured path first.  Streamlit selects
            # the first radio option by default, which must never silently
            # create a caption-only online project.
            options=["local", "online"],
            format_func=lambda value: t(f"input.playback_mode_{value}", lang),
            horizontal=True,
            help=t("input.playback_mode_help", lang),
            key="new_task_playback_mode",
        )
        mode_local, mode_online = st.columns(2)
        with mode_local:
            st.caption(f"💾 {t('input.playback_mode_local_detail', lang)}")
        with mode_online:
            st.caption(f"🌐 {t('input.playback_mode_online_detail', lang)}")
        prefer_native_target_subtitles = st.checkbox(
            t("input.native_target_subtitles", lang),
            value=True,
            help=t("input.native_target_subtitles_help", lang),
            key="prefer_native_target_subtitles",
        )
        uploaded = st.file_uploader(
            t("input.upload_label", lang),
            type=["mp4", "mkv", "webm", "mov", "avi"],
            key="file_upload",
        )
        st.caption(t("app.supported_sources", lang))
        submitted = st.form_submit_button(
            t("input.process_button", lang), type="primary", width="stretch",
        )

    if submitted:
        source = url.strip()
        if not source and uploaded is None:
            st.warning(t("input.source_required", lang))
            return
        if source and uploaded is not None:
            st.warning(t("input.choose_one_source", lang))
            return
        if source and not source.startswith(("http://", "https://")) and not os.path.isfile(source):
            st.warning(t("input.path_missing", lang))
            return
        if playback_mode == "online" and uploaded is not None:
            st.warning(t("input.online_url_only", lang))
            return
        external_video_id = youtube_video_id(source) if playback_mode == "online" else None
        if playback_mode == "online" and not external_video_id:
            st.warning(t("input.online_youtube_only", lang))
            return

        preset = _active_preset()
        setup_error = (
            None if prefer_native_target_subtitles
            else _translation_setup_error(preset)
        )
        if setup_error:
            st.error(setup_error)
            return

        upload_path = None
        original_name = ""
        if uploaded is not None:
            original_name = Path(uploaded.name).name
            suffix = Path(original_name).suffix.lower()
            upload_dir = os.path.join(WORK_DIR, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            upload_path = os.path.join(upload_dir, f"{uuid.uuid4().hex}{suffix}")
            with open(upload_path, "wb") as f:
                f.write(uploaded.getbuffer())
        assert preset is not None
        model_name = st.session_state.model_override or preset["model_name"]
        target_lang_code = st.session_state.target_lang
        if uploaded is not None:
            pending_title = Path(original_name).stem
        else:
            parsed = urllib.parse.urlparse(source)
            host = (parsed.hostname or "").removeprefix("www.")
            video_id = urllib.parse.parse_qs(parsed.query).get("v", [""])[0]
            source_label = f"YouTube · {video_id}" if "youtube.com" in host and video_id else (host or source)
            pending_title = t("projects.pending_title", lang, source=source_label)
        model_used = f"{preset['engine_type']}:{model_name}"
        pending_project = create_project(
            upload_path,
            None,
            target_lang_code,
            model_used,
            title=pending_title,
            source_url=source or None,
            playback_mode=playback_mode,
            external_video_id=external_video_id,
        )
        pending_project["prefer_native_target_subtitles"] = prefer_native_target_subtitles
        append_project(pending_project)
        _start_pipeline_task({
            "kind": "new",
            "project_id": pending_project["id"],
            "url": source,
            "upload_path": upload_path,
            "proxy": st.session_state.proxy,
            "asr_model": st.session_state.asr_model,
            "engine_type": preset["engine_type"],
            "model_name": model_name,
            "api_base": preset["api_base"],
            "api_key": _effective_preset_api_key(preset),
            "target_lang": TARGET_LANGUAGES.get(target_lang_code, target_lang_code),
            "target_lang_code": target_lang_code,
            "max_tokens": preset.get("max_tokens", 32768),
            "max_workers": st.session_state.max_workers,
            "filename": original_name if uploaded else source,
            "display_title": Path(original_name).stem if uploaded else "",
            "playback_mode": playback_mode,
            "prefer_native_target_subtitles": prefer_native_target_subtitles,
        })
        st.toast(t("pipeline.task_started", lang), icon="🚀")
        st.session_state.projects = load_projects()
        # A new task stays in its own library card while it runs.  Keeping the
        # main panel on the welcome state avoids a detached duplicate progress
        # panel and makes concurrent project progress easy to compare.
        st.session_state.current_project = None
        st.session_state.selected_project_id = None
        st.session_state.show_new_task = False
        st.rerun()


def _render_processing_status(task: dict, lang: str) -> None:
    """Live status panel shown when a project has a pipeline running."""
    step = task.get("step", 0)
    label = _task_label(task, lang)
    progress = float(task.get("progress", 0.0) or 0.0)
    is_failed = bool(task.get("error"))

    with st.container(border=True):
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        with c1:
            icon = "❌" if is_failed else "🔄"
            st.subheader(f"{icon}  {t('pipeline.running', lang)}")
            st.caption(t("pipeline.step_progress", lang, step=step, label=label))
        with c2:
            if task.get("batch_total"):
                st.metric(
                    t("pipeline.batch_progress", lang,
                      current=task["batch_current"], total=task["batch_total"]),
                    "",
                )

        if is_failed:
            st.error(task["error"])
        else:
            st.progress(min(0.99, progress), text=label)

        elapsed = int(time.time() - task.get("started_at", time.time()))
        mins, secs = divmod(elapsed, 60)
        st.caption(f"⏱️ {mins:02d}:{secs:02d}")


@st.fragment(run_every=2)
def _render_project_progress_fragment(project_id: str, lang: str) -> None:
    """Render live progress only inside the project that owns the task."""
    task = next(
        (item for item in _active_tasks_for_app().values()
         if item.get("project_id") == project_id and not item["completed"]),
        None,
    )
    if task:
        _render_processing_status(task, lang)


def _queue_project_task(
    kind: str, proj: dict, title: str, *, cue_ids: Optional[list[str]] = None,
) -> bool:
    """Validate the active model once, then queue a safe project operation."""
    preset = _active_preset()
    setup_error = _translation_setup_error(preset)
    if setup_error:
        st.error(setup_error)
        return False
    assert preset is not None
    target_lang_code = st.session_state.target_lang
    params = {
        "kind": kind,
        "project_id": proj["id"],
        "video_path": proj["video_path"],
        "audio_path": proj.get("audio_path"),
        "engine_type": preset["engine_type"],
        "model_name": st.session_state.model_override or preset["model_name"],
        "api_base": preset["api_base"],
        "api_key": _effective_preset_api_key(preset),
        "target_lang": TARGET_LANGUAGES.get(target_lang_code, target_lang_code),
        "target_lang_code": target_lang_code,
        "max_tokens": preset.get("max_tokens", 32768),
        "max_workers": st.session_state.max_workers,
        "filename": title,
        "cue_ids": list(cue_ids or []),
    }
    if kind == "reprocess":
        params["asr_model"] = st.session_state.asr_model
    _start_pipeline_task(params)
    return True


def _retry_new_project(proj: dict, title: str) -> bool:
    """Retry a failed import without creating a duplicate project."""
    preset = _active_preset()
    setup_error = _translation_setup_error(preset)
    if setup_error:
        st.error(setup_error)
        return False
    assert preset is not None
    source_url = (proj.get("source_url") or "").strip()
    video_path = proj.get("video_path")
    if not source_url and not (video_path and os.path.exists(video_path)):
        st.error(t("projects.file_missing", lang))
        return False
    target_lang_code = st.session_state.target_lang
    proj["status"] = "processing"
    proj["error"] = None
    proj["target_lang"] = target_lang_code
    update_project(proj)
    _start_pipeline_task({
        "kind": "new",
        "project_id": proj["id"],
        "url": source_url,
        "upload_path": None if source_url else video_path,
        "proxy": st.session_state.proxy,
        "asr_model": st.session_state.asr_model,
        "engine_type": preset["engine_type"],
        "model_name": st.session_state.model_override or preset["model_name"],
        "api_base": preset["api_base"],
        "api_key": _effective_preset_api_key(preset),
        "target_lang": TARGET_LANGUAGES.get(target_lang_code, target_lang_code),
        "target_lang_code": target_lang_code,
        "max_tokens": preset.get("max_tokens", 32768),
        "max_workers": st.session_state.max_workers,
        "filename": title,
        "display_title": title,
        "playback_mode": proj.get("playback_mode", "local"),
        "prefer_native_target_subtitles": proj.get(
            "prefer_native_target_subtitles", True,
        ),
    })
    st.session_state.projects = load_projects()
    st.session_state.current_project = next(
        (p for p in st.session_state.projects if p["id"] == proj["id"]), proj,
    )
    return True


def _render_project_action_bar(
    proj: dict, title: str, pid: str, lang: str,
    subtitles: Optional[list[dict]] = None,
) -> None:
    """Compact watch-page actions shown below the player, like media-site chips."""
    note_agent.configure(WORK_DIR)
    active_note_id = st.session_state.get("note_job_id")
    active_note = note_agent.get(active_note_id) if active_note_id else None
    note_busy = bool(
        active_note and not active_note.get("completed")
        and str(active_note.get("project_id") or "") == str(pid)
    )
    action_columns = st.columns([1.1, 1.25, 4.8], vertical_alignment="center")
    with action_columns[0]:
        if subtitles:
            st.download_button(
                t("results.download_btn", lang),
                json.dumps(subtitles, ensure_ascii=False, indent=2),
                file_name="subtitles.json", mime="application/json",
                width="stretch",
            )
    with action_columns[1]:
        with st.popover(f"⚙ {t('projects.manage', lang)}", width="stretch"):
            st.caption(t("projects.manage_details", lang))
            st.caption(t("projects.action_rename", lang))
            new_title = st.text_input(
                t("projects.custom_title_placeholder", lang),
                value=proj.get("custom_title") or proj["title"],
                key=f"rename_{pid}", label_visibility="collapsed",
            )
            if st.button(t("sidebar.preset_save_btn", lang), key=f"save_{pid}", width="stretch"):
                update_project_title(pid, new_title if new_title.strip() else None)
                st.session_state.projects = load_projects()
                updated = next((p for p in st.session_state.projects if p["id"] == pid), proj)
                st.session_state.current_project = updated
                st.rerun()

            st.divider()
            st.caption(t("projects.manage_processing", lang))
            if has_subtitles(proj) and st.button(
                f"↻ {t('projects.action_retranslate', lang)}",
                key=f"detail_retrans_{pid}",
                help=t("projects.retranslate", lang),
                width="stretch",
            ):
                if _queue_project_task("retranslate", proj, title):
                    st.rerun()
            if proj.get("playback_mode") != "online" and st.button(
                f"⟳ {t('projects.action_reprocess', lang)}",
                key=f"detail_reproc_{pid}",
                help=t("projects.reprocess", lang),
                width="stretch",
            ):
                if proj.get("video_path") and os.path.exists(proj["video_path"]):
                    if _queue_project_task("reprocess", proj, title):
                        st.rerun()
                else:
                    st.warning(t("projects.file_missing", lang))

            st.divider()
            st.warning(t("projects.delete_confirm", lang))
            st.caption(t("projects.delete_note", lang))
            if note_busy:
                st.info(t("projects.delete_all_busy", lang))
            if st.button(
                t("projects.delete", lang), key=f"del_{pid}",
                type="primary", width="stretch", disabled=note_busy,
            ):
                if active_note_id:
                    note_agent.discard(active_note_id)
                delete_project(pid)
                wordbook_store.remove_if(project_id=pid)
                favorites_store.remove_if(project_id=pid)
                notes_store.remove_if(project_id=pid)
                st.session_state.projects = load_projects()
                st.session_state.selected_project_id = None
                st.session_state.current_project = None
                st.session_state.processed = False
                st.session_state.subtitles = None
                _close_note_composer()
                st.rerun()


def _close_note_composer() -> None:
    st.session_state.show_note_composer = False
    st.session_state.note_draft = None
    st.session_state.note_result = None
    st.session_state.note_job_id = None
    st.session_state.note_job_error = None
    st.session_state.note_composer_nonce += 1


def _note_entry(
    project: dict, draft: dict, *, title: str, body: str,
    raw_idea: str = "", summary: str = "", tags: list[str] | None = None,
    key_points: list[str] | None = None,
    vocabulary: list[dict] | None = None,
    refined_by: str = "manual",
) -> dict:
    return {
        "project_id": project.get("id", ""),
        "project_title": project.get("custom_title") or project.get("title") or "",
        "time": max(0.0, float(draft.get("time", 0.0) or 0.0)),
        "subtitle_id": str(draft.get("subtitle_id") or ""),
        "source_text": str(draft.get("source_text") or "").strip(),
        "source_translation": str(draft.get("source_translation") or "").strip(),
        "title": title.strip() or t("notes.untitled", lang),
        "body": body.strip(),
        "raw_idea": raw_idea.strip(),
        "summary": summary.strip(),
        "tags": tags or [],
        "key_points": key_points or [],
        "vocabulary": vocabulary or [],
        "refined_by": refined_by,
        "updated_at": datetime.now().astimezone().isoformat(),
    }


@st.fragment(run_every=1)
def _poll_note_job_fragment(task_id: str) -> None:
    note_agent.configure(WORK_DIR)
    task = note_agent.get(task_id)
    if not task:
        st.session_state.note_job_id = None
        st.session_state.note_job_error = t("notes.job_missing", lang)
        st.rerun(scope="app")
    if not task.get("completed"):
        stage = task.get("stage", "queued")
        st.info(t(f"notes.stage_{stage}", lang))
        st.progress(0.35 if stage == "transcribing" else 0.72 if stage == "refining" else 0.08)
        return
    st.session_state.note_job_id = None
    if task.get("error"):
        st.session_state.note_job_error = task["error"]
    else:
        result = dict(task.get("result") or {})
        result["raw_text"] = task.get("raw_text") or ""
        st.session_state.note_result = result
        st.session_state.note_job_error = None
    note_agent.discard(task_id)
    st.rerun(scope="app")


def _render_note_composer(project: dict, lang: str) -> None:
    if not st.session_state.get("show_note_composer"):
        return
    draft = st.session_state.get("note_draft") or {
        "project_id": project.get("id"), "time": 0.0,
        "source_text": "", "source_translation": "", "subtitle_id": "",
    }
    # A note remains tied to the video where it was captured.  Switching
    # projects hides (rather than misattributes) the in-progress draft.
    if str(draft.get("project_id") or "") != str(project.get("id") or ""):
        return
    nonce = int(st.session_state.get("note_composer_nonce", 0))
    with st.container(border=True, key="learning_note_composer"):
        st.subheader(t("notes.composer_title", lang))
        st.caption(t("notes.composer_hint", lang))
        source_text = str(draft.get("source_text") or "").strip()
        source_translation = str(draft.get("source_translation") or "").strip()
        if source_text:
            st.markdown(f"> {source_text}")
        if source_translation:
            st.caption(source_translation)

        job_id = st.session_state.get("note_job_id")
        if job_id:
            _poll_note_job_fragment(job_id)
            return

        job_error = st.session_state.get("note_job_error")
        if job_error:
            st.error(t("notes.job_failed", lang, error=job_error))

        result = st.session_state.get("note_result")
        if result:
            warning = str(result.get("warning") or "")
            if warning:
                st.warning(t("notes.draft_warning", lang))
            with st.form(f"note_result_form_{nonce}"):
                title = st.text_input(
                    t("notes.note_title", lang), value=str(result.get("title") or "")[:120],
                )
                body = st.text_area(
                    t("notes.note_body", lang), value=str(result.get("body") or ""), height=180,
                )
                tags_text = st.text_input(
                    t("notes.tags", lang), value=", ".join(result.get("tags") or []),
                )
                save_col, retry_col, cancel_col = st.columns([1.4, 1, 1])
                save = save_col.form_submit_button(
                    t("notes.save", lang), type="primary", width="stretch",
                )
                retry = retry_col.form_submit_button(t("notes.retry", lang), width="stretch")
                cancel = cancel_col.form_submit_button(t("notes.cancel", lang), width="stretch")
            if save:
                if not body.strip():
                    st.error(t("notes.body_required", lang))
                else:
                    tags = [item.strip() for item in re.split(r"[,，]", tags_text) if item.strip()]
                    notes_store.add(_note_entry(
                        project, draft, title=title, body=body,
                        raw_idea=str(result.get("raw_text") or ""),
                        summary=str(result.get("summary") or ""),
                        tags=tags,
                        key_points=list(result.get("key_points") or []),
                        vocabulary=list(result.get("vocabulary") or []),
                        refined_by=str(result.get("refined_by") or "draft"),
                    ))
                    _close_note_composer()
                    st.toast(t("notes.saved", lang), icon="📝")
                    st.rerun()
            if retry:
                st.session_state.note_result = None
                st.session_state.note_job_error = None
                st.rerun()
            if cancel:
                _close_note_composer()
                st.rerun()
            return

        audio = st.audio_input(
            t("notes.record", lang), sample_rate=16000,
            key=f"note_audio_{nonce}", help=t("notes.record_help", lang),
        )
        raw_text = st.text_area(
            t("notes.raw_idea", lang),
            key=f"note_raw_{nonce}",
            placeholder=t("notes.raw_placeholder", lang), height=120,
        ) or ""
        process_col, direct_col, cancel_col = st.columns([1.5, 1.15, 1])
        process = process_col.button(
            t("notes.refine", lang), type="primary", width="stretch",
            disabled=audio is None and not raw_text.strip(), key=f"note_refine_{nonce}",
        )
        direct = direct_col.button(
            t("notes.save_direct", lang), width="stretch",
            disabled=not raw_text.strip(), key=f"note_direct_{nonce}",
        )
        cancel = cancel_col.button(t("notes.cancel", lang), width="stretch", key=f"note_cancel_{nonce}")

        if process:
            audio_path = ""
            if audio is not None:
                content = audio.getvalue()
                if len(content) > 20 * 1024 * 1024:
                    st.error(t("notes.audio_too_large", lang))
                    return
                audio_path = note_agent.save_audio(content, WORK_DIR, project["id"])
            preset = _active_preset() or {}
            model = st.session_state.model_override or preset.get("model_name") or ""
            note_agent.configure(WORK_DIR)
            st.session_state.note_job_id = note_agent.start({
                "project_id": project["id"],
                "audio_path": audio_path,
                "raw_text": raw_text,
                "context": draft,
                "asr_model": st.session_state.asr_model,
                "engine": preset.get("engine_type") or "",
                "model": model,
                "api_base": preset.get("api_base") or "",
                "api_key": _effective_preset_api_key(preset) if preset else "",
                "max_tokens": min(int(preset.get("max_tokens") or 1400), 3000),
                "ui_language": lang,
            })
            st.session_state.note_job_error = None
            st.rerun()
        if direct:
            compact = " ".join(raw_text.split())
            notes_store.add(_note_entry(
                project, draft, title=compact[:36], body=raw_text,
                raw_idea=raw_text, summary=compact[:160], refined_by="manual",
            ))
            _close_note_composer()
            st.toast(t("notes.saved", lang), icon="📝")
            st.rerun()
        if cancel:
            _close_note_composer()
            st.rerun()


def _render_project_detail(lang, presets):
    proj = st.session_state.current_project
    pid = proj["id"]
    title = proj.get("custom_title") or proj["title"]
    flag = LANG_FLAGS.get(proj.get("target_lang", ""), "")

    _sync_browser_title(title)

    # If a background task is running for this project, show live status
    _tasks = _active_tasks_for_app()
    _active = next(
        (tsk for tsk in _tasks.values()
         if tsk.get("project_id") == pid and not tsk["completed"]),
        None,
    )
    if _active:
        st.markdown(
            f'<div class="watch-title">{html_lib.escape(f"{flag} {title}".strip())}</div>',
            unsafe_allow_html=True,
        )
        st.caption(t("projects.progress_in_card", lang))
        return

    if not (has_subtitles(proj) and st.session_state.subtitles):
        st.markdown(
            f'<div class="watch-title">{html_lib.escape(f"{flag} {title}".strip())}</div>',
            unsafe_allow_html=True,
        )
        _render_project_action_bar(proj, title, pid, lang)
        st.divider()
        if proj.get("status") in {"failed", "processing"}:
            if proj.get("status") == "failed":
                st.error(t("projects.task_failed", lang, error=proj.get("error") or t("pipeline.status.failed", lang)))
            else:
                st.warning(t("projects.task_interrupted", lang))
            if st.button(f"↻  {t('projects.retry', lang)}", type="primary", key=f"retry_{pid}"):
                if _retry_new_project(proj, title):
                    st.rerun()
            return
        st.info(t("projects.no_subtitles_hint", lang))
        if proj.get("video_path") and os.path.exists(proj["video_path"]):
            if st.button(f"⚡  {t('projects.process_now', lang)}", type="primary"):
                if _queue_project_task("reprocess", proj, title):
                    st.rerun()
        else:
            st.warning(t("projects.file_missing", lang))
        return

    subtitles = st.session_state.subtitles or load_project_subtitles(pid)
    is_online = proj.get("playback_mode") == "online"
    video_path = proj.get("video_path") or st.session_state.video_path
    server_url = None
    video_url = ""
    if is_online:
        if not proj.get("external_video_id"):
            st.error(t("projects.online_source_missing", lang))
            return
        try:
            # Online projects still need the local lookup bridge, but do not
            # download or proxy the video itself.
            server_url = start_lookup_server()
        except OSError as exc:
            st.error(t("projects.player_start_failed", lang, error=str(exc)))
            return
        video_filename = t("results.online_source", lang)
    else:
        if not video_path or not os.path.exists(video_path):
            st.error(t("projects.file_missing", lang))
            return
        normalized_video_path = normalize_video_path(video_path)
        if normalized_video_path != video_path:
            video_path = normalized_video_path
            proj["video_path"] = video_path
            update_project(proj)
            st.session_state.current_project = proj
        server_url = st.session_state.server_url
        if not server_url or st.session_state.video_path != video_path:
            try:
                server_url = start_video_server(os.path.dirname(video_path), Path(video_path).name)
            except OSError as exc:
                st.error(t("projects.player_start_failed", lang, error=str(exc)))
                return
            st.session_state.server_url = server_url
            st.session_state.video_path = video_path
            st.session_state.audio_path = proj.get("audio_path")
            st.session_state.subtitles = subtitles
            st.session_state.processed = True
        video_filename = Path(video_path).name
        video_url = f"{server_url}/{urllib.parse.quote(video_filename)}"

    pending_ai_lookup = st.session_state.get("pending_ai_lookup")
    if pending_ai_lookup and pending_ai_lookup.get("project_id") != pid:
        pending_ai_lookup = None
    player_html = build_player_html(
        video_url, subtitles, lang,
        project_id=pid,
        notes=notes_store.load(),
        wordbook=wordbook_store.load(),
        initial_seek=st.session_state.get("seek_to", 0.0),
        playback_mode=proj.get("playback_mode", "local"),
        external_video_id=proj.get("external_video_id") or "",
        ai_lookup=pending_ai_lookup,
        target_lang_code=proj.get("target_lang") or st.session_state.target_lang,
        lookup_api_base=server_url or "",
        subtitle_offset=proj.get("subtitle_offset", 0.0),
    )
    if pending_ai_lookup:
        st.session_state.pending_ai_lookup = None
    st.session_state.seek_to = 0.0
    st.iframe(player_html, width="stretch", height=680)
    _render_note_composer(proj, lang)

    st.markdown(
        f'<div class="watch-title">{html_lib.escape(f"{flag} {title}".strip())}</div>',
        unsafe_allow_html=True,
    )
    if is_online:
        st.markdown(
            f'<div class="watch-meta">{html_lib.escape(t("results.info_online", lang, n=len(subtitles)))}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(t("results.info", lang, filename=video_filename, n=len(subtitles), url=server_url))
    timing_source = proj.get("timing_source")
    if timing_source in {"word", "cue"}:
        st.caption(f"⏱ {t(f'results.timing_{timing_source}', lang)}")
    if proj.get("subtitle_source") == "native_target":
        native_language = (
            proj.get("native_subtitle_language")
            or TARGET_LANGUAGES.get(proj.get("target_lang", ""), proj.get("target_lang", ""))
        )
        st.caption(t("projects.native_subtitles_used", lang, language=native_language))
    _render_project_action_bar(proj, title, pid, lang, subtitles)

    with st.expander(t("results.table_header", lang)):
        _render_transcript_sheet(subtitles, lang)
        with st.expander(t("editor.title", lang)):
            _render_subtitle_editor(pid, subtitles, lang)


# ---------------------------------------------------------------------------
# Pipeline execution (background-thread driver lives near top of file)
# ---------------------------------------------------------------------------


# ===========================================================================
# Notes page (AI/manual learning notes + wordbook)
# ===========================================================================

def notes_page():
    """Global learning notes, with the existing wordbook kept as reference."""
    _apply_product_theme()
    _install_sidebar_toggle(lang)
    projects_list = load_projects()
    proj_map = {p["id"]: p for p in projects_list}

    st.title(t("notes.title", lang))
    st.caption(t("notes.page_hint", lang))

    tab_notes, tab_word = st.tabs([
        f"📝 {t('notes.all_notes', lang)}",
        f"📖 {t('collections.wordbook', lang)}",
    ])

    with tab_notes:
        with st.expander(f"＋ {t('notes.new_manual', lang)}"):
            project_options = [""] + [project["id"] for project in projects_list]
            project_names = {
                "": t("notes.no_project", lang),
                **{
                    project["id"]: project.get("custom_title") or project.get("title") or project["id"]
                    for project in projects_list
                },
            }
            with st.form("manual_note_form", clear_on_submit=True):
                selected_project = st.selectbox(
                    t("notes.related_project", lang), project_options,
                    format_func=lambda value: project_names.get(value, value),
                )
                manual_title = st.text_input(t("notes.note_title", lang))
                manual_body = st.text_area(t("notes.note_body", lang), height=160)
                manual_tags = st.text_input(t("notes.tags", lang))
                manual_save = st.form_submit_button(
                    t("notes.save", lang), type="primary", width="stretch",
                )
            if manual_save:
                if not manual_body.strip():
                    st.error(t("notes.body_required", lang))
                else:
                    related = proj_map.get(selected_project) or {
                        "id": "", "title": t("notes.no_project", lang),
                    }
                    tags = [item.strip() for item in re.split(r"[,，]", manual_tags) if item.strip()]
                    notes_store.add(_note_entry(
                        related, {"time": 0}, title=manual_title,
                        body=manual_body, raw_idea=manual_body,
                        summary=" ".join(manual_body.split())[:160], tags=tags,
                    ))
                    st.toast(t("notes.saved", lang), icon="📝")
                    st.rerun()

        search_n = st.text_input(
            t("collections.search", lang), key="notes_search",
            label_visibility="collapsed", placeholder=f"🔍 {t('notes.search', lang)}",
        ).strip().lower()
        note_items = [
            item for item in reversed(notes_store.load())
            if not search_n or search_n in " ".join([
                str(item.get("title") or ""), str(item.get("body") or ""),
                str(item.get("source_text") or ""), " ".join(item.get("tags") or []),
            ]).lower()
        ]
        st.caption(t("collections.count", lang, n=len(note_items)))
        if not note_items:
            st.info(t("notes.empty", lang))
        for note in note_items:
            _render_note_entry(note, proj_map, lang)

    with tab_word:
        words = wordbook_store.load()
        if not words:
            st.caption(t("collections.wordbook_empty", lang))
        else:
            search_w = st.text_input(
                t("collections.search", lang),
                key="wordbook_search",
                label_visibility="collapsed",
                placeholder=f"🔍 {t('collections.search_wordbook', lang)}",
            ).strip().lower()

            filtered_w = [
                w for w in reversed(words)
                if not search_w
                or search_w in (w.get("word", "")).lower()
                or search_w in (w.get("translation", "")).lower()
                or search_w in (w.get("context", "")).lower()
            ]

            st.caption(t("collections.count", lang, n=len(filtered_w)))

            for entry in filtered_w:
                pid = entry.get("project_id", "")
                proj = proj_map.get(pid)
                proj_title = (proj.get("custom_title") or proj.get("title", "?")) if proj else "?"
                _render_word_entry(entry, proj_title, lang)

def _render_note_entry(note: dict, proj_map: dict[str, dict], lang: str) -> None:
    project = proj_map.get(note.get("project_id", ""))
    if project:
        project_title = project.get("custom_title") or project.get("title") or project["id"]
    else:
        project_title = note.get("project_title") or t("notes.no_project", lang)
    with st.container(border=True, key=f"note_card_{note['id']}"):
        title_col, _ = st.columns([6, 1.2], vertical_alignment="center")
        title_col.subheader(note.get("title") or t("notes.untitled", lang))
        created = str(note.get("created_at") or "")[:10]
        seconds = int(float(note.get("time", 0) or 0))
        title_col.caption(f"{project_title} · {seconds // 60}:{seconds % 60:02d} · {created}")
        tags = [str(item) for item in note.get("tags", []) if str(item).strip()]
        if tags:
            st.caption(" · ".join(f"#{item}" for item in tags))
        if note.get("source_text"):
            st.markdown(f"> {note['source_text']}")
            if note.get("source_translation"):
                st.caption(note["source_translation"])
        st.markdown(note.get("body") or "")
        open_col, edit_col, delete_col, _ = st.columns([1.2, 1, 1, 4])
        if project and open_col.button(
            t("collections.open", lang), key=f"note_open_{note['id']}", width="stretch",
        ):
            _open_project_with_seek(project["id"], float(note.get("time", 0) or 0))
        with edit_col.popover(t("notes.edit", lang), width="stretch"):
            edited_title = st.text_input(
                t("notes.note_title", lang), value=note.get("title") or "",
                key=f"note_title_{note['id']}",
            )
            edited_body = st.text_area(
                t("notes.note_body", lang), value=note.get("body") or "", height=180,
                key=f"note_body_{note['id']}",
            )
            edited_tags = st.text_input(
                t("notes.tags", lang), value=", ".join(tags), key=f"note_tags_{note['id']}",
            )
            if st.button(t("notes.save_changes", lang), key=f"note_update_{note['id']}", width="stretch"):
                if edited_body.strip():
                    notes_store.update(
                        note["id"], title=edited_title.strip() or t("notes.untitled", lang),
                        body=edited_body.strip(),
                        tags=[item.strip() for item in re.split(r"[,，]", edited_tags) if item.strip()],
                        updated_at=datetime.now().astimezone().isoformat(),
                    )
                    st.rerun()
                else:
                    st.error(t("notes.body_required", lang))
        if delete_col.button(t("notes.delete", lang), key=f"note_delete_{note['id']}", width="stretch"):
            notes_store.remove(note["id"])
            st.rerun()


def _open_project_with_seek(project_id: str, seek_time: float):
    """Load a project into session state and switch to Home page."""
    projects_list = load_projects()
    proj = next((p for p in projects_list if p["id"] == project_id), None)
    if not proj:
        st.warning(t("projects.file_missing", lang))
        return
    if not has_subtitles(proj):
        st.warning(t("projects.no_subtitles_hint", lang))
        return
    subtitles = load_project_subtitles(project_id)
    if subtitles is None:
        st.warning(t("projects.load_failed", lang))
        return
    try:
        video_path, server_url = _resolve_project_playback(proj, lang)
    except RuntimeError as exc:
        st.warning(str(exc))
        return
    except OSError as exc:
        st.error(t("projects.player_start_failed", lang, error=str(exc)))
        return
    st.session_state.subtitles = subtitles
    st.session_state.video_path = video_path
    st.session_state.audio_path = proj.get("audio_path")
    st.session_state.processed = True
    st.session_state.processing = False
    st.session_state.current_project = proj
    st.session_state.selected_project_id = project_id
    st.session_state.show_new_task = False
    st.session_state.seek_to = seek_time or 0.0
    st.session_state.server_url = server_url
    st.session_state.scroll_main_to_top = True
    st.switch_page(_home_pg)


def _render_word_entry(entry: dict, proj_title: str, lang: str):
    """Render a wordbook entry card."""
    with st.container(border=True):
        c_word, c_proj, c_del = st.columns([5, 2, 1])
        with c_word:
            word_text = f"**{entry.get('word', '')}**"
            if entry.get("pos"):
                word_text += f" *({entry['pos']})*"
            if entry.get("translation"):
                word_text += f"  →  `{entry['translation']}`"
            st.write(word_text)
            if entry.get("pronunciation"):
                st.caption(f"/{entry['pronunciation']}/")
            if entry.get("other_meanings"):
                st.caption(f"other: {'; '.join(entry['other_meanings'])}")
            if entry.get("context"):
                st.caption(f"📝 {entry['context']}")
        with c_proj:
            st.caption(f"🎬 {proj_title[:20]}")
        with c_del:
            if st.button("🗑️", key=f"cw_{entry['id']}", help=t("projects.delete", lang)):
                wordbook_store.remove(entry["id"])
                st.rerun()

        c_open, _ = st.columns([1, 4])
        with c_open:
            if st.button(
                f"▶ {t('collections.open', lang)}",
                key=f"co_{entry['id']}",
                width="stretch",
            ):
                _open_project_with_seek(entry.get("project_id", ""), entry.get("time", 0))


# ===========================================================================
# Navigation
# ===========================================================================

_home_pg = st.Page(home_page, title=t("app.title", lang), icon="🏠")
_settings_pg = st.Page(settings_page, title=t("sidebar.header", lang), icon="⚙️")
_notes_pg = st.Page(notes_page, title=t("notes.title", lang), icon="📝")
pg = st.navigation([_home_pg, _settings_pg, _notes_pg])
pg.run()
