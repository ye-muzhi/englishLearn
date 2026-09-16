"""
i18n: UI text strings in English and Chinese, plus target language definitions.
"""
import streamlit as st

# Target languages for translation (language code -> display name)
TARGET_LANGUAGES: dict[str, str] = {
    "zh": "中文 (Chinese)",
    "ja": "日本語 (Japanese)",
    "ko": "한국어 (Korean)",
    "fr": "Français (French)",
    "de": "Deutsch (German)",
    "es": "Español (Spanish)",
    "pt": "Português (Portuguese)",
    "ru": "Русский (Russian)",
    "ar": "العربية (Arabic)",
    "th": "ไทย (Thai)",
    "vi": "Tiếng Việt (Vietnamese)",
    "en": "English",
}

# UI languages offered in the app
UI_LANGUAGES: dict[str, str] = {
    "en": "English",
    "zh": "中文",
}

# All UI strings, keyed by message_id -> {lang: text}
STRINGS: dict[str, dict[str, str]] = {
    # ---- App header ----
    "app.title": {
        "en": "EnglishLearn",
        "zh": "EnglishLearn",
    },
    "app.caption": {
        "en": "Enter a video URL or upload a local file. The app will extract audio, transcribe with Whisper, translate with LLM, and play back with synced bilingual subtitles.",
        "zh": "输入视频链接或上传本地文件。应用将提取音频、Whisper 转录、大模型翻译，并同步播放双语字幕。",
    },
    "app.eyebrow": {
        "en": "Language learning studio",
        "zh": "沉浸式语言学习工作台",
    },
    "app.start_title": {
        "en": "Turn any video into a lesson you can revisit.",
        "zh": "把任何视频，变成可反复学习的一课。",
    },
    "app.start_body": {
        "en": "Import a video, generate natural bilingual subtitles, then save the words and lines worth keeping.",
        "zh": "导入视频，生成自然的双语字幕，再收藏真正值得复习的单词和句子。",
    },
    "app.continue_learning": {
        "en": "Continue learning",
        "zh": "继续学习",
    },
    "app.recent_projects": {
        "en": "Your latest videos and lessons",
        "zh": "最近的视频与学习项目",
    },
    "app.supported_sources": {
        "en": "Supported sources: YouTube, Bilibili, direct MP4 links, local files, etc.",
        "zh": "支持来源：YouTube、Bilibili、直链 MP4、本地文件等。",
    },

    # ---- Input area ----
    "input.url_placeholder": {
        "en": "e.g. https://www.youtube.com/watch?v=... or a local path",
        "zh": "例如 https://www.youtube.com/watch?v=... 或本地路径",
    },
    "input.url_label": {
        "en": "Video URL",
        "zh": "视频链接",
    },
    "input.upload_label": {
        "en": "Or upload",
        "zh": "或上传",
    },
    "input.playback_mode": {
        "en": "Playback & processing mode",
        "zh": "播放与处理方式",
    },
    "input.playback_mode_online": {
        "en": "Online captions mode",
        "zh": "在线播放模式",
    },
    "input.playback_mode_local": {
        "en": "Local complete mode (default)",
        "zh": "本地完整模式（默认）",
    },
    "input.playback_mode_help": {
        "en": "Choose how this project is created. Local complete mode is selected by default.",
        "zh": "请选择项目的创建方式；默认使用本地完整模式。",
    },
    "input.playback_mode_local_detail": {
        "en": "Downloads the video, can transcribe when captions are unavailable, and uses our learning player.",
        "zh": "下载视频；无字幕时可转写；默认使用我们的学习播放器。",
    },
    "input.playback_mode_online_detail": {
        "en": "Does not download the video. Requires YouTube captions and uses YouTube's embedded playback surface.",
        "zh": "不下载视频；需要 YouTube 已有字幕；播放使用 YouTube 嵌入播放器。",
    },
    "input.native_target_subtitles": {
        "en": "Use the video's original target-language subtitles when available",
        "zh": "有目标语言字幕时，直接使用视频原字幕",
    },
    "input.native_target_subtitles_help": {
        "en": "Preserves the platform subtitle track and skips speech recognition, segmentation, and machine translation. If unavailable, the normal caption/ASR and translation pipeline is used.",
        "zh": "保留平台提供的原始字幕轨道，并跳过语音识别、切句和机器翻译；若没有目标语言字幕，则自动回到常规字幕/语音识别与翻译流程。",
    },
    "input.online_url_only": {
        "en": "Online playback requires a YouTube link, not an uploaded file.",
        "zh": "在线播放模式需要 YouTube 链接，不能使用上传文件。",
    },
    "input.online_youtube_only": {
        "en": "Online playback currently supports standard YouTube links only.",
        "zh": "在线播放目前仅支持有效的 YouTube 视频链接。",
    },
    "input.process_button": {
        "en": "Process Video",
        "zh": "处理视频",
    },
    "input.source_title": {
        "en": "Add a video",
        "zh": "添加视频",
    },
    "input.source_hint": {
        "en": "Paste a supported link or upload a file. Processing continues safely in the background.",
        "zh": "粘贴支持的视频链接，或上传本地文件。处理会在后台安全地继续进行。",
    },
    "input.choose_one_source": {
        "en": "Choose either a link or an uploaded file, not both.",
        "zh": "请仅选择视频链接或上传文件其中一种来源。",
    },
    "input.path_missing": {
        "en": "The local video path does not exist or is not a file.",
        "zh": "本地视频路径不存在，或它不是一个文件。",
    },

    # ---- Pipeline steps ----
    "pipeline.step1": {
        "en": "Step 1/4: Resolving video source...",
        "zh": "步骤 1/4：获取视频源...",
    },
    "pipeline.step1_done": {
        "en": "Step 1/4: Video ready — {filename}",
        "zh": "步骤 1/4：视频就绪 — {filename}",
    },
    "pipeline.step1_saved": {
        "en": "Saved uploaded file: **{filename}**",
        "zh": "已保存上传文件：**{filename}**",
    },
    "pipeline.step1_video": {
        "en": "Video: **{filename}**",
        "zh": "视频：**{filename}**",
    },
    "pipeline.step2_sub": {
        "en": "Trying to pull existing subtitles...",
        "zh": "尝试拉取已有字幕...",
    },
    "pipeline.step2_sub_found": {
        "en": "Found **{n}** subtitles from source — skipping ASR.",
        "zh": "从来源获取到 **{n}** 条字幕 — 跳过语音识别。",
    },
    "pipeline.step2_sub_done": {
        "en": "Subtitles pulled — {n} segments",
        "zh": "字幕已获取 — {n} 段",
    },
    "pipeline.step2_sub_none": {
        "en": "No existing subtitles found, will transcribe.",
        "zh": "未找到已有字幕，将进行语音识别。",
    },
    "pipeline.step2": {
        "en": "Step 2/4: Extracting audio (16 kHz mono WAV)...",
        "zh": "步骤 2/4：提取音频（16kHz 单声道 WAV）...",
    },
    "pipeline.step2_done": {
        "en": "Step 2/4: Audio extracted",
        "zh": "步骤 2/4：音频提取完成",
    },
    "pipeline.step2_size": {
        "en": "Audio extracted: **{size:.0f} KB**",
        "zh": "音频已提取：**{size:.0f} KB**",
    },
    "pipeline.step3": {
        "en": "Step 3/4: Transcribing with Whisper...",
        "zh": "步骤 3/4：Whisper 语音转文字...",
    },
    "pipeline.step3_loading": {
        "en": "Loading faster-whisper model `{model}` (first run downloads the model)...",
        "zh": "正在加载 faster-whisper 模型 `{model}`（首次运行会下载模型）...",
    },
    "pipeline.step3_transcribing": {
        "en": "Transcribing audio (this may take a few minutes)...",
        "zh": "正在转录音频（可能需要几分钟）...",
    },
    "pipeline.step3_segments": {
        "en": "Detected **{n}** subtitle segments",
        "zh": "检测到 **{n}** 个字幕片段",
    },
    "pipeline.step3_done": {
        "en": "Step 3/4: Transcription done — {n} segments",
        "zh": "步骤 3/4：转录完成 — {n} 个片段",
    },
    "pipeline.step3_segmented": {
        "en": "Re-segmented into **{n}** natural sentences (from {raw} raw fragments)",
        "zh": "已重组为 **{n}** 个自然语句（原始 {raw} 个片段）",
    },
    "pipeline.step3_segment": {
        "en": "AI sentence segmentation...",
        "zh": "AI 语句切分中...",
    },
    "pipeline.step4": {
        "en": "Step 4/4: Translating with LLM...",
        "zh": "步骤 4/4：大模型翻译...",
    },
    "pipeline.step4_batch": {
        "en": "Translating batch **{current}/{total}**...",
        "zh": "正在翻译第 **{current}/{total}** 批...",
    },
    "pipeline.step4_done": {
        "en": "Step 4/4: Translation complete",
        "zh": "步骤 4/4：翻译完成",
    },
    "pipeline.step4_complete": {
        "en": "Translation complete — **{n}** entries",
        "zh": "翻译完成 — **{n}** 条",
    },
    "pipeline.failed": {
        "en": "Pipeline failed: {error}",
        "zh": "处理失败：{error}",
    },

    # ---- Results ----
    "results.download_btn": {
        "en": "Download Subtitles (JSON)",
        "zh": "下载字幕 (JSON)",
    },
    "results.info": {
        "en": "Video: `{filename}` | Subtitles: {n} segments | Server: {url}",
        "zh": "视频：`{filename}` | 字幕：{n} 段 | 服务：{url}",
    },
    "results.info_online": {
        "en": "Online YouTube playback | Subtitles: {n} segments | No local video file",
        "zh": "YouTube 在线播放 | 字幕：{n} 段 | 未保存本地视频",
    },
    "results.timing_word": {
        "en": "Timing: word-aligned",
        "zh": "时间轴：词级对齐",
    },
    "results.timing_cue": {
        "en": "Timing: source cue boundaries",
        "zh": "时间轴：来源字幕边界",
    },
    "results.online_source": {
        "en": "YouTube online video",
        "zh": "YouTube 在线视频",
    },
    "results.table_header": {
        "en": "Read full transcript",
        "zh": "阅读完整字幕稿",
    },
    "results.speaker_unknown": {
        "en": "Speaker",
        "zh": "说话人",
    },
    "results.no_results": {
        "en": "No results to display. Try processing a new video.",
        "zh": "暂无结果。请尝试处理一个新的视频。",
    },

    # ---- Sidebar ----
    "sidebar.header": {
        "en": "Settings",
        "zh": "设置",
    },
    "sidebar.collapse": {
        "en": "Collapse sidebar",
        "zh": "收起侧栏",
    },
    "sidebar.show": {
        "en": "Show sidebar",
        "zh": "显示侧栏",
    },
    "sidebar.ui_language": {
        "en": "UI Language",
        "zh": "界面语言",
    },
    "sidebar.section_asr": {
        "en": "ASR Settings",
        "zh": "语音识别设置",
    },
    "sidebar.asr_model": {
        "en": "Whisper Model",
        "zh": "Whisper 模型",
    },
    "sidebar.asr_model_help": {
        "en": "Larger = more accurate but slower. 'base' is a good balance for MVP.",
        "zh": "模型越大越准确但速度越慢。'base' 是一个较好的平衡选择。",
    },
    "sidebar.section_translation": {
        "en": "Translation Settings",
        "zh": "翻译设置",
    },
    "sidebar.target_language": {
        "en": "Target Language",
        "zh": "目标语言",
    },
    "sidebar.model_preset": {
        "en": "Model Preset",
        "zh": "模型预设",
    },
    "sidebar.manage_presets": {
        "en": "Manage Presets",
        "zh": "管理预设",
    },
    "sidebar.preset_name": {
        "en": "Preset Name",
        "zh": "预设名称",
    },
    "sidebar.preset_engine": {
        "en": "Engine Type",
        "zh": "引擎类型",
    },
    "sidebar.preset_model": {
        "en": "Model Name",
        "zh": "模型名称",
    },
    "sidebar.preset_api_base": {
        "en": "API Base URL",
        "zh": "API 地址",
    },
    "sidebar.preset_api_key": {
        "en": "API Key",
        "zh": "API 密钥",
    },
    "sidebar.preset_add": {
        "en": "Add Preset",
        "zh": "添加预设",
    },
    "sidebar.preset_save": {
        "en": "Save Changes",
        "zh": "保存修改",
    },
    "sidebar.preset_delete": {
        "en": "Delete Preset",
        "zh": "删除预设",
    },
    "sidebar.preset_delete_confirm": {
        "en": "Are you sure you want to delete preset '{}'?",
        "zh": "确定要删除预设 '{}' 吗？",
    },
    "sidebar.preset_action_label": {
        "en": "Action",
        "zh": "操作",
    },
    "sidebar.preset_add_btn": {
        "en": "Add",
        "zh": "添加",
    },
    "sidebar.preset_save_btn": {
        "en": "Save",
        "zh": "保存",
    },
    "sidebar.preset_none": {
        "en": "(no presets)",
        "zh": "（无预设）",
    },
    "sidebar.section_network": {
        "en": "Network",
        "zh": "网络",
    },
    "sidebar.proxy": {
        "en": "Proxy (for yt-dlp)",
        "zh": "代理（用于 yt-dlp）",
    },
    "sidebar.proxy_placeholder": {
        "en": "http://127.0.0.1:7897",
        "zh": "http://127.0.0.1:7897",
    },
    "sidebar.proxy_help": {
        "en": "Leave blank to use direct connection. Auto-detected from env HTTPS_PROXY.",
        "zh": "留空使用直连。自动检测环境变量 HTTPS_PROXY。",
    },
    "sidebar.section_actions": {
        "en": "Actions",
        "zh": "操作",
    },
    "sidebar.section_concurrency": {
        "en": "Concurrency",
        "zh": "并发",
    },
    "sidebar.max_workers": {
        "en": "Max Parallel Requests",
        "zh": "最大并发数",
    },
    "sidebar.max_workers_help": {
        "en": "Higher = faster, but may hit rate limits (cloud) or OOM (local Ollama). 6 is a safe default.",
        "zh": "越高越快，但可能触发限流（云端）或显存不足（本地 Ollama）。6 为安全默认值。",
    },
    "engine.ollama": {
        "en": "Ollama (Local)",
        "zh": "Ollama（本地）",
    },
    "engine.openai": {
        "en": "OpenAI (Cloud)",
        "zh": "OpenAI（云端）",
    },
    "engine.hy_mt2_local": {
        "en": "Hy-MT2 (On-device)",
        "zh": "Hy-MT2（本机运行）",
    },
    "settings.local_runtime": {
        "en": "On-device translation runtime",
        "zh": "本机翻译运行环境",
    },
    "settings.local_path": {
        "en": "Local model folder",
        "zh": "本地模型目录",
    },
    "settings.model_ready": {
        "en": "Hy-MT2 is ready. It will load into memory when translation starts.",
        "zh": "Hy-MT2 已就绪，开始翻译时才会载入内存。",
    },
    "settings.model_download": {
        "en": "Download Hy-MT2 (about 4.1 GB)",
        "zh": "下载 Hy-MT2（约 4.1 GB）",
    },
    "settings.model_download_hint": {
        "en": "The official Tencent model is stored locally and never sends subtitle text to a cloud API.",
        "zh": "官方腾讯模型将保存到本机，字幕文本不会发送到云端 API。",
    },
    "settings.model_downloaded": {
        "en": "Hy-MT2 has been downloaded and is ready to use.",
        "zh": "Hy-MT2 下载完成，已可使用。",
    },
    "settings.model_download_failed": {
        "en": "Model download failed: {error}",
        "zh": "模型下载失败：{error}",
    },
    "settings.model_setup": {
        "en": "Set up the project runtime first: ./scripts/setup_local.sh",
        "zh": "请先初始化项目运行环境：./scripts/setup_local.sh",
    },
    "settings.saved": {
        "en": "Settings saved locally.",
        "zh": "设置已保存到本机。",
    },
    "player.no_subtitles": {
        "en": "No subtitles found.",
        "zh": "未找到字幕。",
    },
    "player.pending_translation": {
        "en": "(pending translation)",
        "zh": "（待翻译）",
    },
    "player.speed": {
        "en": "Playback speed",
        "zh": "播放速度",
    },
    "player.bilingual_title": {
        "en": "Show bilingual subtitles",
        "zh": "显示双语字幕",
    },
    "player.bilingual_off_title": {
        "en": "Show translation line",
        "zh": "显示翻译行",
    },
    "player.bilingual_on": {
        "en": "Bilingual",
        "zh": "双语",
    },
    "player.bilingual_off": {
        "en": "Original only",
        "zh": "仅原文",
    },
    "player.bilingual_on_short": {
        "en": "双语",
        "zh": "双语",
    },
    "player.bilingual_off_short": {
        "en": "原文",
        "zh": "原文",
    },
    "player.video_bilingual_title": {
        "en": "Show translation in video subtitles",
        "zh": "显示视频字幕的翻译行",
    },
    "player.video_bilingual_off_title": {
        "en": "Show video subtitle translation",
        "zh": "显示视频字幕翻译",
    },
    "player.video_bilingual_on": {
        "en": "Video bilingual",
        "zh": "视频双语",
    },
    "player.video_bilingual_off": {
        "en": "Video original",
        "zh": "视频原文",
    },
    "player.video_bilingual_on_short": {
        "en": "Dual",
        "zh": "双语",
    },
    "player.video_bilingual_off_short": {
        "en": "Original",
        "zh": "原文",
    },
    "player.video_original_title": {
        "en": "Show source subtitles",
        "zh": "显示原文字幕",
    },
    "player.video_original_off_title": {
        "en": "Show source subtitles",
        "zh": "显示原文字幕",
    },
    "player.video_original_on": {
        "en": "Source subtitles",
        "zh": "原文字幕",
    },
    "player.video_original_off": {
        "en": "Source: Off",
        "zh": "原文：关",
    },
    "player.video_translation_title": {
        "en": "Show translated subtitles",
        "zh": "显示译文字幕",
    },
    "player.video_translation_off_title": {
        "en": "Show translated subtitles",
        "zh": "显示译文字幕",
    },
    "player.video_translation_on": {
        "en": "Translated subtitles",
        "zh": "译文字幕",
    },
    "player.video_translation_off": {
        "en": "Translation: Off",
        "zh": "译文：关",
    },
    "player.video_translation_on_short": {
        "en": "Translation",
        "zh": "译文",
    },
    "player.video_translation_off_short": {
        "en": "Translation off",
        "zh": "译文关",
    },
    "player.subtitle_settings": {
        "en": "Subtitle settings",
        "zh": "字幕设置",
    },
    "player.settings_subtitles": {
        "en": "Subtitles",
        "zh": "字幕",
    },
    "player.word_lookup_setting": {
        "en": "Click word to look up",
        "zh": "点击单词查询",
    },
    "player.subtitle_on_short": {
        "en": "On",
        "zh": "开",
    },
    "player.subtitle_off_short": {
        "en": "Off",
        "zh": "关",
    },
    "player.list_bilingual_title": {
        "en": "Show translation in subtitle list",
        "zh": "显示字幕列表的翻译行",
    },
    "player.list_bilingual_off_title": {
        "en": "Show subtitle-list translation",
        "zh": "显示字幕列表翻译",
    },
    "player.list_bilingual_on": {
        "en": "List bilingual",
        "zh": "列表双语",
    },
    "player.list_bilingual_off": {
        "en": "List original",
        "zh": "列表原文",
    },
    "player.video_subtitles_on": {
        "en": "Video subtitles: On",
        "zh": "视频字幕：开",
    },
    "player.video_subtitles_off": {
        "en": "Video subtitles: Off",
        "zh": "视频字幕：关",
    },
    "player.font_size": {
        "en": "Subtitle size",
        "zh": "字幕字号",
    },
    "player.video_font_size": {
        "en": "Video subtitle size",
        "zh": "视频字幕字号",
    },
    "player.list_font_size": {
        "en": "List text size",
        "zh": "列表字号",
    },
    "player.subtitle_tools": {
        "en": "Search and display options",
        "zh": "搜索与显示设置",
    },
    "player.subtitle_search": {
        "en": "Search subtitles",
        "zh": "搜索字幕",
    },
    "player.subtitle_search_placeholder": {
        "en": "Search source or translation…",
        "zh": "搜索原文或译文…",
    },
    "player.subtitle_search_empty": {
        "en": "No matching subtitles",
        "zh": "没有匹配的字幕",
    },
    "player.subtitle_timing": {
        "en": "Subtitle timing (− early / + late)",
        "zh": "字幕同步（−提前 / +延后）",
    },
    "player.subtitle_timing_short": {
        "en": "Timing",
        "zh": "字幕同步",
    },
    "player.subtitle_timing_controls": {
        "en": "Subtitle timing controls",
        "zh": "字幕同步控制",
    },
    "player.subtitle_timing_hint": {
        "en": "Fine-tune subtitle timing in 0.1-second steps. This setting is saved for this project.",
        "zh": "以 0.1 秒为单位微调字幕时间；设置会保存到当前项目。",
    },
    "player.subtitle_offset_earlier": {
        "en": "Show subtitles 0.1 seconds earlier",
        "zh": "字幕提前 0.1 秒",
    },
    "player.subtitle_offset_later": {
        "en": "Show subtitles 0.1 seconds later",
        "zh": "字幕延后 0.1 秒",
    },
    "player.subtitle_offset_reset": {
        "en": "Reset",
        "zh": "重置",
    },
    "player.subtitle_drag": {
        "en": "Drag subtitle",
        "zh": "拖动字幕",
    },
    "player.subtitle_drag_original": {
        "en": "Move source subtitle",
        "zh": "拖动原文字幕",
    },
    "player.subtitle_drag_translation": {
        "en": "Move translated subtitle",
        "zh": "拖动译文字幕",
    },
    "player.subtitle_drag_hint": {
        "en": "Drag this subtitle layer horizontally or vertically. Arrow keys adjust; Home and End move it to the vertical bounds.",
        "zh": "可上下左右拖动当前字幕层；方向键微调，Home/End 移到上下边界。",
    },
    "player.hover_lookup": {
        "en": "Looking up in dictionary…",
        "zh": "正在查询词典…",
    },
    "player.hover_no_result": {
        "en": "No dictionary entry found.",
        "zh": "未找到词典释义。",
    },
    "player.hover_ai_hint": {
        "en": "Dictionary lookup supports a single English word.",
        "zh": "词典查询仅支持单个英文单词。",
    },
    "player.action_timeout": {
        "en": "Timed out after 2 seconds. Please retry.",
        "zh": "超过 2 秒未完成，请重试。",
    },
    "player.sub_panel_title": {
        "en": "Subtitles",
        "zh": "字幕列表",
    },
    "player.fullscreen": {
        "en": "Immersive fullscreen",
        "zh": "沉浸全屏",
    },
    "player.panel_collapse": {
        "en": "Collapse",
        "zh": "收起",
    },
    "player.panel_restore": {
        "en": "Show subtitles",
        "zh": "显示字幕",
    },
    "player.panel_resize": {
        "en": "Resize subtitle area",
        "zh": "调整字幕区域",
    },
    "player.panel_resize_hint": {
        "en": "Drag to resize. Use arrow keys for precise adjustment; double-click to reset.",
        "zh": "拖动调整占比；方向键精调，双击恢复默认。",
    },
    "player.panel_resize_value": {
        "en": "Subtitle area",
        "zh": "字幕区域",
    },
    "player.follow_on": {
        "en": "Following",
        "zh": "跟随中",
    },
    "player.follow_resume": {
        "en": "Resume follow",
        "zh": "恢复跟随",
    },
    "player.tab_subtitles": {
        "en": "Subtitles",
        "zh": "字幕",
    },
    "player.tab_favorites": {
        "en": "Favorites",
        "zh": "收藏",
    },
    "player.tab_wordbook": {
        "en": "Wordbook",
        "zh": "单词本",
    },
    "player.popup_translation_placeholder": {
        "en": "Translation (optional)...",
        "zh": "翻译（可选）...",
    },
    "player.popup_save": {
        "en": "Translate & save",
        "zh": "翻译并加入单词本",
    },
    "player.word_save": {
        "en": "Add to wordbook",
        "zh": "加入单词本",
    },
    "player.word_remove": {
        "en": "Remove from wordbook",
        "zh": "从单词本移除",
    },
    "player.play_pronunciation": {
        "en": "Play pronunciation",
        "zh": "播放读音",
    },
    "player.phonetic": {
        "en": "Pronunciation",
        "zh": "音标",
    },
    "player.phonetic_uk": {
        "en": "UK",
        "zh": "英",
    },
    "player.phonetic_us": {
        "en": "US",
        "zh": "美",
    },
    "player.wordbook_source": {
        "en": "Saved wordbook entry",
        "zh": "已保存的单词本词条",
    },
    "player.popup_cancel": {
        "en": "Cancel",
        "zh": "取消",
    },
    "player.popup_favorite": {
        "en": "Favorite sentence",
        "zh": "收藏整句",
    },
    "player.popup_unfavorite": {
        "en": "Remove favorite",
        "zh": "取消收藏",
    },
    "player.popup_translating": {
        "en": "Translating in context...",
        "zh": "正在进行语境翻译...",
    },
    "player.popup_saving": {
        "en": "Saving...",
        "zh": "保存中...",
    },
    "player.ai_translate": {
        "en": "Context translation",
        "zh": "语境翻译",
    },
    "player.free_translation_label": {
        "en": "Translation:",
        "zh": "翻译：",
    },
    "player.free_translation_failed": {
        "en": "No dictionary entry or free translation is available right now. Please retry.",
        "zh": "暂未获得词典释义或免费翻译，请稍后重试。",
    },
    "player.fav_empty": {
        "en": "No favorites yet. Click the star on a subtitle to add.",
        "zh": "暂无收藏。点击字幕旁的星标即可收藏。",
    },
    "player.word_empty": {
        "en": "Wordbook is empty. Select text in a subtitle to add a word.",
        "zh": "单词本为空。在字幕中选中文字即可添加单词。",
    },
    "collections.title": {
        "en": "Collections",
        "zh": "收藏合集",
    },
    "collections.wordbook": {
        "en": "Wordbook",
        "zh": "单词本",
    },
    "collections.favorites": {
        "en": "Favorites",
        "zh": "收藏句子",
    },
    "collections.translating_word": {
        "en": "Translating “{word}”...",
        "zh": "正在翻译“{word}”...",
    },
    "collections.word_saved": {
        "en": "“{word}” was added to the wordbook.",
        "zh": "“{word}”已加入单词本。",
    },
    "collections.word_updated": {
        "en": "“{word}” was updated in the wordbook.",
        "zh": "已更新单词本中的“{word}”。",
    },
    "collections.word_removed": {
        "en": "Word removed from the wordbook.",
        "zh": "已从单词本移除。",
    },
    "collections.favorite_saved": {
        "en": "Sentence added to favorites.",
        "zh": "整句已加入收藏。",
    },
    "collections.favorite_removed": {
        "en": "Sentence removed from favorites.",
        "zh": "已取消收藏。",
    },
    "collections.action_invalid": {
        "en": "This collection action is no longer valid. Reload the project and try again.",
        "zh": "此收藏操作已失效，请重新打开项目后再试。",
    },
    "collections.word_translation_empty": {
        "en": "The model returned an empty translation. Nothing was saved.",
        "zh": "模型返回了空翻译，本次未保存。",
    },
    "collections.word_lookup_failed": {
        "en": "AI contextual translation failed: {error}",
        "zh": "AI 语境翻译失败：{error}",
    },
    "collections.action_failed": {
        "en": "Collection action failed: {error}",
        "zh": "收藏操作失败：{error}",
    },
    "collections.action_timeout": {
        "en": "This action exceeded 2 seconds. Please retry.",
        "zh": "操作超过 2 秒未完成，请重试。",
    },
    "collections.wordbook_empty": {
        "en": "No words saved yet.",
        "zh": "还没有收藏的单词。",
    },
    "collections.fav_empty": {
        "en": "No favorites yet.",
        "zh": "还没有收藏的句子。",
    },
    "collections.search": {
        "en": "Search",
        "zh": "搜索",
    },
    "collections.search_wordbook": {
        "en": "Search words...",
        "zh": "搜索单词...",
    },
    "collections.search_favorites": {
        "en": "Search sentences...",
        "zh": "搜索句子...",
    },
    "collections.count": {
        "en": "{n} entries",
        "zh": "共 {n} 条",
    },
    "collections.open": {
        "en": "Open in player",
        "zh": "在播放器中打开",
    },
    "sidebar.clear_reset": {
        "en": "Clear & Reset",
        "zh": "清除并重置",
    },
    "sidebar.work_files": {
        "en": "Work files → `./work/`",
        "zh": "工作文件 → `./work/`",
    },
    "sidebar.api_key_note": {
        "en": "API keys are stored locally. Do not share this file.",
        "zh": "API 密钥存储在本地。请勿分享此文件。",
    },

    # ---- Project panel ----
    "projects.header": {
        "en": "Project Library",
        "zh": "项目库",
    },
    "projects.search": {
        "en": "Search...",
        "zh": "搜索...",
    },
    "projects.search_placeholder": {
        "en": "Search projects",
        "zh": "搜索项目",
    },
    "projects.open_project": {
        "en": "Open project: {title}",
        "zh": "打开项目：{title}",
    },
    "projects.created_at": {
        "en": "Created",
        "zh": "创建于",
    },
    "projects.progress_in_card": {
        "en": "This task is running. Its live progress is shown in its project card in the library.",
        "zh": "任务正在处理中，实时进度显示在左侧项目库对应的项目卡片内。",
    },
    "player.init_failed": {
        "en": "The learning player could not initialize.",
        "zh": "学习播放器初始化失败。",
    },
    "projects.empty": {
        "en": "No projects yet.\n\nProcess a video to get started.",
        "zh": "暂无项目。\n\n处理一个视频以开始使用。",
    },
    "projects.custom_title_placeholder": {
        "en": "Custom title...",
        "zh": "自定义标题...",
    },
    "projects.delete": {
        "en": "Delete",
        "zh": "删除",
    },
    "projects.delete_confirm": {
        "en": "Permanently delete this project?",
        "zh": "永久删除此项目？",
    },
    "projects.delete_note": {
        "en": "Its subtitles, app-managed video/audio, saved words, and favorites will be deleted. A local original outside the app work folder is kept.",
        "zh": "会删除字幕、应用管理的视频/音频、生词和收藏；应用工作目录外的本地原文件会保留。",
    },
    "projects.delete_all": {
        "en": "Delete all learning data",
        "zh": "删除全部学习数据",
    },
    "projects.delete_all_confirm": {
        "en": "This permanently deletes every project, downloaded/uploaded media, subtitles, wordbook entries, and favorites.",
        "zh": "这会永久删除全部项目、已下载/上传的视频、字幕、生词本和收藏。",
    },
    "projects.delete_all_note": {
        "en": "Model downloads, presets, and app settings are kept.",
        "zh": "不会删除模型文件、预设和应用设置。",
    },
    "projects.delete_all_ack": {
        "en": "I understand this cannot be undone.",
        "zh": "我了解此操作无法撤销。",
    },
    "projects.delete_all_busy": {
        "en": "Wait for running imports to finish before deleting all learning data.",
        "zh": "请等待正在导入的任务完成后，再删除全部学习数据。",
    },
    "projects.delete_all_done": {
        "en": "Deleted {n} projects and all learning data.",
        "zh": "已删除 {n} 个项目及全部学习数据。",
    },
    "projects.cancel": {
        "en": "Cancel",
        "zh": "取消",
    },
    "projects.status_completed": {
        "en": "completed",
        "zh": "已完成",
    },
    "projects.status_legacy": {
        "en": "legacy",
        "zh": "旧版",
    },
    "projects.status_processing": {
        "en": "processing",
        "zh": "处理中",
    },
    "projects.load_failed": {
        "en": "Subtitles file not found. The project may need to be re-processed.",
        "zh": "字幕文件未找到。该项目可能需要重新处理。",
    },
    "projects.reprocess_help": {
        "en": "Re-run ASR + translation on this project",
        "zh": "对该项目重新进行语音识别和翻译",
    },
    "projects.file_missing": {
        "en": "Video file not found on disk. It may have been moved or deleted.",
        "zh": "视频文件在磁盘上未找到。可能已被移动或删除。",
    },
    "projects.player_start_failed": {
        "en": "The local video player could not start: {error}",
        "zh": "本地视频播放器无法启动：{error}",
    },
    "projects.online_source_missing": {
        "en": "The online video identifier is missing. Please create this task again from its YouTube link.",
        "zh": "在线播放标识缺失，请使用原 YouTube 链接重新创建任务。",
    },
    "settings.advanced": {
        "en": "Advanced model and network settings",
        "zh": "高级模型与网络设置",
    },
    "settings.local_concurrency": {
        "en": "The on-device model manages inference serially to keep memory use stable.",
        "zh": "本机模型会自动串行推理，以保持内存占用稳定。",
    },
    "input.source_required": {
        "en": "Enter a video link or upload a local video file.",
        "zh": "请输入视频链接，或上传一个本地视频文件。",
    },
    "projects.reprocess": {
        "en": "Re-run ASR + translation from source video",
        "zh": "从源视频重新进行语音识别和翻译",
    },
    "projects.retranslate": {
        "en": "Re-translate with current model (skip ASR)",
        "zh": "使用当前模型重新翻译（跳过语音识别）",
    },
    "projects.action_retranslate": {
        "en": "Re-translate",
        "zh": "重新翻译",
    },
    "projects.action_reprocess": {
        "en": "Re-process",
        "zh": "重新处理",
    },
    "projects.action_rename": {
        "en": "Rename",
        "zh": "重命名",
    },
    "projects.manage": {
        "en": "Manage project",
        "zh": "项目管理",
    },
    "projects.manage_processing": {
        "en": "Subtitle processing",
        "zh": "字幕处理",
    },
    "projects.manage_details": {
        "en": "Project details",
        "zh": "项目信息",
    },
    "projects.native_subtitles_used": {
        "en": "Original {language} subtitles · no ASR or machine translation",
        "zh": "使用视频原生 {language} 字幕 · 未进行语音识别或机器翻译",
    },
    "projects.new_task": {
        "en": "New Task",
        "zh": "新建任务",
    },
    "projects.welcome_hint": {
        "en": "Click the button above to start a new translation, or pick a project from the left.",
        "zh": "点击上方按钮新建翻译任务，或从左侧选择一个项目。",
    },
    "projects.process_now": {
        "en": "Process Now",
        "zh": "立即处理",
    },
    "projects.no_subtitles_hint": {
        "en": "This project has no subtitles yet. Click to run ASR + translation.",
        "zh": "该项目还没有字幕。点击按钮进行语音识别和翻译。",
    },
    "projects.pending_title": {
        "en": "Importing · {source}",
        "zh": "正在导入 · {source}",
    },
    "projects.task_failed": {
        "en": "This project failed to process: {error}",
        "zh": "此项目处理失败：{error}",
    },
    "projects.retry": {
        "en": "Retry processing",
        "zh": "重新处理",
    },
    "projects.task_interrupted": {
        "en": "Processing was interrupted before this project finished. You can safely retry it.",
        "zh": "此项目的处理在完成前中断，可以安全地重新处理。",
    },
    "sidebar.model_override": {
        "en": "Model Name (override preset)",
        "zh": "模型名称（覆盖预设）",
    },
    "sidebar.model_override_help": {
        "en": "Type any model name to override the selected preset. Leave empty to use the preset default. The app auto-detects MT models (containing 'mt'/'translator') and uses line-by-line mode.",
        "zh": "输入任意模型名以覆盖预设。留空则使用预设默认值。应用会自动识别翻译模型（含 'mt'/'translator'）并使用逐行翻译模式。",
    },
    "sidebar.model_reset": {
        "en": "Reset",
        "zh": "重置",
    },
    "sidebar.current_model": {
        "en": "Currently calling:",
        "zh": "当前调用：",
    },
    "sidebar.preset_use": {
        "en": "Use",
        "zh": "使用",
    },
    "sidebar.preset_using": {
        "en": "🟢 Using",
        "zh": "🟢 使用中",
    },
    "sidebar.preset_hide": {
        "en": "Close",
        "zh": "收起",
    },
    "pipeline.retranslate_no_raw": {
        "en": "Raw subtitle data not found on disk. Please re-process the video.",
        "zh": "未找到原始字幕数据。请重新处理视频。",
    },
    "pipeline.model_not_ready": {
        "en": "The selected translation model is not ready: {error}",
        "zh": "所选翻译模型尚未就绪：{error}",
    },
    "pipeline.invalid_model_config": {
        "en": "The selected preset needs a model name and API address before processing can start.",
        "zh": "所选预设需要填写模型名称和 API 地址后才能开始处理。",
    },
    "pipeline.openai_key_missing": {
        "en": "The OpenAI preset has no API key. Add one in Settings, set OPENAI_API_KEY, or switch to the local Hy-MT2 preset.",
        "zh": "OpenAI 预设没有 API Key。请在设置中填写、设置 OPENAI_API_KEY，或切换到本地 Hy-MT2 预设。",
    },
    "pipeline.running": {
        "en": "Processing in background",
        "zh": "后台处理中",
    },
    "pipeline.task_started": {
        "en": "Task started. You can follow its progress here.",
        "zh": "任务已开始，可在此查看处理进度。",
    },
    "pipeline.batch_progress": {
        "en": "Batch {current}/{total}",
        "zh": "第 {current}/{total} 批",
    },
    "pipeline.step_progress": {
        "en": "Step {step}/4 — {label}",
        "zh": "步骤 {step}/4 — {label}",
    },
    "pipeline.status.starting": {
        "en": "Preparing task",
        "zh": "准备任务",
    },
    "pipeline.status.source": {
        "en": "Preparing video source",
        "zh": "准备视频来源",
    },
    "pipeline.status.subtitles": {
        "en": "Looking for existing subtitles",
        "zh": "查找已有字幕",
    },
    "pipeline.status.audio": {
        "en": "Extracting audio",
        "zh": "提取音频",
    },
    "pipeline.status.transcribe": {
        "en": "Transcribing speech",
        "zh": "语音转录",
    },
    "pipeline.status.segment": {
        "en": "Preparing learning sentences",
        "zh": "整理学习语句",
    },
    "pipeline.status.translate": {
        "en": "Translating subtitles",
        "zh": "翻译字幕",
    },
    "pipeline.status.complete": {
        "en": "Completed",
        "zh": "已完成",
    },
    "pipeline.status.failed": {
        "en": "Failed",
        "zh": "失败",
    },
    "pipeline.failed_label": {
        "en": "Failed",
        "zh": "失败",
    },
}

# Language flag emoji for target language codes
LANG_FLAGS: dict[str, str] = {
    "zh": "🇨🇳", "ja": "🇯🇵", "ko": "🇰🇷", "fr": "🇫🇷",
    "de": "🇩🇪", "es": "🇪🇸", "pt": "🇵🇹", "ru": "🇷🇺",
    "ar": "🇸🇦", "th": "🇹🇭", "vi": "🇻🇳", "en": "🇺🇸",
}


def t(key: str, lang: str = None, **fmt) -> str:
    """Return the translation for *key* in the current UI language.

    If *lang* is None, reads from st.session_state["ui_lang"], defaulting to "en".
    Extra kwargs are used for str.format() interpolation.
    """
    if lang is None:
        lang = st.session_state.get("ui_lang", "en")
    text = STRINGS.get(key, {}).get(lang, key)
    if fmt:
        text = text.format(**fmt)
    return text
