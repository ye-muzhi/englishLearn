# EnglishLearn

[![Tests](https://github.com/ye-muzhi/englishLearn/actions/workflows/tests.yml/badge.svg)](https://github.com/ye-muzhi/englishLearn/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Platforms](https://img.shields.io/badge/平台-macOS%20%7C%20Windows-60C?logo=windows)

**本地优先的视频语言学习应用**：导入视频，自动生成或提取字幕，AI 翻译成双语字幕，在自研沉浸式播放器中边看边学 —— 随时暂停查词、收藏生词、用语音或文字记录想法，由本地转写 + AI 整理为可回看原视频位置的学习笔记。

## ✨ 核心特性

- **字幕获取双通道** —— 优先提取视频自带字幕；无字幕时使用 faster-whisper（本地 ASR，含词级时间戳）自动转写。
- **时间轴感知断句** —— 依据时间戳智能切分句级字幕，保持双语逐句对齐，另可选 LLM 辅助断句。
- **多引擎翻译** —— OpenAI 兼容接口、Ollama，以及开箱即用的本地模型 Hy-MT2-1.8B（约 4.1 GB，按需下载），翻译全程批量执行。
- **沉浸式双语播放器** —— YouTube 风格深色主题，字幕按时间轴高亮跟随，支持播放器内直接查词、加生词本、收藏句子、记笔记。
- **学习笔记闭环** —— 播放中一键暂停，录音或文字记录想法；本地转写后由 AI 按语境整理，保存时自动关联原视频时间点。
- **生词本 · 收藏 · 笔记** —— 三类学习集本地持久化，支持导出复习。
- **可选本地词典** —— 索引式离线词典，查词不联网。
- **中英双语界面** —— 全量 i18n，界面语言一键切换。
- **跨平台桌面体验** —— 提供 macOS / Windows 双击启动器，以及 Tauri v2 原生桌面壳。

## 🚀 快速开始

### 方式一：双击启动（推荐普通用户）

- **macOS**：双击 `EnglishLearn.command`（首次被 Gatekeeper 拦截时，在 Finder 中右键选择「打开」）。
- **Windows**：双击 `EnglishLearn-Windows.bat`。

首次启动会自动准备独立的 Python 3.11 运行环境并安装依赖（含 FFmpeg），可能需要几分钟；之后再次双击会直接启动并打开 `http://localhost:8501`。**无需预装 Python、uv 或 FFmpeg，也不需要管理员权限**；仅首次安装与使用媒体处理功能时需要联网。

### 方式二：命令行

```bash
./scripts/setup_local.sh   # 首次安装（Windows 用 scripts/run_app_windows.ps1 同理）
./scripts/run_app.sh       # 启动应用
```

### 用户数据在哪里？

默认保存在**系统用户数据目录**而非源码目录，升级或替换应用文件不会影响你的数据：

| 系统 | 路径 |
| --- | --- |
| macOS | `~/Library/Application Support/EnglishLearn` |
| Windows | `%LOCALAPPDATA%\EnglishLearn` |
| Linux | `${XDG_DATA_HOME:-~/.local/share}/EnglishLearn` |

开发者可通过环境变量 `ENGLISHLEARN_WORK_DIR` 自定义位置。

### 翻译引擎配置

进入「设置」页按需选择：配置任意 OpenAI 兼容接口、本地 Ollama，或直接下载本地模型 Hy-MT2-1.8B（约 4.1 GB，不随发行包分发，下载后完全离线翻译）。

## 📦 桌面版（Tauri）

`desktop/` 包含 Tauri v2 桌面壳：使用系统 WebView 展示本地 EnglishLearn，Python 后端由 PyInstaller 构建为平台 sidecar，应用关闭时同步结束本地服务。

本地构建需要 Rust、Node.js 22 和已准备好的 `.venv`：

```bash
uv pip install --python .venv/bin/python pyinstaller
.venv/bin/python scripts/build_desktop_sidecar.py
cd desktop && npm install && npm run build
```

仓库内置 `.github/workflows/desktop-release.yml`，在原生 macOS / Windows 环境构建 DMG 与安装包。正式对外发布前需在 GitHub 仓库密钥中配置 Apple 公证、Windows 代码签名与 Tauri 更新签名凭据；未配置时构建出的包仅适合内部测试。

## 🔒 隐私与数据

- **本地优先**：转写、断句、播放、笔记整理任务都在本机执行；仅在你主动下载视频/模型或调用自己配置的云端翻译接口时联网。
- **数据不出本机**：项目、字幕、生词本、收藏、笔记全部保存在本地数据目录，不进入 Git 仓库，不做任何埋点上报。
- **密钥安全**：API 密钥仅保存在本机数据目录的配置文件中，`.gitignore` 已排除所有敏感文件。

## 🧩 项目结构

```text
app.py                       Streamlit 入口与页面组合
englishlearn/
  paths.py                   项目根目录与用户数据路径
  i18n.py                    中英文 UI 文案
  media/                     视频下载、字幕解析、本地媒体服务、离线词典索引
  processing/                ASR、时间轴断句、后台处理流水线
  notes/                     语音转写与 AI 笔记整理任务
  storage/                   项目库、学习集与字幕编辑持久化
  translation/               OpenAI / Ollama / Hy-MT2 本地翻译引擎
desktop/                     Tauri v2 桌面壳（Rust + Web 前端）
docs/                        功能规格与调研文档
scripts/                     安装、启动与构建脚本
tests/                       自动化回归测试
work/                        运行期用户数据目录（不入库，仅保留占位）
EnglishLearn.command         macOS 双击启动入口
EnglishLearn-Windows.bat     Windows 双击启动入口
```

> `work/` 是用户数据目录，不应作为源码清理或重构的一部分删除；其中内容（项目、媒体、模型、密钥配置）均被 Git 忽略。

## 🛠️ 开发

```bash
./scripts/setup_local.sh                          # 创建 .venv 并安装依赖
./scripts/run_app.sh                              # 启动开发服务
.venv/bin/python -B -m pytest -q -p no:cacheprovider   # 运行全部测试
.venv/bin/python scripts/build_release.py --version 1.1.0   # 构建不含个人数据的分享包
```

架构与开发约定详见 [AGENTS.md](AGENTS.md)，功能规格详见 [docs/FEATURES.md](docs/FEATURES.md)。

## 📄 许可

本项目暂未选择开源许可证（未设置 LICENSE 期间，默认保留所有权利）。
