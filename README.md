# EnglishLearn

本地视频语言学习应用：导入视频、生成/提取字幕、翻译并在自有播放器中学习。

## 最简单的启动方式

- macOS：双击 `EnglishLearn.command`。
- Windows：双击 `EnglishLearn-Windows.bat`。

首次启动会自动准备独立的 Python 3.11 环境并安装依赖，可能需要几分钟；之后双击会直接打开应用。启动器会复用已经运行的服务，并打开 `http://localhost:8501`。无需预装 Python、uv 或 FFmpeg，也不需要管理员权限；首次安装和首次使用媒体处理功能需要联网。

用户数据默认保存在系统用户目录，而不是源码目录：macOS 为 `~/Library/Application Support/EnglishLearn`，Windows 为 `%LOCALAPPDATA%\EnglishLearn`。因此升级或替换应用文件不会覆盖项目、字幕和模型。开发者可通过 `ENGLISHLEARN_WORK_DIR` 指定其他位置。

Hy-MT2 约 4.1 GB，不随发行包分发。进入「设置」后可按需下载；也可以配置 OpenAI 兼容接口或 Ollama。

macOS 如果首次打开被系统拦截，请在 Finder 中右键 `EnglishLearn.command`，选择「打开」。Windows 如果公司策略禁用了 PowerShell 脚本，需要让管理员允许当前用户运行本地脚本。

终端用户仍可运行：

```bash
./scripts/setup_local.sh       # 首次安装
./scripts/run_app.sh           # 启动
```

## 分享与发行

生成不含个人数据、视频、字幕、词库、密钥、虚拟环境和模型的分享包：

```bash
.venv/bin/python scripts/build_release.py --version 1.1.0
```

## 正式桌面安装包

`desktop/` 包含 Tauri v2 桌面壳。它使用系统 WebView 展示本地
EnglishLearn，不会启动或接管 Chrome。Python 后端由 PyInstaller 构建为
平台 sidecar，应用关闭时会同步结束本地服务。

本地构建需要 Rust、Node.js 22 和已经准备好的 `.venv`：

```bash
uv pip install --python .venv/bin/python pyinstaller
.venv/bin/python scripts/build_desktop_sidecar.py
cd desktop
npm install
npm run build
```

仓库同时提供 `.github/workflows/desktop-release.yml`，会在原生 macOS 与
Windows 环境构建 DMG/安装包。正式对外发布前，应在 GitHub 仓库密钥中
配置 Apple 公证、Windows 代码签名和 Tauri 更新签名所需凭据；未配置时
生成的包只适合内部测试。

源码分享包位于 `dist/EnglishLearn-1.1.0.zip`。Windows 的运行环境必须在 Windows 上首次安装，macOS 同理；不要跨系统复制已经生成的 `.venv/`。

## 项目结构

```text
app.py                       Streamlit 入口与页面组合
englishlearn/
  paths.py                   项目根目录和用户数据路径
  i18n.py                    中英文 UI 文案
  media/                     下载、字幕解析、本地媒体服务
  processing/                ASR、断句、后台处理流水线
  storage/                   项目库、收藏与单词本持久化
  translation/               OpenAI/Ollama/Hy-MT2 翻译引擎
scripts/                     安装与命令行启动脚本
tests/                       自动化回归测试
work/                        用户项目、媒体、字幕与本地模型
EnglishLearn.command         macOS 双击启动入口
EnglishLearn-Windows.bat     Windows 双击启动入口
```

`work/` 是用户数据目录，不应作为源码清理或重构的一部分删除。

## 版本保护

源码使用 Git 管理，`work/`、`.venv/`、`.tools/`、密钥文件与构建产物均被忽略。开始较大改动前建议创建分支，稳定节点创建提交或标签；用户数据仍需单独备份，因为它不会进入 Git。

## 测试

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider
```
