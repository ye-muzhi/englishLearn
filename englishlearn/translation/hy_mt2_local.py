"""On-device Tencent Hy-MT2-1.8B translation runtime.

The module is intentionally lazy: importing the Streamlit application does not
download or load a 4 GB model.  The model is downloaded into ``work/models``
only when requested from Settings or the setup command.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

from ..paths import work_dir


MODEL_ID = "tencent/Hy-MT2-1.8B"
DEFAULT_MODEL_DIR = work_dir() / "models" / "Hy-MT2-1.8B"
_RUNTIME: Optional[dict] = None
_LOAD_LOCK = threading.Lock()
_GENERATE_LOCK = threading.Lock()
_CACHE_LOCK = threading.Lock()
_TRANSLATION_CACHE: OrderedDict[tuple[str, str, str, int], str] = OrderedDict()
_CACHE_LIMIT = 512

# Hy-MT2's README recommends these values for the 1.8B/7B checkpoints.  Keep
# them in one place so a future backend change cannot silently drift from the
# upstream inference contract.
HY_MT2_TEMPERATURE = 0.7
HY_MT2_TOP_P = 0.6
HY_MT2_TOP_K = 20
HY_MT2_REPETITION_PENALTY = 1.05
HY_MT2_MAX_TOKENS = 4096

_CHINESE_LANGUAGE_NAMES = {
    "Chinese": "中文", "zh-Hant": "繁体中文", "Traditional Chinese": "繁体中文",
    "English": "英语", "French": "法语", "Portuguese": "葡萄牙语",
    "Spanish": "西班牙语", "Japanese": "日语", "Turkish": "土耳其语",
    "Russian": "俄语", "Arabic": "阿拉伯语", "Korean": "韩语", "Thai": "泰语",
    "Italian": "意大利语", "German": "德语", "Vietnamese": "越南语",
    "Malay": "马来语", "Indonesian": "印度尼西亚语", "Filipino": "菲律宾语",
    "Hindi": "印地语", "Polish": "波兰语", "Czech": "捷克语", "Dutch": "荷兰语",
    "Khmer": "高棉语", "Burmese": "缅甸语", "Persian": "波斯语",
    "Gujarati": "古吉拉特语", "Urdu": "乌尔都语", "Telugu": "泰卢固语",
    "Marathi": "马拉地语", "Hebrew": "希伯来语", "Bengali": "孟加拉语",
    "Tamil": "泰米尔语", "Ukrainian": "乌克兰语", "Tibetan": "藏语",
    "Kazakh": "哈萨克语", "Mongolian": "蒙古语", "Uyghur": "维吾尔语",
    "Cantonese": "粤语",
}


def default_model_dir() -> str:
    return os.environ.get("HY_MT2_MODEL_DIR", str(DEFAULT_MODEL_DIR))


def _version_at_least(version: str, minimum: tuple[int, int]) -> bool:
    parts = [int(p) for p in re.findall(r"\d+", version)[:2]]
    return tuple((parts + [0, 0])[:2]) >= minimum


def model_status(model_dir: Optional[str] = None) -> dict:
    """Return installation information without importing heavyweight libraries."""
    path = Path(model_dir or default_model_dir()).expanduser()
    status = {
        "model_id": MODEL_ID,
        "path": str(path),
        "downloaded": (path / "model.safetensors").exists(),
        "dependencies_ready": False,
        "transformers_version": None,
        "torch_ready": False,
        "message": "",
    }
    try:
        import transformers  # noqa: PLC0415
        status["transformers_version"] = transformers.__version__
        status["dependencies_ready"] = _version_at_least(transformers.__version__, (5, 6))
    except ImportError:
        status["message"] = "Transformers is not installed. Run ./scripts/setup_local.sh first."
        return status

    try:
        import torch  # noqa: PLC0415,F401
        status["torch_ready"] = True
    except ImportError:
        status["message"] = "PyTorch is not installed. Run ./scripts/setup_local.sh first."
        return status

    if not status["dependencies_ready"]:
        status["message"] = "Hy-MT2 requires transformers 5.6 or newer. Run ./scripts/setup_local.sh first."
    elif not status["downloaded"]:
        status["message"] = "The model has not been downloaded yet."
    else:
        status["message"] = "Ready for on-device translation. The model loads on first use."
    return status


def download_model(model_dir: Optional[str] = None) -> str:
    """Download the official model into the project-local model directory."""
    # The standard HTTPS downloader is more reliable than the optional Xet
    # transport on managed desktop networks, and it resumes partial downloads.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is unavailable. Run ./scripts/setup_local.sh first.") from exc

    target = Path(model_dir or default_model_dir()).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(target),
        ignore_patterns=["train/*", "*.md", "imgs/*"],
    )


def _target_name(target_lang: str) -> str:
    """Return the full native language name expected by the Chinese prompt."""
    value = str(target_lang or "").strip()
    # The UI passes values such as ``中文 (Chinese)`` and older projects may
    # pass a bare language code.  Resolve both forms to the upstream full name.
    code_names = {
        "zh": "中文", "zh-hant": "繁体中文", "en": "英语", "fr": "法语",
        "pt": "葡萄牙语", "es": "西班牙语", "ja": "日语", "tr": "土耳其语",
        "ru": "俄语", "ar": "阿拉伯语", "ko": "韩语", "th": "泰语",
        "it": "意大利语", "de": "德语", "vi": "越南语", "ms": "马来语",
        "id": "印度尼西亚语", "tl": "菲律宾语", "hi": "印地语", "pl": "波兰语",
        "cs": "捷克语", "nl": "荷兰语", "km": "高棉语", "my": "缅甸语",
        "fa": "波斯语", "gu": "古吉拉特语", "ur": "乌尔都语", "te": "泰卢固语",
        "mr": "马拉地语", "he": "希伯来语", "bn": "孟加拉语", "ta": "泰米尔语",
        "uk": "乌克兰语", "bo": "藏语", "kk": "哈萨克语", "mn": "蒙古语",
        "ug": "维吾尔语", "yue": "粤语",
    }
    if value.lower() in code_names:
        return code_names[value.lower()]
    if "traditional chinese" in value.lower():
        return "繁体中文"
    for needle, chinese_name in _CHINESE_LANGUAGE_NAMES.items():
        if needle.lower() in value.lower():
            return chinese_name
    # Preserve a full name supplied by a caller instead of inventing a short
    # alias; this is important for newly supported Hy-MT2 languages.
    return value


def _output_budget(text: str, configured: int) -> int:
    """Choose a small safe budget for subtitle-sized inputs.

    The official ceiling remains 4096, but allocating that many tokens for a
    two-word cue makes every generation slower.  A character-based upper bound
    keeps short cues fast while leaving enough room for scripts with a larger
    token/character ratio.
    """
    configured = max(32, int(configured or HY_MT2_MAX_TOKENS))
    heuristic = max(64, len(text.strip()) * 4 + 32)
    return min(HY_MT2_MAX_TOKENS, configured, heuristic)


def _cache_get(key: tuple[str, str, str, int]) -> Optional[str]:
    with _CACHE_LOCK:
        value = _TRANSLATION_CACHE.get(key)
        if value is not None:
            _TRANSLATION_CACHE.move_to_end(key)
        return value


def _cache_put(key: tuple[str, str, str, int], value: str) -> None:
    with _CACHE_LOCK:
        _TRANSLATION_CACHE[key] = value
        _TRANSLATION_CACHE.move_to_end(key)
        while len(_TRANSLATION_CACHE) > _CACHE_LIMIT:
            _TRANSLATION_CACHE.popitem(last=False)


def _load_runtime(model_dir: Optional[str] = None) -> dict:
    global _RUNTIME
    requested_path = str(Path(model_dir or default_model_dir()).expanduser())
    if _RUNTIME and _RUNTIME["path"] == requested_path:
        return _RUNTIME

    with _LOAD_LOCK:
        if _RUNTIME and _RUNTIME["path"] == requested_path:
            return _RUNTIME

        status = model_status(requested_path)
        if not status["dependencies_ready"] or not status["downloaded"]:
            raise RuntimeError(status["message"])

        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        # Hy-MT2 ships extra RoPE tuning fields. Transformers 5 currently
        # ignores those fields after loading the compatible dynamic RoPE
        # implementation, but emits a misleading warning for each model load.
        # Keep genuine exceptions visible while suppressing only that logger.
        logging.getLogger("transformers.modeling_rope_utils").setLevel(logging.ERROR)

        if torch.cuda.is_available():
            device, dtype = "cuda", torch.bfloat16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16
        else:
            device, dtype = "cpu", torch.bfloat16

        tokenizer = AutoTokenizer.from_pretrained(requested_path, trust_remote_code=True)
        kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        try:
            model = AutoModelForCausalLM.from_pretrained(requested_path, dtype=dtype, **kwargs)
        except TypeError:  # Compatibility with the older torch_dtype argument.
            model = AutoModelForCausalLM.from_pretrained(requested_path, torch_dtype=dtype, **kwargs)
        model.to(device)
        model.eval()
        _RUNTIME = {"path": requested_path, "device": device, "torch": torch, "model": model, "tokenizer": tokenizer}
        return _RUNTIME


def _generate(text: str, target_lang: str, model_dir: Optional[str], max_tokens: int) -> str:
    runtime = _load_runtime(model_dir)
    tokenizer, model, torch = runtime["tokenizer"], runtime["model"], runtime["torch"]
    target = _target_name(target_lang)
    # This is the official Hy-MT2 default Chinese instruction.  Hy-MT2 has no
    # default system prompt, so the complete task instruction belongs in the
    # user message and is wrapped by the tokenizer's chat template.
    prompt = f"将以下文本翻译为 {target}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
    messages = [{"role": "user", "content": prompt}]

    with _GENERATE_LOCK, torch.inference_mode():
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        ).to(runtime["device"])
        generation_kwargs = {
            "max_new_tokens": _output_budget(text, max_tokens),
            "temperature": HY_MT2_TEMPERATURE,
            "top_p": HY_MT2_TOP_P,
            "top_k": HY_MT2_TOP_K,
            "repetition_penalty": HY_MT2_REPETITION_PENALTY,
            "do_sample": True,
            "use_cache": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        # Transformers 5 returns a BatchEncoding here; older releases return a
        # tensor. Supporting both keeps this local backend upgrade-safe.
        if hasattr(inputs, "get") and inputs.get("input_ids") is not None:
            input_ids = inputs["input_ids"]
            outputs = model.generate(**inputs, **generation_kwargs)
        else:
            input_ids = inputs
            outputs = model.generate(input_ids=input_ids, **generation_kwargs)
    return tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def translate_text(text: str, target_lang: str, model_dir: Optional[str] = None, max_tokens: int = 1024) -> str:
    normalized = text.strip()
    if not normalized:
        return ""
    key = (
        str(Path(model_dir or default_model_dir()).expanduser()),
        str(target_lang),
        normalized,
        int(max_tokens or HY_MT2_MAX_TOKENS),
    )
    cached = _cache_get(key)
    if cached is not None:
        return cached
    result = _generate(normalized, target_lang, model_dir, max_tokens)
    _cache_put(key, result)
    return result


def translate_subtitles(
    subtitles: list[dict],
    target_lang: str,
    model_dir: Optional[str] = None,
    max_tokens: int = 1024,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[dict]:
    """Translate subtitles predictably, one display sentence at a time.

    Hy-MT2 is very strong at direct translation. Processing individual display
    sentences prevents a generative model from changing list delimiters and
    keeps subtitle timing aligned. The shared generation lock protects the
    model when two Streamlit tasks are started at the same time.
    """
    translated = [dict(item) for item in subtitles]
    total = len(translated)
    # translate_text has a bounded process-local cache, so duplicate cues
    # (common in intros, applause, or repeated chorus lines) are generated once.
    for index, subtitle in enumerate(translated, start=1):
        subtitle["translation"] = translate_text(
            subtitle.get("text", ""), target_lang, model_dir, max_tokens,
        )
        if on_progress:
            on_progress(index, total)
    return translated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Tencent Hy-MT2-1.8B for this project.")
    parser.add_argument("--download", action="store_true", help="Download the official model files.")
    parser.add_argument("--path", default=default_model_dir(), help="Local model directory.")
    args = parser.parse_args()
    if args.download:
        print(download_model(args.path))
    else:
        print(model_status(args.path))
