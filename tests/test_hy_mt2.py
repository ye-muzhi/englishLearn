from englishlearn.translation import hy_mt2_local as hy


class _InputIds(list):
    @property
    def shape(self):
        return (1, len(self))


class _Inputs(dict):
    def to(self, _device):
        return self


class _Torch:
    class _Inference:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def inference_mode(self):
        return self._Inference()


class _Tokenizer:
    eos_token_id = 0

    def __init__(self):
        self.prompts = []
        self.decoded_inputs = []

    def apply_chat_template(self, messages, **kwargs):
        self.prompts.append((messages, kwargs))
        return _Inputs(input_ids=_InputIds([11, 12]))

    def decode(self, generated, **_kwargs):
        self.decoded_inputs.append(list(generated))
        return "译文"


class _Model:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [[11, 12, 21, 22]]


def _runtime(tokenizer, model):
    return {
        "tokenizer": tokenizer,
        "model": model,
        "torch": _Torch(),
        "device": "cpu",
    }


def test_official_prompt_and_generation_contract(monkeypatch):
    tokenizer = _Tokenizer()
    model = _Model()
    monkeypatch.setattr(hy, "_load_runtime", lambda _path=None: _runtime(tokenizer, model))

    result = hy._generate("Hello world", "中文 (Chinese)", None, 4096)

    assert result == "译文"
    messages, template_kwargs = tokenizer.prompts[0]
    assert messages == [{
        "role": "user",
        "content": "将以下文本翻译为 中文，注意只需要输出翻译后的结果，不要额外解释：\n\nHello world",
    }]
    assert template_kwargs == {"add_generation_prompt": True, "return_tensors": "pt"}
    kwargs = model.calls[0]
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.6
    assert kwargs["top_k"] == 20
    assert kwargs["repetition_penalty"] == 1.05
    assert kwargs["do_sample"] is True
    assert kwargs["use_cache"] is True
    assert kwargs["max_new_tokens"] <= 4096
    # The prompt tokens must not leak into the displayed translation.
    assert tokenizer.decoded_inputs == [[21, 22]]


def test_duplicate_subtitle_cues_use_cache(monkeypatch):
    tokenizer = _Tokenizer()
    model = _Model()
    hy._TRANSLATION_CACHE.clear()
    monkeypatch.setattr(hy, "_load_runtime", lambda _path=None: _runtime(tokenizer, model))

    progress = []
    translated = hy.translate_subtitles(
        [{"id": 0, "start": 0, "end": 1, "text": "Repeat"},
         {"id": 1, "start": 1, "end": 2, "text": "Repeat"}],
        "en",
        max_tokens=4096,
        on_progress=lambda cur, total: progress.append((cur, total)),
    )

    assert [item["translation"] for item in translated] == ["译文", "译文"]
    assert len(model.calls) == 1
    assert progress == [(1, 2), (2, 2)]


def test_target_language_codes_are_full_native_names():
    assert hy._target_name("zh") == "中文"
    assert hy._target_name("Turkish") == "土耳其语"
    assert hy._target_name("中文 (Chinese)") == "中文"
