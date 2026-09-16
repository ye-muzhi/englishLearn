from englishlearn.translation.llm_translator import _parse_response


def test_parser_rejects_missing_line_instead_of_shifting_translations():
    raw = "1. 第一条\n2. 第二条"

    assert _parse_response(raw, [40, 41, 42], model="hy-mt2") == []


def test_numbered_parser_uses_labels_when_model_reorders_lines():
    raw = "2. 第二条\n1. 第一条\n3. 第三条"

    assert _parse_response(raw, [40, 41, 42], model="hy-mt2") == [
        {"id": 40, "translation": "第一条"},
        {"id": 41, "translation": "第二条"},
        {"id": 42, "translation": "第三条"},
    ]


def test_json_parser_rebases_only_a_complete_local_batch():
    raw = '[{"id": 1, "translation": "甲"}, {"id": 2, "translation": "乙"}]'

    assert _parse_response(raw, [30, 31], model="general") == [
        {"id": 30, "translation": "甲"},
        {"id": 31, "translation": "乙"},
    ]


def test_json_parser_rejects_duplicate_or_incomplete_ids():
    raw = '[{"id": 0, "translation": "甲"}, {"id": 0, "translation": "乙"}]'

    assert _parse_response(raw, [30, 31], model="general") == []
