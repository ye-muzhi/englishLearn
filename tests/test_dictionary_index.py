import csv
import time

from englishlearn.media.dictionary_index import (
    ensure_csv_index,
    lookup_sqlite,
    word_forms,
)


def _write_dictionary(path):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["word", "phonetic", "definition", "translation", "pos"],
        )
        writer.writeheader()
        writer.writerow({
            "word": "study", "phonetic": "ˈstʌdi", "definition": "to learn",
            "translation": "学习；研究", "pos": "v./n.",
        })
        writer.writerow({
            "word": "run", "phonetic": "rʌn", "definition": "move quickly",
            "translation": "跑；运行", "pos": "v.",
        })


def test_morphology_candidates_cover_common_inflections():
    assert "study" in word_forms("studies")
    assert "study" in word_forms("studied")
    assert "run" in word_forms("running")


def test_csv_builds_sqlite_index_and_inflected_lookup(tmp_path):
    csv_path = tmp_path / "ecdict.csv"
    _write_dictionary(csv_path)
    index = ensure_csv_index(str(csv_path), str(tmp_path))

    assert index
    result = lookup_sqlite(index, "studies")
    assert result["matched_form"] == "study"
    assert result["translation"] == "学习；研究"


def test_local_dictionary_hit_does_not_wait_for_network(monkeypatch, tmp_path):
    from englishlearn.media import media_manager

    dictionary_dir = tmp_path / "dictionaries"
    dictionary_dir.mkdir()
    csv_path = dictionary_dir / "ecdict.mini.csv"
    _write_dictionary(csv_path)
    monkeypatch.setattr(media_manager, "_dictionary_work_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        media_manager, "_fetch_public_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network must not run")),
    )
    media_manager._LOOKUP_CACHE.clear()

    started = time.perf_counter()
    result = media_manager.free_word_lookup("running", "zh-CN")
    elapsed = time.perf_counter() - started

    assert result["lookup_source"] == "local"
    assert result["local"]["matched_form"] == "run"
    assert "跑" in result["translation"]
    assert elapsed < 1.0
