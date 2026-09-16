def test_seek_and_timing_refresh_cannot_accumulate_active_rows():
    import app

    html = app.build_player_html(
        "http://127.0.0.1/video.mp4",
        [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "First", "translation": "一"},
            {"id": 1, "start": 1.0, "end": 2.0, "text": "Second", "translation": "二"},
        ],
        lang="zh",
        project_id="highlight-test",
    )

    assert "function resetActiveHighlight()" in html
    assert "document.querySelectorAll('.sub-row.active')" in html
    assert (
        "video.addEventListener('seeked', function() {\n"
        "    resetActiveHighlight();\n"
        "    updateActive();"
    ) in html
    # Only the state declaration and reset helper may assign this sentinel.
    assert html.count("lastActiveId = -1") == 2
