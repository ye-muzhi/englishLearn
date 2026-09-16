import json


def test_local_api_health_and_token_boundary(monkeypatch, tmp_path):
    from englishlearn import api_server
    from englishlearn.processing import pipeline_runner
    from englishlearn.storage import project_store

    monkeypatch.setattr(project_store, "WORK_DIR", str(tmp_path))
    pipeline_runner.configure_task_store(str(tmp_path))
    status, health = api_server.route_get("/health", "", "test-token")
    assert status == 200 and health["ok"] is True
    status, body = api_server.route_get("/api/v1/projects", "", "test-token")
    assert status == 401 and body["error"] == "unauthorized"
    status, projects = api_server.route_get(
        "/api/v1/projects", "test-token", "test-token",
    )
    assert status == 200 and projects == {"ok": True, "projects": []}


def test_desktop_streamlit_options_are_loopback_only():
    from englishlearn.desktop_entry import streamlit_options

    options = streamlit_options(8501)
    assert options["server_address"] == "127.0.0.1"
    assert options["server_port"] == 8501
    assert options["server_headless"] is True
    assert options["server_fileWatcherType"] == "none"


def test_tauri_desktop_project_is_structurally_complete():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "desktop/src-tauri/tauri.conf.json").read_text())
    capabilities = json.loads((root / "desktop/src-tauri/capabilities/default.json").read_text())

    assert config["identifier"] == "com.englishlearn.desktop"
    assert config["build"]["frontendDist"] == "../frontend"
    assert config["bundle"]["externalBin"] == ["binaries/englishlearn-server"]
    assert any(permission == "core:default" for permission in capabilities["permissions"])
    shell_permission = next(
        item for item in capabilities["permissions"] if isinstance(item, dict)
    )
    assert shell_permission["allow"][0]["name"] == "binaries/englishlearn-server"
    rust_source = (root / "desktop/src-tauri/src/lib.rs").read_text()
    assert "tauri_plugin_single_instance::init" in rust_source
    assert (root / "desktop/frontend/index.html").exists()
    assert (root / ".github/workflows/desktop-release.yml").exists()
