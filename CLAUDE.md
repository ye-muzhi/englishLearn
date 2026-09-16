# CLAUDE.md

Use [AGENTS.md](AGENTS.md) as the canonical repository guide.

Quick commands:

```bash
./scripts/run_app.sh
.venv/bin/python -B -m pytest -q -p no:cacheprovider
```

The Streamlit entrypoint is `app.py`; application modules live under
`englishlearn/`; user data and local models live under `work/`.
