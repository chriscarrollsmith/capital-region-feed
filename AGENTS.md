# AGENTS.md

## Cursor Cloud specific instructions

Single Python 3.12 service: a self-hosted Bluesky custom feed generator for the
Albany/Capital Region. See `README.md` for the full architecture and command
reference. Dependencies are installed into a project virtualenv at `.venv`
(refreshed automatically by the startup update script).

### Running / testing / lint

- Always invoke Python through the venv: `.venv/bin/python`, `.venv/bin/pytest`
  (there is no bare `python`/`uv` on `PATH`; only `python3`/`pip3`).
- Tests (no network needed): `.venv/bin/pytest -q`
- Matcher precision/recall eval (no network needed): `.venv/bin/python scripts/eval_filter.py`
- Run the server (dev): `.venv/bin/python -m server` (Waitress on `PORT`, default 8080).
- No linter is configured in the repo despite `.ruff_cache`/`.mypy_cache` entries
  in `.gitignore`; there is nothing to run for lint.

### Non-obvious caveats

- `.env` is gitignored. For local dev, copy `.env.example` to `.env` and override
  two values that are production-only:
  - `FEEDGEN_HOSTNAME=localhost` and `SERVICE_DID=did:web:localhost` — otherwise
    `/.well-known/did.json` returns 404 locally, and the app refuses to start if
    `FEEDGEN_HOSTNAME` is unset or equal to the shell's `HOSTNAME=cursor`.
  - `DATABASE_PATH=./feed_database.db` — the default `/data/...` path is the Fly.io
    volume mount and is not writable locally.
- `server/config.py` calls `load_dotenv(override=True)`, so values in `.env`
  win over exported shell variables. Edit `.env` to change config; exporting env
  vars in the shell will not take effect for keys present in `.env`.
- The Jetstream consumer runs in a background daemon thread started on app import
  (`server/app.py`). Egress to `wss://jetstream2.us-east.bsky.network` works in
  this environment; live Capital Region posts flow into SQLite within seconds and
  surface via `GET /xrpc/app.bsky.feed.getFeedSkeleton?feed=$FEED_URI`. The feed
  is empty only until a matching post arrives.
- `publish_feed.py` requires real Bluesky app-password credentials and mutates a
  live feed record — do NOT run it in cloud agents.
