# AGENTS.md

## Cursor Cloud specific instructions

Single Python 3.14 service: a self-hosted Bluesky custom feed generator for the
Albany/Capital Region. See `README.md` for the full architecture and command
reference. Dependencies are managed with [uv](https://docs.astral.sh/uv/)
(`pyproject.toml` + `uv.lock`); the project virtualenv lives at `.venv`.

### Running / testing / lint

- Prefer `uv run …` (or activate `.venv` after `uv sync`).
- Install/sync: `uv sync`
- Tests (no network needed): `uv run pytest -q`
- Matcher precision/recall eval (no network needed):
  `uv run python scripts/eval_filter.py`
- Lint/format: `uv run ruff check .` and `uv run ruff format .`
- Type check: `uv run ty check`
- Run the server (dev): `uv run python -m server` (uvicorn on `PORT`, default
  8080).

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
- The Jetstream consumer runs in a background daemon thread started from the
  FastAPI lifespan in `server/app.py` (when uvicorn boots the app). Egress to
  `wss://jetstream2.us-east.bsky.network` works in this environment; live Capital
  Region posts flow into SQLite within seconds and surface via
  `GET /xrpc/app.bsky.feed.getFeedSkeleton?feed=$FEED_URI`. The feed is empty
  only until a matching post arrives.
- `publish_feed.py` requires real Bluesky app-password credentials and mutates a
  live feed record — do NOT run it in cloud agents.
