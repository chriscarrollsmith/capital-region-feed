# Capital Region Bluesky Feed

Self-hosted Bluesky custom feed for **Albany, NY / Capital Region**.

Replaces the SkyFeed Builder regex feed at:

`at://did:plc:xndplob7sicvv6balxdzh3jk/app.bsky.feed.generator/aaagkkw3yejuk`

## Why this exists

The SkyFeed version matches bare place names (`albany`, `latham`, `colonie`, …) and pulls in Chicago’s Albany Park, New Albany (MS/IN), French *colonie*, NFL’s JC Latham, Saratoga Springs UT, and similar noise.

This service:

1. Subscribes to Bluesky **Jetstream** (`app.bsky.feed.post`)
2. Filters with stricter Capital Region rules + optional local account allowlists
3. Stores matches in SQLite
4. Serves `getFeedSkeleton` for Bluesky’s AppView

## Architecture

```
Jetstream ──► matcher ──► SQLite ──► getFeedSkeleton
                              ▲
                     allowlist handles/DIDs
```

Key files:

| Path | Role |
|------|------|
| `server/matcher.py` | Pure filtering logic (unit-tested) |
| `server/jetstream.py` | Websocket consumer + cursor persistence |
| `server/indexer.py` | Match → DB writes / deletes |
| `server/app.py` | XRPC + `did:web` endpoints |
| `data/eval_cases.json` | Stratified keep/drop fixtures (text, author, event signals) |
| `publish_feed.py` | Create/update the generator record (cutover) |

## Local setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14+.

```bash
uv sync
cp .env.example .env
# edit FEEDGEN_HOSTNAME / FEED_URI for local testing if needed
```

### Lint, type-check, and test

```bash
uv run ruff check .
uv run ruff format .
uv run ty check
uv run pytest -q
```

### Eval the matcher (no Bluesky login required)

```bash
uv run python scripts/eval_filter.py --verbose
```

Reports stratified precision/recall (by split, signal, and bucket). Iterate by editing
`server/matcher.py` and adding rows to `data/eval_cases.json`. Use `--strict` to also
fail on known recall-gap cases (`regression: false`).

### Run the feedgen locally

```bash
# Prefer setting these in .env (config load_dotenv overrides the shell)
# FEEDGEN_HOSTNAME=localhost
# SERVICE_DID=did:web:localhost
# FEED_URI=<your feed URI>
# DATABASE_PATH=./feed_database.db
uv run python -m server
```

Endpoints:

- `GET /healthz`
- `GET /.well-known/did.json`
- `GET /xrpc/app.bsky.feed.describeFeedGenerator`
- `GET /xrpc/app.bsky.feed.getFeedSkeleton?feed=<FEED_URI>`

## Deploy on Fly.io

Prereqs: [flyctl](https://fly.io/docs/flyctl/install/) logged in (`fly auth login`).

```bash
fly apps create capital-region-feed   # once; or rename app/hostname in fly.toml
fly volumes create feed_data --region ewr --size 1   # once if mounts aren't auto-created
fly deploy
```

Confirm:

```bash
curl -s https://capital-region-feed.fly.dev/healthz
curl -s https://capital-region-feed.fly.dev/.well-known/did.json
```

`fly.toml` keeps **one always-on machine** (`auto_stop_machines = "off"`) because Jetstream must stay connected.

## Cut over the existing feed URI

Yes — you can keep the same public feed URL/subscribers by overwriting the existing generator record’s `did` pointer.

1. Deploy the Fly app and verify `did.json` + skeleton endpoint.
2. Copy `.env.example` → `.env` and set:
   - `HANDLE` / `PASSWORD` (app password)
   - `HOSTNAME=capital-region-feed.fly.dev`
   - `RECORD_NAME=aaagkkw3yejuk`  ← existing rkey
   - display name / description as desired
3. Publish:

```bash
uv run python publish_feed.py
```

Note: prefer `FEEDGEN_HOSTNAME` / `SERVICE_DID` in `.env`. Plain `HOSTNAME` can be
overridden by the OS/shell variable (e.g. `cursor`), which would publish a bad DID.

That `putRecord`s over:

`at://did:plc:xndplob7sicvv6balxdzh3jk/app.bsky.feed.generator/aaagkkw3yejuk`

with `did:web:capital-region-feed.fly.dev` instead of `did:web:skyfeed.me`.

4. Ensure Fly env `FEED_URI` matches that URI (already set in `fly.toml`).
5. In the Bluesky app, open the feed and confirm posts hydrate from the new service (`isOnline` via `getFeedGenerator`).

**Do not paste your app password into cloud agents.** Run `publish_feed.py` on your machine.

### Rollback

Re-point the record at SkyFeed (or republish from SkyFeed’s UI) if needed, or temporarily set `SERVICE_DID=did:web:skyfeed.me` only if SkyFeed still hosts a compatible generator for that rkey.

## Matching policy

The feed optimizes for **both** false positives and false negatives:

| Goal | Target |
| ---- | ------ |
| Precision | No SkyFeed-style off-region noise (Albany Park, New Albany, DC “Capital Region”, …) |
| Recall | Local posts should not need to say “Albany” to appear — especially allowlisted authors and regional events |

Bare / ambiguous place names stay precision-gated (NY or other local context required).
Author allowlists and event/venue cues are the main recall levers for posts without
placenames. See `BACKLOG.md` for the sequenced work (allowlists → events → classifier).

### Current keep / drop rules (v1)

**Keep** when:

- Text/alt matches strong local phrases in `server/matcher.py` (`_STRONG_POSITIVE`: Capital Region, Schenectady, Niskayuna, I-787, `#AlbanyNY`, `r/Albany`, …)
- Ambiguous towns (`Troy`, `Latham`, `Saratoga Springs`, bare `Albany`, …) appear **with** NY / local context
- Author is on the local allowlist (`data/allowlist_handles.txt` / `allowlist_dids.txt`), even with no place words in the text

**Drop** hard negatives:

- Albany Park, New Albany, other state Albanys
- National Capital Region (DC), JC Latham, French *colonie*, Albany Road, Saratoga Springs UT

### Eval expectations

`data/eval_cases.json` is stratified so scores are not only about SkyFeed false positives:

- **signal:** `text` · `author` · `event` — how locality is supposed to arrive
- **bucket:** e.g. `skyfeed_fp`, `precision_gate`, `local_org_no_placename`, `regional_event`
- **split:** `dev` (iterate) vs `holdout` (report separately)
- **regression:** `false` marks known recall gaps (tracked, not CI-failing) until backlog items close them

Judge matcher changes on stratified precision **and** recall (`scripts/eval_filter.py`),
not aggregate F1 alone. Author-signal and event-signal strata are first-class recall
targets; `precision_gate` / `skyfeed_fp` must not regress.

## Suggested iteration loop

Grow the eval set from **authors / near-misses / events**, not only from an
existing place-name feed:

```bash
# Local org/media accounts (allowlist) — recall without placenames
uv run python scripts/collect_eval_sample.py authors --from-allowlist \
  > /tmp/sample-authors.jsonl

# Ambiguous / off-region homographs that should usually drop
uv run python scripts/collect_eval_sample.py near-miss \
  > /tmp/sample-near-miss.jsonl

# Event-like regional announcements
uv run python scripts/collect_eval_sample.py events \
  > /tmp/sample-events.jsonl

# Optional: still sample a live custom feed (place-name biased)
uv run python scripts/collect_eval_sample.py feed --feed "$FEED_URI" \
  > /tmp/sample-feed.jsonl
```

1. Set each row’s `expected` to `true` (keep) or `false` (drop). Adjust
   `signal` / `bucket` / `split` if the suggestions are wrong; rename `id` to a
   short slug if you prefer.
2. Append labeled rows (skips unlabeled + duplicate ids):

```bash
uv run python scripts/append_eval_cases.py --input /tmp/labeled.jsonl
```

3. Adjust `server/matcher.py` as needed.
4. `uv run python scripts/eval_filter.py && uv run pytest -q`
5. `fly deploy`

`collect_eval_sample.py search --query '…'` is available for one-off queries.
Rows already present in `data/eval_cases.json` are skipped by default.

Optional later (see `BACKLOG.md`): broader allowlists, event/venue cues, hybrid
classifier, engagement ranking, muted keywords.
