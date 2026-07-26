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

Merges to `main` deploy automatically via GitHub Actions (`.github/workflows/ci.yml`)
after CI checks pass, using the repo secret `FLY_API_TOKEN`.

For first-time setup or a manual deploy, use [flyctl](https://fly.io/docs/flyctl/install/)
(`fly auth login`):

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

Bare / ambiguous place names stay precision-gated (NY or other local context required),
unless the author has earned a **soft prior** from repeated strong local text matches
or the second-stage classifier keeps a neighborhood/micro + event combination.
Hard allowlists and event/venue cues remain primary recall levers for posts with no
placenames. Homographs also resolve via the checked-in gazetteer
(`data/gazetteer/places.json`). Optional feed ranking and muted keywords are
configured with `RANKING_MODE` / `MUTED_KEYWORDS`.

### Current keep / drop rules (v1)

**Keep** when:

- Text/alt matches strong local phrases in `server/matcher.py` (`_STRONG_POSITIVE`: Capital Region, Schenectady, Niskayuna, I-787, `#AlbanyNY`, `r/Albany`, named Cap Region events like `Eufuria` / `Alive at 5 After Party`, …)
- Ambiguous towns (`Troy`, `Latham`, `Saratoga Springs`, bare `Albany`, …) appear **with** NY / local context (`NY`, `NYC`, `New York`, …). National newspaper mastheads (`New York Times`, `New York Post`, …) do **not** count as place context.
- Author is on the local allowlist (`data/allowlist_handles.txt` / `allowlist_dids.txt`), even with no place words in the text. Jetstream supplies DIDs only — keep DIDs in sync with `uv run python scripts/resolve_allowlist_dids.py` after editing handles. Allowlisting targets high signal/noise Cap Region voices (not firehose-volume or business-slop accounts); screen candidates with `scripts/screen_allowlist_candidates.py`.
- Author has a soft prior (`SOFT_PRIOR_MIN_STRONG` strong text matches within `SOFT_PRIOR_WINDOW_DAYS`, tracked in `AuthorLocalStats`) and the post uses a bare ambiguous place name. Soft priors do **not** override hard negatives and do not keep arbitrary no-placename posts (that remains allowlist-only).
- Text has upcoming-event phrasing (`tonight`, `tickets`, `this weekend`, …) **and** a high-confidence Capital Region venue (`Proctors`, `MVP Arena`, `SPAC season/lawn`, `Music Haven`, `at The Egg`, …). Event cues alone never keep bare `Albany` or other ambiguous towns; off-region venues stay dropped.
- After the regex floor, the ambiguous-case classifier (`server/classifier.py`, weights in `data/models/ambiguous_clf_v1.json`) keeps posts with Capital Region neighborhood/landmark micro-signals plus event (or place) cues — reason `classifier:…`. Hard negatives never reach this stage; bare Albany events without micro-signals still drop.
- Gazetteer local entities (`entity_local:…`) keep distinctive Capital Region places/counties; other-region entities drop earlier as `entity_other:…`.

**Drop** hard negatives / non-local entities:

- Albany Park, New Albany, other state Albanys
- National Capital Region (DC), JC Latham, French *colonie*, Albany Road, Saratoga Springs UT
- French-only `langs` posts with bare *colonie* and no NY cues (`lang_non_local:fr`)

### Eval expectations

`data/eval_cases.json` is stratified so scores are not only about SkyFeed false positives:

- **signal:** `text` · `author` · `event` — how locality is supposed to arrive
- **bucket:** e.g. `skyfeed_fp`, `precision_gate`, `local_org_no_placename`,
  `regional_event`, `ambiguous_classifier`
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

1. Optionally propose labels offline with an LLM (human confirmation still
   required — do not append blindly):

```bash
# Defaults: DEEPSEEK_API_KEY + deepseek-v4-pro @ api.deepseek.com
uv run python scripts/llm_label_judge.py --input /tmp/sample-near-miss.jsonl \
  --output /tmp/proposed.jsonl
# edit /tmp/proposed.jsonl, keep only rows you confirm
```

2. Set each row’s `expected` to `true` (keep) or `false` (drop). Adjust
   `signal` / `bucket` / `split` if the suggestions are wrong; rename `id` to a
   short slug if you prefer.
3. Append labeled rows (skips unlabeled + duplicate ids):

```bash
uv run python scripts/append_eval_cases.py --input /tmp/labeled.jsonl
```

4. Adjust `server/matcher.py` as needed.
5. `uv run python scripts/eval_filter.py && uv run pytest -q`
6. Merge to `main` (CI deploys to Fly) or `fly deploy` manually

`collect_eval_sample.py search --query '…'` is available for one-off queries.
Rows already present in `data/eval_cases.json` are skipped by default.

### Audit / purge stale indexed posts

Indexed rows keep their URI until Jetstream delete or the retention prune.
After matcher changes, rematch what is still being served:

```bash
# Subscribers' view (AppView getFeed)
uv run python scripts/audit_indexed_feed.py --source feed --feed "$FEED_URI" --limit 100

# Full SQLite index (copy from Fly first; see scripts/audit_indexed_feed.py)
uv run python scripts/audit_indexed_feed.py --source db \
  --database ./feed_database.db --apply-soft-priors --jsonl /tmp/audit.jsonl

# Delete would-drop URIs from that database file
uv run python scripts/audit_indexed_feed.py --source db \
  --database ./feed_database.db --apply-soft-priors --purge
```

Ranking among matches: set `RANKING_MODE` to `indexed` (default), `created`
(author time), or `engagement` (likes + 2×reposts, updated from Jetstream
like/repost commits). Optional `MUTED_KEYWORDS` (comma-separated) skips indexing
posts that contain those substrings.
