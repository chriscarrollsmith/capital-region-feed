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
| `data/eval_cases.json` | Labeled true/false fixtures from live feed noise |
| `publish_feed.py` | Create/update the generator record (cutover) |

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit HOSTNAME / FEED_URI for local testing if needed
```

### Eval the matcher (no Bluesky login required)

```bash
python scripts/eval_filter.py --verbose
pytest -q
```

Iterate by editing `server/matcher.py` and adding rows to `data/eval_cases.json`.

### Run the feedgen locally

```bash
export HOSTNAME=localhost
export FEED_URI=at://did:plc:xndplob7sicvv6balxdzh3jk/app.bsky.feed.generator/aaagkkw3yejuk
export DATABASE_PATH=./feed_database.db
python -m server
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
python publish_feed.py
```

That `putRecord`s over:

`at://did:plc:xndplob7sicvv6balxdzh3jk/app.bsky.feed.generator/aaagkkw3yejuk`

with `did:web:capital-region-feed.fly.dev` instead of `did:web:skyfeed.me`.

4. Ensure Fly env `FEED_URI` matches that URI (already set in `fly.toml`).
5. In the Bluesky app, open the feed and confirm posts hydrate from the new service (`isOnline` via `getFeedGenerator`).

**Do not paste your app password into cloud agents.** Run `publish_feed.py` on your machine.

### Rollback

Re-point the record at SkyFeed (or republish from SkyFeed’s UI) if needed, or temporarily set `SERVICE_DID=did:web:skyfeed.me` only if SkyFeed still hosts a compatible generator for that rkey.

## Matching strategy (v1)

**Keep** when text/alt matches:

- Strong local phrases: `Albany, NY`, `Capital Region`, `Schenectady`, `Niskayuna`, `I-787`, `#AlbanyNY`, `r/Albany`, …
- Ambiguous towns (`Troy`, `Latham`, `Saratoga Springs`, bare `Albany`, …) **only with** NY / local context
- Local allowlisted accounts (see `data/allowlist_handles.txt`)

**Drop** hard negatives:

- Albany Park, New Albany, other state Albanys
- National Capital Region (DC), JC Latham, French *colonie*, Albany Road, Saratoga Springs UT

Precision is favored over recall for bare `Albany`.

## Suggested iteration loop

1. `python scripts/collect_eval_sample.py > /tmp/sample.jsonl`
2. Label new misses/false hits into `data/eval_cases.json`
3. Adjust `server/matcher.py`
4. `python scripts/eval_filter.py && pytest -q`
5. `fly deploy`

Optional later: engagement ranking, muted keywords, curated DID lists from Bluesky starter packs.
