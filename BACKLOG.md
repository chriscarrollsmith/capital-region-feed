# Matcher & feed quality backlog

Living backlog for improving Capital Region feed quality beyond the current
precision-first regex gate in `server/matcher.py`.

## Product goals

- **Balance false negatives and false positives.** Precision remains important
  (SkyFeed-style off-region noise is unacceptable), but recall is now a first-class
  goal: local posts should not need to say “Albany” to appear.
- **Include local voices without placenames.** Posts from Capital Region
  businesses, clubs, nonprofits, and influencers should be eligible even when the
  text never mentions location.
- **Surface upcoming regional events.** Event announcements and “this weekend in
  …” style posts should match when they are about the Capital Region, including
  cases where the venue/org implies locality more than the placename does.
- **Measure generalization.** Eval should reflect live traffic and silent misses,
  not only the curated SkyFeed false-positive set.

## Priority legend

| Priority | Meaning |
| -------- | ------- |
| P0 | Foundation / unblocker for later work |
| P1 | High impact on stated product goals |
| P2 | Valuable next stage after P0/P1 |
| P3 | Later / optional |

Status: `todo` · `in progress` · `done` · `blocked`

---

## P0 — Eval dataset & measurement

### B-001 — Expand eval beyond keyword-biased samples
- **Status:** done ([#8](https://github.com/chriscarrollsmith/capital-region-feed/pull/8))
- **Why:** Current `data/eval_cases.json` is small and skewed toward known SkyFeed
  FPs. Perfect scores there do not prove recall for local posts without placenames.
- **Work:**
  - Add labeled cases from: global/Jetstream negatives, known-local accounts,
    event announcements, and posts that *should* match with no place words.
  - Track stratified metrics (by reason bucket, by signal type: text / author /
    event).
  - Keep a holdout split so matcher changes are not fitted only to the full set.
- **Done when:** Eval suite includes explicit FN-focused cases (local org, no
  placename; regional event) and reports precision/recall with stratification,
  not only aggregate F1.

### B-002 — Improve sampling tooling for labeling
- **Status:** done ([#9](https://github.com/chriscarrollsmith/capital-region-feed/pull/9))
- **Depends on:** —
- **Work:**
  - Extend `scripts/collect_eval_sample.py` (or add siblings) to pull:
    - posts from candidate local DIDs/handles
    - firehose/near-miss negatives (ambiguous place names that should drop)
    - event-like posts (date/time/venue cues)
  - Make it easy to append hand labels into `data/eval_cases.json`.
- **Done when:** A documented loop can grow the dataset without only sampling
  from an existing place-name feed.

### B-003 — Rebalance product policy in docs/eval
- **Status:** done ([#10](https://github.com/chriscarrollsmith/capital-region-feed/pull/10))
- **Work:** Update README matching notes and eval expectations so “precision over
  recall for bare Albany” is no longer the sole policy; document the dual FN/FP
  goal and the author/event recall targets.
- **Done when:** Docs and eval commentary match the product goals above.

---

## P1 — Author & identity signals (recall without placenames)

### B-010 — Wire allowlists into the live indexer path
- **Status:** done ([#11](https://github.com/chriscarrollsmith/capital-region-feed/pull/11))
- **Why:** `allowlist_handles.txt` is evaluated in `match_post`, but Jetstream
  ingest typically supplies DIDs only; handle allowlisting is ineffective unless
  handles are resolved or DIDs are listed.
- **Work:**
  - Resolve DID → handle (cache) and/or populate `allowlist_dids.txt`.
  - Ensure allowlisted authors’ non-reply posts are indexed even with no local
    text cues.
  - Align eval harness with production (load real allowlist files).
- **Done when:** A post from an allowlisted local media/org account with no
  placename text is kept in production and covered by eval.

### B-011 — Curate Capital Region account lists
- **Status:** done ([#12](https://github.com/chriscarrollsmith/capital-region-feed/pull/12))
- **Depends on:** B-010 (or can start as data collection in parallel)
- **Quality bar:** Always-keep allowlisting is for **high signal/noise** Cap
  Region voices. Exclude accounts whose feeds are firehose-volume (tens to
  hundreds of posts/day), mostly non-human / templated, or generic
  business-slop. Prefer media, civic/elected, institutions, venues, and local
  orgs/creators with human-written, regionally relevant posts. Selection is an
  agent curation pass (`scripts/screen_allowlist_candidates.py` + feed skim);
  human feedback can reinforce later, but is not required to land candidates.
- **Work:**
  - Grow allowlists beyond local TV/newspaper handles to businesses, clubs,
    nonprofits, venues, municipal accounts, and relevant influencers.
  - Prefer starter packs / community lists where available; store as data files
    under `data/` with clear provenance comments.
  - Define tiers if needed (always-keep vs soft-prior).
  - Re-screen periodically; drop accounts that drift into volume/slop.
- **Done when:** A non-trivial curated set of local orgs/creators is checked in
  and exercised by eval cases that have empty/non-local-looking text.

### B-012 — Soft author priors for ambiguous posts
- **Status:** in progress
- **Depends on:** B-010, B-001
- **Work:** For authors with repeated strong local matches, allow weaker text
  (or no placename) to keep. Start rule-based; leave room for learned weights.
  v1 unlocks bare ambiguous places only (not hard negatives / not every
  no-placename post); durable counts live in `AuthorLocalStats`.
- **Done when:** Eval shows recall gains on local-author posts without a
  measurable spike in off-region FPs.

---

## P1 — Regional events

### B-020 — Event-oriented matching cues
- **Status:** todo
- **Depends on:** B-001 (event FN cases)
- **Work:**
  - Identify patterns/signals for upcoming local events: venue names, “at [local
    place]”, ticket/link domains, ISO/date phrasing + local org author, etc.
  - Decide whether event detection is regex heuristics first or part of the
    ambiguous-case classifier (B-030).
  - Add dedicated eval cases for true events and off-region event lookalikes.
- **Done when:** Representative Capital Region event posts match even when the
  only locality signal is venue/org/author, and off-region events still drop.

---

## P2 — Staged classifier (after rules + data)

### B-030 — Hybrid pipeline: regex floor + ambiguous-case classifier
- **Status:** todo
- **Depends on:** B-001, B-010
- **Work:**
  - Keep strong positives / hard negatives as a high-precision regex floor.
  - Route ambiguous / author-prior / event-near-miss cases to a second stage
    (embedding + linear model, or similar small in-process classifier).
  - Preserve `MatchResult.reason` for debugging (“strong_positive”,
    “allowlist”, “classifier”, …).
- **Done when:** Ambiguous-bucket recall improves on holdout without regressing
  hard-negative precision.

### B-031 — Bootstrap labels with an LLM judge (offline only)
- **Status:** todo
- **Depends on:** B-002
- **Work:** Use an LLM offline to propose labels/rationales for unlabeled
  samples; humans confirm before cases enter the scored eval set. Do not require
  a live LLM in the feed path.
- **Done when:** Labeling throughput improves and the human-confirmed set grows
  materially.

### B-032 — Entity / gazetteer disambiguation (optional track)
- **Status:** todo
- **Work:** Resolve place strings to geo entities; keep Capital Region hits,
  drop Albany Park / New Albany / etc. more systematically than negative regex.
- **Done when:** Homograph FPs are covered by entity identity rather than
  one-off patterns—or the approach is rejected with notes.

---

## P3 — Ranking & feed UX (post-matching)

### B-040 — Engagement / recency ranking among matches
- **Status:** todo
- **Why:** Orthogonal to match quality, but useful once recall rises.
- **Work:** Optional ranking among indexed posts (likes/reposts/recency);
  muted keywords; per-user preferences later if needed.

### B-041 — Language-aware heuristics
- **Status:** todo
- **Work:** Use Jetstream `langs` (already reserved in matcher) and simple
  language cues to reduce non-local FPs (e.g. French *colonie*) without harming
  English local recall.

---

## Suggested sequencing

```text
B-001/B-002/B-003  dataset + policy
        │
        ▼
B-010/B-011/B-012  author signals  ──►  local orgs/influencers without placenames
        │
        ▼
B-020              event cues
        │
        ▼
B-030/B-031        hybrid classifier + label bootstrap
        │
        ▼
B-032/B-040/B-041  optional disambiguation, ranking, langs
```

## Out of scope (for now)

- Live LLM classification in the Jetstream hot path
- Mutating the published Bluesky feed record via `publish_feed.py` from agents
- Replacing the entire matcher in one jump without an expanded eval set

## How to add items

Append a new `B-xxx` under the right priority. Include status, why, work bullets,
and a crisp “done when.” Prefer linking related PRs/commits in the status line
as work lands.
