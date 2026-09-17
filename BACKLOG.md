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
- **Status:** done ([#13](https://github.com/chriscarrollsmith/capital-region-feed/pull/13))
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
- **Status:** done ([#14](https://github.com/chriscarrollsmith/capital-region-feed/pull/14))
- **Depends on:** B-001 (event FN cases)
- **Work:**
  - Identify patterns/signals for upcoming local events: venue names, “at [local
    place]”, ticket/link domains, ISO/date phrasing + local org author, etc.
  - Decide whether event detection is regex heuristics first or part of the
    ambiguous-case classifier (B-030).
  - Add dedicated eval cases for true events and off-region event lookalikes.
- **Done when:** Representative Capital Region event posts match even when the
  only locality signal is venue/org/author, and off-region events still drop.
- **Notes:** v1 is regex heuristics (`_EVENT_CUE` + `_LOCAL_EVENT_VENUE` →
  `event_local_venue:*`). Ticket/link domains deferred until ingest exposes URL
  fields beyond text/alt. Classifier routing left for B-030.

---

## P2 — Staged classifier (after rules + data)

### B-030 — Hybrid pipeline: regex floor + ambiguous-case classifier
- **Status:** done ([#15](https://github.com/chriscarrollsmith/capital-region-feed/pull/15))
- **Depends on:** B-001, B-010
- **Work:**
  - Keep strong positives / hard negatives as a high-precision regex floor.
  - Route ambiguous / author-prior / event-near-miss cases to a second stage
    (embedding + linear model, or similar small in-process classifier).
  - Preserve `MatchResult.reason` for debugging (“strong_positive”,
    “allowlist”, “classifier”, …).
- **Done when:** Ambiguous-bucket recall improves on holdout without regressing
  hard-negative precision.
- **Notes:** v1 is an in-process linear scorer (`server/classifier.py`) with
  checked-in weights at `data/models/ambiguous_clf_v1.json`. Soft priors still
  hard-keep before the classifier; classifier reasons are not strong-match
  priors. Eval bucket `ambiguous_classifier` covers neighborhood/micro + event
  keeps and precision anchors.

### B-031 — Bootstrap labels with an LLM judge (offline only)
- **Status:** done
- **Depends on:** B-002
- **Work:** Use an LLM offline to propose labels/rationales for unlabeled
  samples; humans confirm before cases enter the scored eval set. Do not require
  a live LLM in the feed path.
- **Done when:** Labeling throughput improves and the human-confirmed set grows
  materially.
- **Notes:** `scripts/llm_label_judge.py` proposes JSONL labels via DeepSeek by default (`DEEPSEEK_API_KEY`, `deepseek-v4-pro`, `https://api.deepseek.com/v1/chat/completions`; OpenAI fallback). Output always sets `needs_human_confirm`; `append_eval_cases.py` remains the only path into `eval_cases.json`. Live DeepSeek pass added 25 human-confirmed cases (3 bare-Albany recall gaps marked `regression: false`).

### B-032 — Entity / gazetteer disambiguation (optional track)
- **Status:** done
- **Work:** Resolve place strings to geo entities; keep Capital Region hits,
  drop Albany Park / New Albany / etc. more systematically than negative regex.
- **Done when:** Homograph FPs are covered by entity identity rather than
  one-off patterns—or the approach is rejected with notes.
- **Notes:** Checked-in gazetteer at `data/gazetteer/places.json` with longest surface match; reasons `entity_other:*` / `entity_local:*`. Hard-negative regex remains as a safety floor. Bare `albany` stays on the ambiguous path.

---

## P3 — Ranking & feed UX (post-matching)

### B-040 — Engagement / recency ranking among matches
- **Status:** done
- **Why:** Orthogonal to match quality, but useful once recall rises.
- **Work:** Optional ranking among indexed posts (likes/reposts/recency);
  muted keywords; per-user preferences later if needed.
- **Notes:** `RANKING_MODE=indexed|created|engagement`. Jetstream also subscribes to like/repost commits and increments counts for indexed URIs. Built-in ACAB mutes + optional `MUTED_KEYWORDS` drop matching posts at index time; `data/blocklist_*.txt` drops curated authors. Per-user prefs deferred.

### B-041 — Language-aware heuristics
- **Status:** done
- **Work:** Use Jetstream `langs` (already reserved in matcher) and simple
  language cues to reduce non-local FPs (e.g. French *colonie*) without harming
  English local recall.
- **Notes:** `langs` threaded through indexer / `match_post` / eval. French-only `colonie` without NY cues → `lang_non_local:fr`; bilingual `en`+`fr` and Colonie NY keeps stay on the regex/entity path.

---

## P1 — Bare-Albany recall (post-backlog)

### B-050 — Close bare-Albany recall gaps without loosening precision gates
- **Status:** done ([#17](https://github.com/chriscarrollsmith/capital-region-feed/pull/17))
- **Why:** After B-031 labeling, three live/human-confirmed keeps still miss as
  `bare_albany` (`regression: false`): Alive at 5 After Party, Albany+NYC
  politics contrast, and Eufuria in Albany. Generic “Albany this weekend”
  / Veterans Day parade cases must stay dropped.
- **Work:**
  - Treat distinctive Cap Region named events (`Eufuria`, `Black Pawrade`,
    `Alive at 5 After Party`) as strong positives.
  - Treat `\bnyc\b` as NY context for bare Albany (same role as `NY` / `New York`).
  - Flip the outdated `fp-albany-unspecified-event` Eufuria precision-gate case
    to a true positive under the dual FN/FP policy.
  - Promote the three `regression: false` bare-Albany cases to regression once
    they keep; leave `gap-local-org-unknown-handle-no-place` for allowlist work.
- **Done when:** Eval keeps those bare-Albany event/local cases with precision
  still 1.000 on `precision_gate` / `skyfeed_fp` (including Veterans Day parade).

### B-051 — Grow venue/org allowlist coverage for no-placename local posts
- **Status:** done ([#18](https://github.com/chriscarrollsmith/capital-region-feed/pull/18))
- **Depends on:** B-011, B-050 (remaining tracked FN)
- **Why:** After B-050, the only scored recall gap was
  `gap-local-org-unknown-handle-no-place` — generic venue posts with no
  placename stay allowlist-only by design. Cap Region theatre, radio, film
  clubs, bookstores, and beat journalists still underrepresented on the list.
- **Work:**
  - Screen follow-graph candidates from existing local media/org seeds
    (`scripts/screen_allowlist_candidates.py`); keep high signal/noise only.
  - Add venue/arts/community orgs and Cap Region beat journalists/hosts;
    sync DIDs via `scripts/resolve_allowlist_dids.py`.
  - Replace the synthetic unlisted-venue gap with regression cases for real
    allowlisted venues (handle + DID paths).
- **Done when:** Eval recall is 1.000 with no `regression: false` gaps; new
  venue/org no-placename cases keep; precision stays 1.000 on
  `precision_gate` / `skyfeed_fp`.

---

## P2 — Indexed feed hygiene

### B-052 — Audit & purge posts that no longer match
- **Status:** done ([#21](https://github.com/chriscarrollsmith/capital-region-feed/pull/21))
- **Why:** SQLite stores URI metadata only; matcher/gazetteer/allowlist changes do
  not revisit indexed rows. Stale keeps (e.g. Troy Jackson + “New York Times”
  masthead before #20, Albany GA listings, off-region “capital region” copy)
  keep appearing in `getFeedSkeleton` until the 7-day prune.
- **Work:**
  - Add `scripts/audit_indexed_feed.py` to hydrate AppView `getFeed` / DB URIs,
    rematch with production allowlists (+ optional soft priors), and optionally
    `--purge` would-drop rows.
  - Run against the Fly volume DB and delete stale URIs.
  - Add eval anchors for live stale FPs caught in the audit.
- **Done when:** Tooling is documented; prod index has no rematch would-drops
  from the audit pass; eval covers the FP shapes that were lingering.
- **Notes:** First pass purged 12/186 prod rows (9 Troy Jackson masthead FPs,
  Albany GA listing, Madrid/`Hauptstadtregion`, 1 not_found). Follow-up: rematch
  alone misses active FPs — demoted collision micro-toponyms (delmar/ravena/
  altamont/sand lake/green island) to ambiguous, hard-negative spaced Del Mar,
  and capped embed description/quote text; re-purged prod after that fix.

---

### B-053 — Daily feed audit: collision micros, #518, New Scotland, Upstate NY, Albany County WY
- **Status:** done ([#23](https://github.com/chriscarrollsmith/capital-region-feed/pull/23))
- **Why:** 2026-07-26 live feed still kept off-region posts after #22: Metra `#518`,
  "a new Scotland" / New Scotland Shirt, bare Upstate NY (Syracuse), Central Ave /
  Lincoln Park / 14th→4th Street classifier micros, and Albany County WY near-misses.
- **Work:**
  - Narrow strong positives (`#518ny`/`#518area`, New Scotland + NY/town; drop bare
    `upstate ny` as strong).
  - Gate collision micros on Cap Region hints; fix `4th Street` digit lookbehind;
    require `doors at/open` for event cues.
  - Gazetteer + matcher conflict for Albany County Wyoming.
  - Grow eval with today's FP/TP anchors.
- **Done when:** Today's audited FPs drop; eval precision/recall stay 1.000; unit
  tests cover the new gates.

### B-054 — Daily feed audit: Center Square wire, capital regional, Canadian CR, handle NYC
- **Status:** done ([#24](https://github.com/chriscarrollsmith/capital-region-feed/pull/24))
- **Why:** 2026-07-27 live feed (39 UTC posts) kept off-region noise after #23:
  Illinois House syndication via `(The Center Square)` + November event cue
  (`classifier:local_micro`), Spanish `capital regional` matching `capital region`,
  Ottawa/Canada “capital region” + `#CanadianInnovation`, and national share posts
  where `@…albany…` + `@….nyc` unlocked `albany_with_ny_context`.
- **Work:**
  - Scrub Center Square news-wire bylines from classifier micro hits.
  - Word-boundary `capital region|district`; Canadian capital-region conflict helper.
  - Strip `@handle` mentions (not email local-parts) before ambiguous place / NYC context.
  - Grow eval with 2026-07-27 FP/TP anchors (+12 cases).
- **Done when:** Today's audited FPs drop; eval precision/recall stay 1.000; unit
  tests cover the new gates.

### B-055 — Daily feed audit: BC/Denmark/MS capital region, Brunswick/Scotia, WI Troy, Washington Park AZ, Saratoga venues
- **Status:** done ([#25](https://github.com/chriscarrollsmith/capital-region-feed/pull/25))
- **Why:** 2026-07-28 audit of last-24h feed (~142 posts) found active FPs after #24:
  Victoria BC CFAX “capital region” (`#yyj`/`#BCpoli`), Copenhagen “Capital Region Of
  Denmark” jobs, Mississippi “Capital Region Bureau”, New Brunswick/Nova Scotia
  `multi_local_places` via Brunswick+Scotia, Wisconsin East Troy+Waterford `#wiwx`,
  bare Washington Park AZ (`classifier:local_micro`), and University of Galway cards
  with a New York world mention. Rematch also dropped true Saratoga Race Course /
  SPAC tourism posts (`ambiguous_no_context:saratoga`).
- **Work:**
  - Expand Canadian capital-region conflict to BC cues; hard-negative Denmark /
    Mississippi bureau phrases.
  - Lookbehind so Nova Scotia / New Brunswick do not unlock Scotia/Brunswick.
  - Gate WI East Troy+Waterford multi-local; Galway Ireland conflict.
  - Move bare Washington Park to collision micros; keep farmers-market distinctive.
  - Strong-positive Saratoga Race Course / SPAC / Oklahoma Training Track; allowlist
    Saratoga tourism handles.
  - Grow eval with 2026-07-28 FP/TP anchors (+14 cases).
- **Done when:** Audited FPs drop; Saratoga venue FNs keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-056 — Daily feed audit: Bethlehem PA, Galway United, Albany County WY bracket, Clark/Lark, Proctor
- **Status:** done ([#26](https://github.com/chriscarrollsmith/capital-region-feed/pull/26))
- **Why:** 2026-07-29 audit of last-24h feed (63 posts) found active FPs after #25:
  Bethlehem PA tour dates unlocked by NYC context, League of Ireland
  `Galway United` + Waterford scorelines via `multi_local_places`, Cheyenne NWS
  Albany County alerts tagged `[WY]` / Laramie (prior WY gate missed brackets),
  Chicago Clark Street via `lark street` substring micro, and Dr. Sian Proctor
  matching `\bproctors?\b` venue. Rematch also surfaced `#SPAC` + Saratoga as a
  recall gap versus full “Saratoga Performing Arts Center” phrasing.
- **Work:**
  - Bethlehem PA conflict helper; expand Galway Ireland to United/FC + Waterford.
  - Albany County WY conflict accepts `[WY]` and Laramie.
  - Word-boundary `\blark street`; require `\bproctors\b` (theatre), not surname.
  - Strong-positive `#spac` co-occurring with Saratoga.
  - Grow eval with 2026-07-29 FP/TP anchors.
- **Done when:** Audited FPs drop; `#SPAC`+Saratoga keeps; eval P/R stay 1.000;
  unit tests cover the new gates.


### B-057 — Daily feed audit: troy weight, MD capital region, SPAC Springs, Museum of Racing, New Albany bus
- **Status:** done ([#28](https://github.com/chriscarrollsmith/capital-region-feed/pull/28))
- **Why:** 2026-07-30 audit of last-day feed (34 posts; feed quiet after ~11:00 UTC) found active FPs after #26:
  antique "10.8 troy" + NYC kept as Troy NY, and Maryland "capital region" listings for
  Montgomery / Prince George's counties via bare `capital region` strong positive. Rematch
  also dropped true Cap Region posts: Saratoga Springs Performing Arts Center, National
  Museum of Racing, racing "debut at Saratoga", and Empire State Plaza coverage blocked by
  hard-negative `New Albany Bus Station`.
- **Work:**
  - Exclude troy weight (`troy oz` / digit-bounded `troy`) from ambiguous Troy place hits.
  - MD/DC capital-region conflict helper (Prince George's / Maryland / DMV cues).
  - Strong-positive SPAC with optional Springs; Museum of Racing; debut at Saratoga.
  - `new albany` hard-negative exception for bus station/terminal/depot.
  - Grow eval with 2026-07-30 FP/TP anchors.
- **Done when:** Audited FPs drop; Saratoga/Empire State Plaza FNs keep; eval P/R stay 1.000;
  unit tests cover the new gates.

---

### B-059 — Daily feed audit: Victoria CRD, Clifton Park UK, Galway tourism, N.Y./NYS, troy-
- **Status:** done ([#32](https://github.com/chriscarrollsmith/capital-region-feed/pull/32))
- **Why:** 2026-08-01 audit of last-24h AppView feed (~126 posts) found active FPs and
  wire-dateline recall gaps after #28:
  Greater Victoria / Livable CRD "capital region" via Times Colonist cards; Yorkshire/Durham
  cricket at Clifton Park (England); Ireland tourism itineraries (Dingle/Galway/Kinsale)
  unlocked by NYC; hyphenated `troy-` artist domains with "New York" art titles; and
  `N.Y.` / `NYS` failing to supply NY context so `SARATOGA SPRINGS, N.Y.` / bare Albany+NYS
  dropped.
- **Work:**
  - Expand Canadian capital-region geo cues (Greater Victoria, Livable CRD, timescolonist).
  - Clifton Park UK cricket conflict helper (Yorkshire / Durham / cricket).
  - Broaden Galway Ireland cues (Ireland, VisitIreland, Wild Atlantic Way, Dingle, Kinsale).
  - Treat `N.Y.` and `NYS` as NY context; wire dateline strong positives for Albany/Saratoga/Troy.
  - Ignore hyphenated `troy-` names/domains in ambiguous Troy matching.
  - Grow eval with 2026-08-01 FP/TP anchors.
- **Done when:** Audited FPs drop; `N.Y.`/`NYS` Cap Region posts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-060 — Daily feed audit: Malta/Rotterdam AIS, LA/PA capital region
- **Status:** done ([#33](https://github.com/chriscarrollsmith/capital-region-feed/pull/33))
- **Why:** 2026-08-02 audit of last-24h AppView feed (~102 posts) found active FPs after #32:
  European AIS `Flag: Malta` + `Dest.: ROTTERDAM` unlocked `multi_local_places` (Town of
  Malta/Rotterdam NY collisions); WBRZ Baton Rouge weather kept on bare `Capital Region`
  (often with no LA cue in the body); Harrisburg `Capital Region Water` /
  `Pennsylvania Capital Region` also kept. Rematch already dropped stale Malta/Troy MI /
  Jacksonville Brunswick rows from prior gates. No new Cap Region FNs in the window
  (search misses were replies, MD/Canadian capital-region drops, or Proctor surname).
- **Work:**
  - Malta/Europe AIS conflict helper (MMSI / VesselAlert / Flag: Malta / LMML / Portugal).
  - Louisiana capital-region conflict (Baton Rouge / WBRZ handle gate).
  - Pennsylvania capital-region conflict + hard-negative `Capital Region Water`.
  - Grow eval with 2026-08-02 FP/TP anchors.
- **Done when:** Audited FPs drop; Malta/Rotterdam NY counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-061 — Daily feed audit: Sacramento/Korea capital region, Rotherham Clifton Park, Snowbirds, Caffe Lena
- **Status:** done ([#34](https://github.com/chriscarrollsmith/capital-region-feed/pull/34))
- **Why:** 2026-08-04 audit of last-24h AppView feed (~171 posts since 2026-08-03T09:00Z)
  found active FPs after #32/#33: Sacramento Bee / Seoul–Gyeonggi "capital
  region"; Times Colonist Snowbirds / Parkland Secondary cards; Rotherham Show at Clifton
  Park (England) beyond the cricket-only UK gate. Also restored recall for Caffe Lena and
  High Rock Park Pavilions in Saratoga Springs without `, NY`. WBRZ Baton Rouge remains
  covered by the B-060 Louisiana gate.
- **Work:**
  - California capital-region conflict (Sacramento / SacBee handle).
  - Korea capital-region conflict (Seoul / Gyeonggi / Greater Seoul).
  - Expand Clifton Park UK cues (Rotherham / .gov.uk / South Yorkshire).
  - Expand Canadian capital-region cues (Snowbirds, Parkland Secondary, timescolonist handle).
  - Strong positives for `Caffe Lena` and High Rock Park + Saratoga.
  - Grow eval with 2026-08-04 FP/TP anchors.
- **Done when:** Audited FPs drop; Saratoga venue FNs keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-062 — Daily feed audit: Ukrainian capital region, Montreal Scotia, Watervliet MI, The Egg
- **Status:** done ([#35](https://github.com/chriscarrollsmith/capital-region-feed/pull/35))
- **Why:** 2026-08-05 audit of last-24h AppView feed (~190 posts since 2026-08-04T09:00Z)
  found active FPs after #32/#33/#34: AP/wire "Ukrainian capital
  region" / Kyiv mirrors (~10 keeps); Osheaga Scotia Forest Stage and Cinéma Banque Scotia
  Montréal + New York plot unlocked Village of Scotia; Galway tourism with "Irish" (not
  "Ireland") + New York; Watervliet, MI hospital jobs via bare `watervliet` entity/strong.
  FN: The Egg + Albany tour footnotes without an explicit NY token.
- **Work:**
  - Ukraine capital-region conflict + hard-negative `ukrainian capital region`.
  - Scotia Montreal conflict (Banque Scotia / Osheaga / Parc Jean-Drapeau).
  - Expand Galway Ireland cues (`irish`, day-tripping from Galway).
  - Gazetteer `watervliet_mi` + MI conflict helper / hard-negative block.
  - Strong positive for The Egg co-occurring with Albany.
  - Grow eval with 2026-08-05 FP/TP anchors.
- **Done when:** Audited FPs drop; Scotia/Watervliet/Egg Cap Region counterparts keep;
  eval P/R stay 1.000; unit tests cover the new gates.

### B-063 — Daily feed audit: Rotherham Watersplash, LA Ascension/WBRZnews2, Sudan, Schenectady hashtags, River Street
- **Status:** done ([#36](https://github.com/chriscarrollsmith/capital-region-feed/pull/36))
- **Why:** 2026-08-06 audit of last-24h AppView feed (~234 posts since 2026-08-05T09:00Z)
  found active FPs after #32/#33/#34/#35: Rotherham council
  Clifton Park Watersplash (handle omitted "Rotherham" in body); Ascension Parish /
  Prairieville–Sorrento–St. Amant "Capital Region" storage ads; `wbrznews2` Capital Region
  race qualifying (prior `\bwbrz\b` handle gate missed the prefix); Sudan/Khartoum
  "capital region" UNEP cards; `#schenectadyparkcleanup` hashtag stuffing via bare
  `schenectady` substring; Canadian River Street / Quill & Quire Instagram via distinctive
  `river street` micro. Also closed hyphenated `Brussels-Capital Region`. No new Cap Region
  FNs in the window (search misses were off-region capital-region drops or non-local).
- **Work:**
  - Expand Clifton Park UK cues (`watersplash`) + `rotherham` author_handle gate.
  - Expand Louisiana capital-region cues (Ascension / Prairieville / Sorrento / St. Amant)
    and WBRZ handle prefix (`wbrznews2`).
  - Sudan/Khartoum capital-region conflict + hard-negative phrases.
  - Word-boundary `schenectady` / `guilderland` / `niskayuna` / `watervliet` strong positives.
  - Scrub Canadian River Street / Quill & Quire cues from classifier micros.
  - Hyphen-aware `Brussels-Capital Region` hard negative.
  - Grow eval with 2026-08-06 FP/TP anchors.
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-064 — Daily feed audit: Bogotá Capital District, MD Banner, CFAX, Virginia, Disney Saratoga
- **Status:** done ([#37](https://github.com/chriscarrollsmith/capital-region-feed/pull/37))
- **Why:** 2026-08-07 audit of last-24h AppView feed (~276 posts since 2026-08-06T09:00Z)
  found active FPs after #32/#33/#34/#35/#36: Bogotá Capital District
  flight trackers; Maryland Banner MoCo/PG County "capital region" weekend guides (Olney /
  National Harbor; handle omitted Maryland); CFAX Victoria BC capital-region call-ins with
  `#yyj` beyond the prior 160-char window; Virginia Spanberger/Dominion "Capital region";
  Disney Saratoga Springs / Treehouse Villas via hyphenated WDW URLs unlocking
  `multi_local_places` from nested `saratoga` ⊂ `saratoga springs`. No new matcher FNs
  (The Egg + Albany search hit was an indexing gap; matcher already keeps).
- **Work:**
  - Colombia Capital District conflict + hard-negative phrases.
  - Expand MD/DC cues (Olney Theatre, National Harbor) + `bannermoco` /
    `bannerpgcounty` author_handle gates.
  - Widen Canadian capital-region window to 240 + `cfax` handle gate.
  - Virginia capital-region conflict (Spanberger / Dominion / Richmond).
  - Disney Saratoga Springs conflict + collapse nested ambiguous tokens for multi_local.
  - Grow eval with 2026-08-07 FP/TP anchors.
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-065 — Daily feed audit: Center Square wire/Hollywood, Ottawa Citizen, Albany GA, Waterford CT, Donna Troy, Loudonville OH, Reinvent Albany
- **Status:** done ([#38](https://github.com/chriscarrollsmith/capital-region-feed/pull/38))
- **Why:** 2026-08-08 audit of last-24h AppView feed (~185 posts since 2026-08-07T09:03Z)
  found active FPs after #37: Illinois "The Center Square" dash bylines + Hollywood Squares /
  Paul Lynde "center square" classifier keeps; Ottawa Citizen "capital region" (handle/domain
  omitted from prior Canadian gate); Nielsen `Albany GA` + Brunswick `multi_local_places`
  (gazetteer lacked comma-less `albany ga`); Waterford CT beach + New London unlocked via
  Long Island NY context; Donna Troy comic + New York; Loudonville, OH HVAC via bare
  `loudonville` strong positive; Reinvent Albany NYC advocacy via `albany_with_ny_context`.
  No new matcher FNs in Cap Region search (non-reply) for the window.
- **Work:**
  - Expand Center Square wire scrub (dash bylines) + Hollywood Squares / Paul Lynde scrub.
  - Ottawa Citizen domain + `ottawacitizen` author_handle Canadian capital-region gate.
  - Gazetteer `albany ga` (and peer no-comma state surfaces) + hard-negative `albany ga`.
  - Waterford CT conflict (New London / Hartford Tpke / Connecticut).
  - Exclude `Donna Troy` from troy ambiguous place matching.
  - Hard-negative `loudonville, OH` and `reinvent albany` / `reinventalbany`.
  - Grow eval with 2026-08-08 FP/TP anchors.
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-066 — Daily feed audit: Iceland/Japan capital region, drought burnt hills, Roblox Rensselaer, rowonebrand, E Greenbush
- **Status:** done ([#39](https://github.com/chriscarrollsmith/capital-region-feed/pull/39))
- **Why:** 2026-08-09 audit of last-24h AppView feed (~176 posts since 2026-08-08T09:00Z)
  found active FPs after #38: Iceland Reykjavik "capital region" / Capital Region Police
  (`mbl.is`); Japanese senryu "capital district"; image alt `drought/burnt hills` unlocking
  Burnt Hills strong positive; Roblox `Rensselaer County` game maps; rowonebrand Albany
  city-list spam via `albany_with_ny_context`. FN: NWS `E Greenbush` / `N Greenbush`
  abbreviations missed `east|north greenbush` strong positives.
- **Work:**
  - Iceland capital-region conflict + hard-negative phrases.
  - Japan capital-district conflict (senryu / banzai / CJK window).
  - Burnt Hills descriptive (drought/wildfire) conflict.
  - Rensselaer County + Roblox conflict (entity + strong paths).
  - Hard-negative `rowonebrand`.
  - Strong-positive aliases `E.? Greenbush` / `N.? Greenbush`.
  - Grow eval with 2026-08-09 FP/TP anchors.
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-067 — Daily feed audit: Finland/Australia/Atlanta capital region, Malta Gozo film, Rotterdam NL, LOI, Loudonville OH hashtags, Aging journal, Delmar Ave
- **Status:** done ([#40](https://github.com/chriscarrollsmith/capital-region-feed/pull/40))
- **Why:** 2026-08-10 audit of last-24h AppView feed (~159 posts since 2026-08-09T09:00Z)
  found active FPs after #39: Helsinki HSL "capital region"; Canberra Rise Above Capital
  Region Cancer Relief; Atlanta ICE "capital region"; Malta/Gozo film tourism unlocking
  `multi_local` via the film title Troy; Rotterdam NL ArchDaily + ODA New York;
  League of Ireland Galway+Waterford fixture lists; `#Ohio`+`#Loudonville` strong
  positives; PubMed `Aging (Albany NY)` journal; USPS `DELMAR AVE` street + unrelated NY.
  Cap Region search found no new matcher FNs in the window (non-reply).
- **Work:**
  - Finland / Australia / Georgia-Atlanta capital-region conflicts + hard-negative phrases.
  - Expand Malta/Europe cues (Gozo / Netherlands / ArchDaily); stop bare Troy/New York
    from canceling the Europe conflict.
  - Expand Galway Ireland cues for LOI club lists (Derry City, Dundalk, Shels, Bohs).
  - Loudonville + `#Ohio` conflict; hard-negative `Aging (Albany NY)` and `Delmar Ave`.
  - Deduplicate accidental redefinitions of LA–Sudan capital-region regexes.
  - Grow eval with 2026-08-10 FP/TP anchors.
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-068 — Daily feed audit: DC Snipers capital region, PNG National Capital District, Troy Avenue, Bay Area Albany
- **Status:** done ([#41](https://github.com/chriscarrollsmith/capital-region-feed/pull/41))
- **Why:** 2026-08-11 audit of last-24h AppView feed (~181 posts since 2026-08-10T09:00Z)
  found active FPs after cherry-picking #40: Montgomery County MD `mymcmedia` DC Snipers
  "capital region" exhibit; Papua New Guinea `National Capital District` / Port Moresby
  health-authority wire (substring of Cap District strong positive); Brooklyn Troy Avenue
  + East New York / Crown Heights unlocking `ambiguous_with_context:troy`; Bay Area Albany
  (Piedmont/Atherton) + "new york city" consolidation talk → `albany_with_ny_context`.
  Cap Region search found no new matcher FNs in the window (non-reply). Intentional keeps
  include ROTTERDAM, NY (WRGB) republished on europesays.nl and Albany N.Y. Uganda wire.
- **Work:**
  - Expand MD/DC capital-region cues (`DC Snipers`, National Law Enforcement Museum) +
    `mymcmedia` handle gate.
  - Gazetteer + hard-negative `national capital district` / PNG conflict helper.
  - Hard-negative `Troy Avenue` / `Troy Street` (mirror Delmar Ave).
  - Bay Area Albany conflict before `albany_with_ny_context`.
  - Grow eval with 2026-08-11 FP/TP anchors.
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-087 — Daily feed audit: Albany wire Albuquerque, Ithaca/Tompkins Albany contrast; Glens Falls / Empire Underground FNs
- **Status:** done ([#64](https://github.com/chriscarrollsmith/capital-region-feed/pull/64))
- **Why:** 2026-09-04 audit of last-24h AppView feed (~204 posts since 2026-09-03T09:02Z)
  found active FPs: GlobeNewswire `ALBANY, N.Y.` dateline for a Curia Albuquerque /
  New Mexico facility ribbon-cutting matched `albany, n.y.` strong; Ithaca/Tompkins
  poet-laureate copy unlocked `albany_with_ny_context` via "New York" while Albany was
  only a bureaucracy contrast. FNs: `GLENS FALLS NY` climate bots and Glens Falls sports
  (no strong token); Empire Underground show/tour copy without a city qualifier. ~49
  stale AppView rows already rematch-drop (Malta/Brunswick/Bethlehem PA weather plus
  Seattle Times Union / MD-DC capital region banners gated earlier). AppView `searchPosts`
  still often 403.
- **Work:**
  - `_albany_wire_remote_conflict` after strong / Albany-with-context; Ithaca/Tompkins
    Albany contrast conflict; `\bglens\s+falls\b` and `empire underground` strong (+ venue).
  - Grow eval with 2026-09-04 FP/TP/FN anchors (builds on #63 / B-086).
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-086 — Daily feed audit: New York Times Union, Ithaca Troy Road, NWS near Troy, Troy Achilles film; Yaddo / Rotterdam CC FNs
- **Status:** done ([#63](https://github.com/chriscarrollsmith/capital-region-feed/pull/63))
- **Why:** 2026-09-03 audit of last-24h AppView feed (~203 posts since 2026-09-02T09:01Z)
  found active FPs after #62 (still open): Front Office Sports / Kalshi cards titled
  `New York Times Union Demands…` matched `times union` strong (York lookbehind gap after
  Seattle gate); Ithaca/Tompkins `Troy Road` solar wires unlocked by `#NY`; NWS Binghamton
  `over Springfield, or near Troy` bypassed the `over Troy` Bradford PA gate; YouTube
  `Troy Achilles Speech` / Brad Pitt `Troy (2004)` unlocked by NY politician names. FNs:
  `Yaddo` mansion with bare Saratoga; `@rotterdamcc.bsky.social` concert copy saying only
  "Don't forget, Rotterdam!" (community-center strong requires the full phrase). ~49 stale
  AppView rows already rematch-drop (Malta/Waterford/Troy MI weather). AppView `searchPosts`
  still often 403. Folded onto #62 branch.
- **Work:**
  - `(?<!york\s)` Times Union lookbehind + `new york times(?: union)?` hard-neg;
    Ithaca/Tompkins Troy Road conflict; expand Troy PA `near Troy` / NWS Binghamton cues;
    Troy Achilles / Myrmidons / Brad Pitt / `(2004)` person-film conflict; `\byaddo\b`
    strong; allowlist Rotterdam Community Center handle+DID.
  - Grow eval with 2026-09-03 FP/TP/FN anchors (builds on #62 / B-085).
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-085 — Daily feed audit: Seattle Times union, Helderberg #southafrica, Schaghticoke Rd CT, Watervliet MI HS, Troy PA, Rotterdam world cities, Troy Johnson; Riverfront Jazz / Powers Park / Saratoga Derby FNs
- **Status:** done ([#62](https://github.com/chriscarrollsmith/capital-region-feed/pull/62))
- **Why:** 2026-09-02 audit of last-24h AppView feed (~228 posts since 2026-09-01T09:03Z)
  found active FPs after #61 (still open): `Seattle Times union` / `Seattle Times Union`
  matched `times union` strong; Cape/SA `Helderberg` + `#southafrica` (no space) bypassed
  `south africa` hard-neg; Kent CT `Schaghticoke Rd.` matched bare `schaghticoke` strong;
  Bridgman/Buchanan/Red Arrow HS ratings kept `entity_local:watervliet_ny` without MI tokens;
  NWS Binghamton `Bradford County` / `over Troy` unlocked by Binghamton NY; Archinect
  world-city architecture list unlocked NL Rotterdam via NYC; AALBC `Founder, Troy Johnson`
  unlocked by New York City. FNs: Albany Riverfront Jazz / Jennings Landing; Troy Powers
  Park; Saratoga Derby / Saratoga barn; Liberty Park Albany; Burdett Birth Center; Rotterdam
  Community Center. ~53 stale AppView rows already rematch-drop (Malta/Brunswick/Stillwater).
  AppView `searchPosts` still often 403. Folded onto #61 branch.
- **Work:**
  - Seattle Times lookbehind + hard-neg; Helderberg `#southafrica`/`Sweet Paws` gates;
    Schaghticoke CT conflict; Watervliet MI HS cues; Troy PA Bradford conflict; Rotterdam
    world-city/`archinect` malta-europe cues; Troy Johnson / founder person conflict;
    strong positives for Riverfront Jazz, Powers Park, Saratoga Derby/barn, Liberty Park,
    Burdett, Rotterdam Community Center.
  - Grow eval with 2026-09-02 FP/TP/FN anchors (builds on #61 / B-084).
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-084 — Daily feed audit: Latham & Watkins office, Oregon Mid-Willamette South/West Albany; LASNNY / Amtrak ALB window / Saratoga Special / 1777 FNs
- **Status:** done ([#61](https://github.com/chriscarrollsmith/capital-region-feed/pull/61))
- **Why:** 2026-09-01 audit of last-24h AppView feed (~180 posts since 2026-08-31T09:01Z)
  found active FPs after #60: law-firm `Latham & Watkins' Hong Kong Office` matched the
  reverse `latham…office|hq` HQ strong; OregonLive Mid-Willamette / South Albany & West
  Albany HS football cards matched `south albany` strong. FNs: `LASNNY` foreclosure
  attorney (bare Albany); Amtrak Empire Service status where `Albany (ALB)` sits >160 chars
  below the `AMTRAK` header (and `NYP->ALB` arrows); DRF `Grade 2 Saratoga Special` without
  leading `the`; America 250 `1777` / Saratoga turning-point posts. ~22 stale AppView rows
  already rematch-drop (Malta / Saratoga CA / Waterford IE). AppView `searchPosts` still 403.
- **Work:**
  - Hard-neg + blocks-strong for Latham & Watkins and Oregon Mid-Willamette / South|West
    Albany HS; narrow Latham HQ strong to `office|hq in Latham` / Latham regional HQ (no
    Watkins); add Latham Circle/Farms; widen Amtrak↔Albany(ALB) window + Empire Service /
    NYP→ALB arrows; `\blasnny\b`; optional `the`/Grade 2 on Saratoga Special; `1777`↔Saratoga.
  - Grow eval with 2026-09-01 FP/TP/FN anchors.
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-083 — Daily feed audit: Troy Deeney/Mt Kisco, 99 River Street film, New Brunswick URL/`tartomány` NY; Albany Med / Amtrak ALB / Saratoga breeze / BJ's Rotterdam FNs
- **Status:** done ([#60](https://github.com/chriscarrollsmith/capital-region-feed/pull/60))
- **Why:** 2026-08-31 audit of last-24h AppView feed (~116 posts since 2026-08-30T09:02Z)
  found active FPs after #59 (still open): footballer `Troy Deeney` + vocative
  `Welcome to Mt Kisco, NY, Troy.` unlocked `ambiguous_with_context:troy`; boxing-movie
  podcast `99 River Street` / `#filmnoir` unlocked `classifier:local_micro`; Hungarian
  `New Brunswick` card kept via `new-brunswick-…` URL slug (hyphen bypasses `new `
  lookbehind) plus false `ny` inside `tartomány` unlocking `_NY_CONTEXT`. FNs from
  allowlisted local media without placename: `Albany Medical Center` / `Albany Med Health
  System`; Amtrak Maple Leaf `stopped in Albany (ALB)`; workout `breeze … at Saratoga` /
  `Saratoga immortality`; `BJ's` warehouse in Town of Rotterdam. AppView `searchPosts`
  still 403. Folded onto #59 branch.
- **Work:**
  - Unicode `\w` gates for bare `NY`/`ny`; `new-` lookbehind for Brunswick; Troy Deeney /
    Mt Kisco person conflict; scrub 99 River Street / film-noir micros; strong positives for
    Albany Med, Amtrak ALB, Saratoga breeze/immortality, BJ's Rotterdam.
  - Grow eval with 2026-08-31 FP/TP/FN anchors (builds on #59 / B-082).
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-082 — Daily feed audit: Boston River Street, Bethlehem PA steel/UNESCO, person Troy, Schenectady-style, Albany State U; CapitalRep / Travers Day / Saratoga Campaign FNs
- **Status:** done ([#59](https://github.com/chriscarrollsmith/capital-region-feed/pull/59))
- **Why:** 2026-08-30 audit of last-24h AppView feed (~183 posts since 2026-08-29T09:10Z)
  found active FPs after #58: Boston Open Streets `River Street` / Mattapan unlocked
  `classifier:local_micro`; Bethlehem PA CNN steel/UNESCO/Christmas card unlocked by
  "upstate NY"; person-named `Jana and Troy` unlocked by New York State; Wilmington
  `Schenectady-style` food; `Albany State University` (GA) card wrongly saying capital
  city. FNs: `#CapitalRep` / Proctors Collaborative without event cue; `Travers Day` /
  `Travers weekend` / `ahead of the Travers`; bare `Saratoga Campaign`. ~23 stale AppView
  rows already rematch-drop (Malta/Scotia OR/Troy Perry). AppView `searchPosts` still 403.
- **Work:**
  - Scrub Boston/Mattapan/Blue Hill River Street from classifier micros; expand Bethlehem PA
    heritage cues; person-name `and Troy,` conflict; Schenectady-style cuisine gate; gazetteer
    + hard-neg Albany State University; strong positives for CapitalRep / Travers Day|weekend /
    Saratoga Campaign.
  - Grow eval with 2026-08-30 FP/TP/FN anchors.
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-081 — Daily feed audit: Sun-Times/Times Union, Newtonville MA, DC Capital district, French/Iceland CR, Green Island Sangha, Center Square reports, CT Capital District, Albany County Library WY, Brunswick Schools; Rensselaer sheriff / Latham–Halfmoon HQ FNs
- **Status:** done ([#58](https://github.com/chriscarrollsmith/capital-region-feed/pull/58))
- **Why:** 2026-08-29 audit of last-24h AppView feed (~204 posts since 2026-08-28T09:00Z)
  found active FPs after #57 (still open): Chicago `Sun-Times union` matched `times union`;
  Boston Newtonville / MBTA Worcester Line matched Colonie `newtonville`; DC
  `Capital district, Washington, D.C.` lacked capital-district coverage in the MD/DC gate;
  Paris `French capital region` YouTube cards; Iceland `Capital Region` + Kringlan/mbl.is
  beyond the prior 160-char window; Long Island `Green Island Sangha` + Adelphi unlocked by
  NY context; classifier micros for prose `The Center Square reports`; CT East Hartford
  bank-job `Capital District`; Wyoming `Albany County Library` (WPM) without WY place tokens;
  Brunswick Schools OH + NYC tournament cards. FNs: bare `Rensselaer sheriff` ICE suit;
  BizJournals Latham regional HQ / Halfmoon HQ without `, NY`. AppView `searchPosts` still 403.
- **Work:**
  - Guard `times union` with hyphen lookbehind; Newtonville MA/MBTA conflict; extend MD/DC
    gate to capital district; French capital-region conflict; widen Iceland window + Kringlan;
    Green Island Sangha/Adelphi/LI hard-neg; scrub `The Center Square reports`; CT Capital
    District conflict; Albany County Library / WPM WY gate; Brunswick Schools+NYC conflict;
    strong positives for Rensselaer sheriff and Latham/Halfmoon HQ copy.
  - Grow eval with 2026-08-29 FP/TP/FN anchors (builds on #57 / B-080).
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-080 — Daily feed audit: Woodbine/#saratoga, Center Square Rd NJ, Crossgates Tammany, Chippewa River Street, onça-troy; Forego/Jerkens/Albany Exec FNs
- **Status:** done ([#57](https://github.com/chriscarrollsmith/capital-region-feed/pull/57))
- **Why:** 2026-08-28 audit of last-24h AppView feed (~350 posts since 2026-08-27T09:12Z)
  found active FPs after #56 (still open): multi-track handicap stuffing with `#woodbine` /
  `#charlestownraces` / `#remingtonpark` + `#saratoga` (Parx/SoCal gates missed); classifier
  micros for Gloucester Co `Center Square Rd`, St. Tammany `Crossgates Wastewater` + `#LA`,
  and Chippewa Falls `River Street` + `#Wisconsin`; Portuguese `onça-troy` gold wires unlocked
  by New York Mercantile Exchange. FNs: H. Allen Jerkens / Forego / Grade 1 at Saratoga /
  Skidmore; `Albany/Amsterdam` Legal Aid; `ALBANY EXEC`; South Albany Airport; Corinth +
  Saratoga Springs storm cluster. AppView `searchPosts` still 403.
- **Work:**
  - Extend multi-track `#saratoga` hard-neg with Woodbine/Charlestown/Remington; scrub
    Center Square Rd, Crossgates+Tammany/LA, River Street+Wisconsin from classifier micros;
    guard `onça-troy` / `onza-troy` in `_TROY_PLACE`; strong positives for named stakes /
    Grade 1 at Saratoga / Albany Exec / South Albany / Albany/Amsterdam / Corinth+Springs.
  - Grow eval with 2026-08-28 FP/TP/FN anchors (builds on #56 / B-079).
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-079 — Daily feed audit: Helderberg #CapeTown, Burnt hillside, MI hashtag Troy, Saratoga Terrace, Parx/#saratoga, Long Island #albanyny, Travers/Lark Hall/Mayor FNs
- **Status:** done ([#56](https://github.com/chriscarrollsmith/capital-region-feed/pull/56))
- **Why:** 2026-08-27 audit of last-24h AppView feed (~196 posts since 2026-08-26T09:02Z)
  found active FPs after #55 (still open): Cape Town municipal alerts with `Helderberg College`
  + `#CapeTown` (prior SA gate required spaced `cape town` within 100 chars); Huddersfield
  wildfire alt `Burnt hillside` matched strong `burnt hills` without a trailing word boundary;
  Michigan metro hashtag dumps (`#Michigan`…`#Troy`…`#Waterford`) unlocked `multi_local_places`
  beyond the 40-char Troy/MI window; Binghamton `Saratoga Terrace` housing + `#NY`; multi-track
  thoroughbred hashtag stuffing (`#parxracing`/`#thistledown`/`#horseshoeindy` + `#saratoga`)
  without `#socal`; Long Island SEO tags with `#albanyny`. FNs: BloodHorse Travers / "wins at
  Saratoga" / "Saratoga feature"; Lark Hall venue listings; `Albany Mayor` politics. AppView
  `searchPosts` still 403.
- **Work:**
  - Expand Helderberg SA hard-neg (`Helderberg College`, `#capetown`, 280-char window);
    `burnt hills\b` + hillside/gorse descriptive conflict; widen Michigan `#troy` hashtag
    window; hard-neg `Saratoga Terrace` and Parx/Thistledown/Assiniboia `#saratoga` stuffing;
    Long Island `#albany(ny)` hashtag gate; Travers / Saratoga feature / wins-at-Saratoga,
    Lark Hall, Albany Mayor strong positives.
  - Grow eval with 2026-08-27 FP/TP/FN anchors (builds on #55 / B-078).
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-078 — Daily feed audit: Berlin-Brandenburg, VA Capital District, LA CRPC, 271NY Malta, Brunswick Pike, Saratoga maiden/CCC FNs
- **Status:** done ([#55](https://github.com/chriscarrollsmith/capital-region-feed/pull/55))
- **Why:** 2026-08-26 audit of last-24h AppView feed (~187 posts since 2026-08-25T09:05Z)
  found active FPs after #54 (still open): Berlin-Brandenburg "capital region" games-funding
  wires; Richmond.com VHSL "Capital District" HS football (Henrico/Hanover); VA/MD
  `#CapitalRegion` sweeps where `#VirginiaNews`/`#MarylandNews` lacked word boundaries and
  geo windows were too tight; Louisiana Gonzales/Ascension "Capital Region Planning
  Commission" (CRPC) with cues >160 chars from the phrase; Wizz Air Malta unlocked by
  aircraft type suffix `A321-271NY`; USPS `Brunswick Pike` NJ dumps unlocked by Far
  Rockaway `NY`. FNs: Maiden Watch / races at Saratoga; CCC battlefield preservation at
  Saratoga. AppView `searchPosts` still 403.
- **Work:**
  - Germany Berlin-Brandenburg capital-region hard-neg + conflict helper; VHSL Capital
    District conflict (NY-aware); expand VA/MD windows + `#virginia*`/`#maryland*` hashtags;
    LA CRPC / `#la` / wider window; reject digit-prefixed `NY` aircraft suffixes; Brunswick
    Pike hard-neg; Wizz Air Malta / `9H-` cues; Saratoga maiden/CCC strong positives.
  - Grow eval with 2026-08-26 FP/TP/FN anchors (builds on #54 / B-077).
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-077 — Daily feed audit: E Greenbush boundary, Brussels slash, Helderberg SA, NBC4, Albany Post Rd, RentRedi spam, 9H-NYC, SoCal #saratoga, Saratoga meet/Schuyler FNs
- **Status:** done ([#54](https://github.com/chriscarrollsmith/capital-region-feed/pull/54))
- **Why:** 2026-08-25 audit of last-24h AppView feed (~183 posts since 2026-08-24T09:01Z)
  found active FPs after #51: Madison "white Greenbush Bakery" via `e Greenbush` substring;
  Cape Town BP Helderberg / Western Cape; Brussels Times `Brussels/Capital Region` slash;
  Malta→Palma flight tracker unlocked by aircraft reg `9H-NYC`; Daily Beast NBC4/Telemundo 44
  DC "capital region" card; Old Albany Post Road (Garrison/Putnam); bare `#rentredi` SEO spam;
  SoCal `#thoroughbreds #delmar #saratoga` hashtag stuffing; Instapundit "document obtained by
  The Center Square" classifier micro. FNs: `Saratoga meet` race wires; Fort Schuyler / Oriskany
  Saratoga campaign posts. ~37 stale AppView rows already rematch-drop (Malta ID weather /
  East Troy Phish / Latham & Watkins).
- **Work:**
  - Word-bound `E`/`N` Greenbush abbreviations; Helderberg+Cape Town hard-neg; Brussels `/`
    capital-region; strip hyphenated aircraft regs from NYC context; NBC4/Telemundo 44 MD/DC cues;
    Albany Post Road hard-neg; RentRedi requires Latham/Albany/CR; SoCal+#saratoga blocks strong;
    Center Square "obtained by" scrub; `Saratoga meet` + Fort Schuyler/Oriskany strong positives.
  - Grow eval with 2026-08-25 FP/TP/FN anchors.
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-076 — Daily feed audit: Bulgaria/Japan/Michigan capital region, Albany WY override, around-Lake, Brunswick Tulsa, Saratoga Park CA, GTA Albany, Thacher/harness/SPAC FNs
- **Status:** done ([#51](https://github.com/chriscarrollsmith/capital-region-feed/pull/51))
- **Why:** 2026-08-23 audit of last-24h AppView feed (~166 posts since 2026-08-22T09:01Z)
  found active FPs after cherry-picking #49/#50: Sofia/Bulgaria "capital region" GPS
  jamming wires; Tokyo/Ibaraki "capital region" earthquake cards; Lansing MI Capital
  Region International Airport / NWS Grand Rapids (`grr.nws`) statements; Albany County
  WY flash floods kept via `strong_positive_over_negative` (Albany, WY hard-neg rescued
  by bare `Albany County`); Georgia O'Keeffe "around Lake George" matching ambiguous
  `round lake` (missing leading word boundary); Brunswick Corp Tulsa OK jobs unlocked
  by Ambrook NYC co-listing; Montclair CA Saratoga Park via classifier; GTA IV Liberty
  City car named Albany + New York. FNs: Saratoga Springs Harness Track; Thacher State
  Park / WildPlay Thacher; Philadelphia Orchestra at SPAC (needed `orchestra` event cue).
  AppView `searchPosts` 403'd. Intentional keeps include allowlisted News10/Saratoga
  tourism and Saratoga race cards. Residual: allowlisted non-local; NYC↔Albany
  statehouse; bare-Albany jobs; person-named Troy fiction.
- **Work:**
  - Bulgaria + Japan + Michigan capital-region conflict helpers (`grr` handle gate).
  - Check Albany County WY before `strong_positive_over_negative`; expand `#wywx` /
    Cheyenne cues; block strong override in `_HARD_NEGATIVE_BLOCKS_STRONG`.
  - `\bround\s+lake\b` boundary; Brunswick OK / Saratoga Park CA / GTA Liberty City gates.
  - Strong/event positives: harness track; Thacher / WildPlay; `orchestra`/`philharmonic`
    event cues with `at SPAC`.
  - Grow eval with 2026-08-23 FP/TP/FN anchors.
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-075 — Daily feed audit: Denmark/Alberta capital region, NPS outside CR, Victoria Island Peers, Troy Fautanu/Helen/SC, Rotterdam Hotel NY, ValleyCats/Egg/Crossgates FNs
- **Status:** done ([#50](https://github.com/chriscarrollsmith/capital-region-feed/pull/50))
- **Why:** 2026-08-22 audit of last-24h AppView feed (~233 posts since 2026-08-21T09:01Z)
  found active FPs after cherry-picking #49: Danish "Capital Region, Denmark" flight
  trackers and Hovedstaden Letbane (Ishøj/Lundtofte/Gladsaxe) light-rail wires; Alberta
  `#AbPoli` / Glubish data-centre "Capital region" cards; NPS funding copy about
  "national parks outside the capital region"; Victoria BC Peers podcast via concatenated
  `vancouverisland` domain (spaced `vancouver island` cue missed); NFL Troy Fautanu +
  New York; Helen of Troy poetry + New York Times Book Review; Troy SC unlocked because
  bare `\bupstate\b` matched "Upstate South Carolina"; New York Hotel in Rotterdam book
  photos. FNs: Tri-City ValleyCats / Bruno Stadium; `The Egg presents`; Crossgates
  Commons; Park Playhouse; `#Saratoga250` / Burgoyne. AppView `searchPosts` 403'd.
  Intentional keeps include allowlisted News10/Times Union and UAlbany college-fair
  entity hits. Residual: person-named Troy + nyc-suburb alt fiction; allowlisted
  non-local; NYC↔Albany statehouse; bare-Albany jobs.
- **Work:**
  - Denmark + Alberta capital-region conflict helpers (and `-ab.` handle gate).
  - Expand Canadian cues (`vancouverisland` / `peers victoria`); MD/DC
    `national parks outside`.
  - Narrow `_NY_CONTEXT` bare `upstate` away from Carolinas; Troy SC / GSP handle gate.
  - Troy Fautanu lookbehind; Helen of Troy hard-negative; New York Hotel Rotterdam.
  - Strong/event positives: ValleyCats / Bruno Stadium / Crossgates / Park Playhouse /
    The Egg presents / `#Saratoga250` / Burgoyne.
  - Grow eval with 2026-08-22 FP/TP/FN anchors.
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-074 — Daily feed audit: Waynedale Loudonville handle, DC go-go capital region, Rotterdam Film Festival, Troy Nyhammer, Brunswick Records, ABR/RentRedi/Saratoga race recall
- **Status:** done ([#49](https://github.com/chriscarrollsmith/capital-region-feed/pull/49))
- **Why:** 2026-08-21 audit of last-24h AppView feed (~349 posts since 2026-08-20T09:01Z)
  found active FPs: Waynedale HS soccer handle vs bare Loudonville (Ohio cues only in
  body text, not handles); DC go-go "capital region" weekend cards (Banner MoCo/PG
  handles already gated; zuriberry.com was not); Rotterdam Film Festival trailers
  unlocked via NYC + ambiguous Rotterdam; Tromsø XI "Troy Nyhammer" matched the
  `troy`+`ny` strong positive without a word boundary after `ny`; Brunswick Records
  discographies `(Brunswick, 1925)` + New York studio dates unlocked Town of Brunswick.
  FNs: bare `Albany Business Review` publisher copy; RentRedi-in-Latham without NY;
  cross-country pick cards `Saratoga – Race 5` (en dash). Stale European Malta /
  Stillwater County MT / Troy University rows rematch-drop. AppView `searchPosts`
  partially 403'd. Intentional keeps include allowlisted News10/Times Union,
  Cohoes/Saratoga County local news, and residual NYC↔Albany statehouse cards.
- **Work:**
  - Gate Loudonville OH via `waynedale` author handles.
  - Add DC go-go cues to MD/DC capital-region conflict.
  - Expand Malta/Europe cues for Rotterdam Film Festival / IFFR.
  - Require `\b` after `ny` in `troy`+`ny` strong positive.
  - Hard-negative Brunswick Records discography patterns.
  - Strong positives: Albany Business Review; RentRedi; `Saratoga – Race N`.
  - Grow eval with 2026-08-21 FP/TP/FN anchors.
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-073 — Daily feed audit: I-787 study N, Troy Aikman, Loudonville OH soccer, NL New York Pizza, River Street off-region, Saratoga/Albany recall
- **Status:** done ([#48](https://github.com/chriscarrollsmith/capital-region-feed/pull/48); from [#47](https://github.com/chriscarrollsmith/capital-region-feed/pull/47))
- **Why:** 2026-08-20 audit of last-24h AppView feed (~152 posts since 2026-08-19T09:02Z)
  found active FPs after cherry-picking #43–#46: medical PR "Study on 787 Brain Tumor
  Patients" matched bare `\bon\s+787\b`; Troy Aikman + New York Jets unlocked Troy;
  Waynedale Golden Bears vs Loudonville OH soccer (no Ohio token); Dutch New York Pizza
  bankruptcies in Rotterdam (`rotterdamse` / `failliet` / Rijnmond / dagblad010);
  River Street Writing (Calgary) and Virginia River Street Networks broadband cards via
  classifier micro + month cues. FNs: `race at Saratoga` / `Stakes @ Saratoga` /
  The Saratoga Special / Battles of Saratoga; `Albany region` business copy. AppView
  `searchPosts` returned 403 in this environment (FN search limited to feed drops +
  synthetic probes). Intentional keeps include allowlisted News10/Times Union, Demon
  Hunter lawsuit alt mentioning Albany NY, Excelsior/YWCA Greater Capital Region, and
  residual NYC↔Albany statehouse cards.
- **Work:**
  - Require I-/interstate/route prefix for 787 strong positives.
  - Exclude `Troy Aikman` from ambiguous Troy (mirror Donna Troy).
  - Expand Loudonville OH cues (Waynedale / Golden Bears / OHSAA).
  - Expand Malta/Europe Rotterdam cues for Dutch New York Pizza chain copy.
  - Scrub River Street Writing / Networks / Franklin County VA / citizenportal micros.
  - Strong positives: race at Saratoga; Stakes @ Saratoga; The Saratoga Special;
    Battles of Saratoga; Albany region.
  - Grow eval with 2026-08-20 FP/TP/FN anchors.
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-072 — Daily feed audit: Russia capital region, Stillwater Road, Malta JFK tourism, Saratoga Battlefield/racing FNs
- **Status:** done ([#48](https://github.com/chriscarrollsmith/capital-region-feed/pull/48); from [#46](https://github.com/chriscarrollsmith/capital-region-feed/pull/46))
- **Why:** 2026-08-15 audit of last-24h AppView feed (~133 posts since 2026-08-14T09:01Z)
  found active FPs after cherry-picking #43/#44/#45: Russia/Moscow drone wire "capital region"
  (Ukraine-style international gate missing); Lewis Co / Croghan `Stillwater Road` unlocked via
  NY context + Buffalo NWS bots; European Malta Endless Summer / Delta JFK-Malta / Malta Tourism
  cards unlocked via New York JFK. FNs: Saratoga Battlefield tourism without `, NY`; Christophe
  Clement / Glens Falls / `#Saratoga` race cards without "stakes at Saratoga". Stale AppView
  Malta/Saratoga Av/bare-Albany rows already rematch-drop. Intentional keeps include allowlisted
  News10/Times Union non-local, NWS Albany CWA into MA/VT, and residual Gothamist "legislation
  in Albany" statehouse cards.
- **Work:**
  - Russia capital-region conflict helper + hard-negative phrase list.
  - Stillwater Road / Lewis Co / Croghan / `buf.nws` conflict before ambiguous Stillwater keep.
  - Expand Malta Europe cues (JFK-Malta, Malta Tourism, Endless Summer, Delta nonstop, eturbonews).
  - Strong positives: Saratoga Battlefield; Christophe Clement ↔ Saratoga; Glens Falls ↔ Saratoga;
    `#Saratoga` + race/odds cues; `#SaratogaRacing`.
  - Grow eval with 2026-08-15 FP/TP/FN anchors.
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-071 — Daily feed audit: Victoria BC island rail, Waterford Crystal, Norwegian Ny+Troy, Rensselaer/Saratoga FNs
- **Status:** done ([#48](https://github.com/chriscarrollsmith/capital-region-feed/pull/48); from [#45](https://github.com/chriscarrollsmith/capital-region-feed/pull/45))
- **Why:** 2026-08-14 audit of last-24h AppView feed (~134 posts since 2026-08-13T09:03Z)
  found active FPs after cherry-picking #43/#44: Vancouver Island `Capital Region` +
  Goldstream→Victoria / `restoreislandrail` (Canadian gate lacked Goldstream / island-rail
  cues and bare Victoria); Waterford Crystal Etsy cards unlocked via Huntington Station NY
  ship-from; Norwegian sentence-initial `Ny` ("New") unlocked person Troy via case-insensitive
  `\bny\b`. FNs: Thoroughbred/OTTB + Fasig-Tipton + stakes-at-Saratoga without `, NY`; City of
  Rensselaer NY civic posts (gazetteer/strong only had `Rensselaer County`). ~30 stale AppView
  rows already rematch-drop (Malta/Rotterdam/bare Albany jobs). Intentional keeps include
  allowlisted News10/Times Union non-local and NWS Albany CWA into VT.
- **Work:**
  - Expand Canadian capital-region cues (`goldstream`, Vancouver Island, restoreislandrail)
    + handle gate.
  - Hard-negative Waterford Crystal / Wedgwood; case-sensitive `NY`/`ny` abbreviation.
  - Ambiguous `rensselaer` + Indiana conflict; strong `rensselaer polytechnic`.
  - Strong positives: thoroughbred/OTTB/aftercare ↔ Saratoga; Fasig-Tipton Saratoga;
    stakes at Saratoga.
  - Grow eval with 2026-08-14 FP/TP/FN anchors.
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-070 — Daily feed audit: Iceland Capital District, Egg Art Garden KC, Center Square conjunction, ponies FN
- **Status:** done ([#48](https://github.com/chriscarrollsmith/capital-region-feed/pull/48); from [#44](https://github.com/chriscarrollsmith/capital-region-feed/pull/44))
- **Why:** 2026-08-13 audit of last-24h AppView feed (~150 posts since 2026-08-12T09:17Z)
  found active FPs after cherry-picking #43: Iceland `Capital District Fire and Rescue`
  (mbl.is / `#reykjavik`) still kept via strong `capital district` because the Iceland
  conflict only gated `capital region`; Kansas City `The Egg and Art Garden` unlocked
  `event_local_venue:at the egg`; Washington State Standard **and The Center Square**
  wire conjunction + August month cue unlocked `classifier:local_micro`. FN: Saratoga
  Springs dining + "play the ponies" without `, NY` (`ambiguous_no_context:saratoga springs`).
  Stale AppView Malta/bare-Saratoga rows already rematch-drop. Intentional keeps include
  allowlisted non-local posts and NWS Albany CWA warnings into VT.
- **Work:**
  - Expand Iceland capital-region conflict to `capital district` + Fire and Rescue branding.
  - Hard-negative Egg and Art Garden / Kansas City before The Egg venue keep.
  - Scrub Center Square wire conjunction / reporting-by attributions in classifier micros.
  - Strong positive: `play the ponies` ↔ Saratoga.
  - Grow eval with 2026-08-13 FP/TP/FN anchors.
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-069 — Daily feed audit: Stillwater film, Troy MI, van Helderbergh, Bay Area multi_local, Indiana wx, Saratoga FNs
- **Status:** done ([#43](https://github.com/chriscarrollsmith/capital-region-feed/pull/43))
- **Why:** 2026-08-12 audit of last-24h AppView feed (~186 posts since 2026-08-11T09:03Z)
  found active FPs after #41: Matt Damon *Stillwater* + New York Film Festival
  (`ambiguous_with_context:stillwater`); Troy Michigan / Detroit/Troy itineraries unlocked
  via Ithaca NYC context; Belgian sculptor "van Helderbergh" via bare `helderberg` strong
  positive; Bay Area albany+saratoga consolidation lists via `multi_local_places` (prior
  Bay Area gate was albany-only); Indiana `#inwx` Albany+Saratoga+Muncie weather via
  `multi_local_places`; Brooklyn MTA `Saratoga Av` subway + NYPD. FNs: Amtrak / bare SPAC /
  Saratoga Jazz Festival without `, NY`. ~55 stale AppView rows already rematch-drop
  (Malta/Saratoga/Troy MI jobs).
- **Work:**
  - Stillwater film conflict; Troy Michigan hard-negative + conflict; Helderberg word-boundary
    + `van Helderbergh` hard-negative.
  - Expand Bay Area cues (berkeley/cupertino/…) and apply on multi_local albany/saratoga.
  - Indiana Albany+Saratoga weather multi_local conflict (`#inwx` / Muncie / Ball State).
  - Hard-negative `Saratoga Av(enue)` (mirror Troy Avenue).
  - Strong positives: Amtrak+Saratoga Springs, bare SPAC+Saratoga, Saratoga Jazz Festival.
  - Grow eval with 2026-08-12 FP/TP/FN anchors.
- **Done when:** Audited FPs drop; Cap Region counterparts keep; eval P/R stay 1.000;
  unit tests cover the new gates.

### B-058 — Jetstream catch-up keepalive wedge (quiet feed)
- **Status:** done
- **Why:** From ~2026-07-30T16:30Z the Fly machine stayed healthy (`/healthz` 200) while
  Jetstream looped on `keepalive ping timeout` about once a minute. Cursor lagged ~24h
  behind live; AppView served stale SQLite rows. Sync `handle_event` (matcher + Peewee)
  ran on the asyncio loop that also services websocket pings; during catch-up the
  like/repost firehose starved pings. `IGNORE_ARCHIVED_POSTS` used wall-clock age, so a
  ≥24h lag would also drop the backlog as it aged out. Deep replay on one shared CPU
  also starved `did:web` HTTP enough for AppView “could not resolve identity”.
- **Work:**
  - Run `on_event` / cursor persists in a thread executor; raise `ping_timeout`.
  - Pause like/repost handling when stream lag > 2 minutes.
  - Archive check relative to event `time_us` during catch-up.
  - `/healthz` reports `jetstream_lag_s` / `jetstream_ok` (HTTP still 200).
  - Skip cursor to live when lag exceeds 15 minutes (gap beats a wedged consumer).
  - Lengthen Fly HTTP check `grace_period` so deploys do not flap did:web.
- **Done when:** Deployed consumer skips extreme lag to live, stays connected near the
  tip without ping-timeout loops, and did:web / getFeedGenerator stay online.

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
        │
        ▼
B-050              bare-Albany named-event / NYC-context recall
        │
        ▼
B-051              venue/org allowlist growth (close remaining author FN)
        │
        ▼
B-052              audit/purge indexed posts after matcher changes
        │
        ▼
B-053              daily feed audit FP gates (micros / #518 / WY)
```

## Out of scope (for now)

- Live LLM classification in the Jetstream hot path
- Mutating the published Bluesky feed record via `publish_feed.py` from agents
- Replacing the entire matcher in one jump without an expanded eval set

## How to add items

Append a new `B-xxx` under the right priority. Include status, why, work bullets,
and a crisp “done when.” Prefer linking related PRs/commits in the status line
as work lands.
