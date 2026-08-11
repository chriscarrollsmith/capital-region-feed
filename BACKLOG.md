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
- **Notes:** `RANKING_MODE=indexed|created|engagement`. Jetstream also subscribes to like/repost commits and increments counts for indexed URIs. `MUTED_KEYWORDS` drops matching posts at index time. Per-user prefs deferred.

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
