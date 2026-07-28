"""Ambiguous-case classifier for the hybrid matcher pipeline.

The regex floor in ``matcher.match_post`` keeps strong positives / hard
negatives / allowlists / soft priors / definitive event+venue hits. Remaining
ambiguous and event-near-miss candidates are scored here with a small linear
model (hand-crafted features + checked-in weights). No network or live LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

# Distinctive Capital Region neighborhoods / corridors / landmarks that are
# weaker than ``_STRONG_POSITIVE`` alone but useful when combined with event or
# place cues. Kept out of the regex floor so offhand mentions stay droppable.
_DISTINCTIVE_LOCAL_MICRO = re.compile(
    r"""
    (?:
        lark\s+street
      | pine\s+hills
      | center\s+square
      # Distinctive Albany market name — bare "Washington Park" is nationwide.
      | washington\s+park\s+farmers\s+market
      | corning\s+preserve
      | crossgates(?:\s+mall)?
      | stuyvesant\s+plaza
      | river\s+street
      | new\s+scotland\s+(?:avenue|ave|road|rd)\b
      | empire\s+state\s+plaza
      | buckingham\s+lake
      | normanskill
      | thrifty\s+shopper
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Common street/park names that exist nationwide. Only count as local micro
# when a Capital Region hint is also present (avoids Hackensack Central Ave,
# Chicago Lincoln Park, "14th"/"34th" substring traps, Guyana 4th Street, …).
_COLLISION_LOCAL_MICRO = re.compile(
    r"""
    (?:
        (?<!\d)(?:fourth|4th)\s+street
      | central\s+(?:avenue|ave)\b
      | delaware\s+(?:avenue|ave)\b
      | western\s+(?:avenue|ave)\b
      | lincoln\s+park
      | washington\s+park
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Cap Region anchors that unlock collision micros. Intentionally narrower than
# matcher ``_NY_CONTEXT`` (bare NYC / "New York" alone must not qualify).
_CAP_REGION_HINT = re.compile(
    r"""
    (?:
        capital\s+(?:region|district)\b
      | \balbany\b
      | \btroy\b(?!@)
      | schenectady
      | \bcolonie\b
      | guilderland
      | niskayuna
      | watervliet
      | \bcohoes\b
      | \blatham\b
      | \bdelmar\b
      | clifton\s+park
      | loudonville
      | \bualbany\b
      | local\s*518
      | \#518(?:ny|area)?\b
      | \#albanyny\b
      | times\s+union\b
      | rensselaer
      | \bsaratoga\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# News wire "(The Center Square)" — not Albany's Center Square neighborhood.
_CENTER_SQUARE_WIRE = re.compile(r'\(\s*the\s+center\s+square\s*\)', re.IGNORECASE)

# Back-compat alias for tests / callers that imported ``_LOCAL_MICRO``.
_LOCAL_MICRO = _DISTINCTIVE_LOCAL_MICRO


def _local_micro_hits(haystack: str) -> list[str]:
    """Return micro-signal hits eligible for classifier features."""
    # Scrub wire bylines before matching so they cannot unlock event+micro keeps.
    scan = _CENTER_SQUARE_WIRE.sub(' ', haystack)
    hits = list(_DISTINCTIVE_LOCAL_MICRO.findall(scan))
    if _CAP_REGION_HINT.search(scan):
        hits.extend(_COLLISION_LOCAL_MICRO.findall(scan))
    return hits


_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / 'data' / 'models' / 'ambiguous_clf_v1.json'
)


@dataclass(frozen=True)
class ClassifierDecision:
    matched: bool
    reason: str
    score: float


@dataclass(frozen=True)
class ClassifierModel:
    version: str
    threshold: float
    weights: dict[str, float]

    def score(self, features: dict[str, float]) -> float:
        total = self.weights.get('bias', 0.0)
        for name, value in features.items():
            if name == 'bias':
                continue
            total += self.weights.get(name, 0.0) * value
        return total


def extract_features(
    haystack: str,
    *,
    term: str | None,
    has_event_cue: bool,
    has_local_venue: bool,
) -> dict[str, float]:
    """Build the sparse feature vector for an ambiguous / near-miss candidate."""
    micro_hits = _local_micro_hits(haystack)
    local_micro_count = float(min(len(micro_hits), 3))
    local_micro = 1.0 if micro_hits else 0.0
    has_ambiguous = 1.0 if term else 0.0
    is_bare_albany = 1.0 if term == 'albany' else 0.0
    is_bare_troy = 1.0 if term == 'troy' else 0.0
    event = 1.0 if has_event_cue else 0.0
    venue = 1.0 if has_local_venue else 0.0

    return {
        'has_event_cue': event,
        'has_local_venue': venue,
        'has_ambiguous': has_ambiguous,
        'is_bare_albany': is_bare_albany,
        'is_bare_troy': is_bare_troy,
        'local_micro': local_micro,
        'local_micro_count': local_micro_count,
        'event_and_micro': 1.0 if has_event_cue and micro_hits else 0.0,
        'ambiguous_and_micro': 1.0 if term and micro_hits else 0.0,
        'event_and_ambiguous': 1.0 if has_event_cue and term else 0.0,
        'venue_without_event': 1.0 if has_local_venue and not has_event_cue else 0.0,
        'albany_event_no_micro': (
            1.0 if term == 'albany' and has_event_cue and not micro_hits else 0.0
        ),
    }


def load_model(path: Path | None = None) -> ClassifierModel:
    model_path = path or _DEFAULT_MODEL_PATH
    raw: dict[str, Any] = json.loads(model_path.read_text(encoding='utf-8'))
    weights = {str(k): float(v) for k, v in dict(raw.get('weights') or {}).items()}
    return ClassifierModel(
        version=str(raw.get('version') or 'unknown'),
        threshold=float(raw.get('threshold', 0.0)),
        weights=weights,
    )


@lru_cache(maxsize=1)
def _cached_default_model() -> ClassifierModel:
    return load_model()


def clear_model_cache() -> None:
    """Test helper: drop the cached default model."""
    _cached_default_model.cache_clear()


def classify_candidate(
    haystack: str,
    *,
    term: str | None,
    has_event_cue: bool,
    has_local_venue: bool,
    model: ClassifierModel | None = None,
) -> ClassifierDecision | None:
    """Score an ambiguous candidate; return a keep decision or None to drop.

    Returns ``None`` when the model declines so the caller can preserve the
    original regex-floor drop reason (``bare_albany``, ``no_match``, …).
    """
    features = extract_features(
        haystack,
        term=term,
        has_event_cue=has_event_cue,
        has_local_venue=has_local_venue,
    )
    # Skip scoring when there is nothing for the model to latch onto.
    if (
        features['local_micro'] < 1.0
        and features['has_local_venue'] < 1.0
        and features['has_ambiguous'] < 1.0
    ):
        return None

    clf = model or _cached_default_model()
    score = clf.score(features)
    if score < clf.threshold:
        return None

    if term:
        label = f'ambiguous:{term}'
    elif features['local_micro'] >= 1.0:
        label = 'local_micro'
    elif has_local_venue:
        label = 'venue_near_miss'
    else:
        label = 'ambiguous'
    return ClassifierDecision(True, f'classifier:{label}', score)
