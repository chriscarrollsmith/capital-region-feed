"""Capital Region post matcher.

Rejects the main SkyFeed false positives:
- Albany Park (Chicago), New Albany (MS/IN), other U.S. Albanys
- French "colonie", NFL "JC Latham", Saratoga Springs UT
- Bare town names without NY / local context

Also aims for recall without placenames via author allowlists, soft author
priors (earned from repeated strong local matches), and event/venue cues
(local venue + upcoming-event phrasing). Ambiguous leftovers and event
near-misses route to a small linear classifier. Precision stays strict for
ambiguous bare names unless a soft prior or classifier keep applies; see
README matching policy and BACKLOG.md.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from server.classifier import ClassifierModel, classify_candidate


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    reason: str


# Unique phrases / places that almost always mean NY Capital Region.
_STRONG_POSITIVE = re.compile(
    r"""
    (?:
        albany\s*,?\s*(?:ny|new\s+york)
      | \#albanyny\b
      | \#albany_ny\b
      | (?:new\s+york(?:'s)?\s+)?capital\s+(?:region|district)
      | greater\s+albany
      | albany\s+county
      | rensselaer\s+county
      | schenectady\s+county
      | saratoga\s+county
      | schenectady
      | guilderland
      | niskayuna
      | watervliet
      | \bcohoes\b
      | \bdelmar\b
      | \bmenands\b
      | loudonville
      | slingerlands
      | voorheesville
      | \bravena\b
      | \baltamont\b
      | schaghticoke
      | hoosick\s+falls
      | wynantskill
      | poestenkill
      | \bschodack\b
      | \bduanesburg\b
      | \bdelanson\b
      | east\s+greenbush
      | north\s+greenbush
      | green\s+island
      | mechanicville
      | burnt\s+hills
      | ballston\s+spa
      | clifton\s+park
      | new\s+scotland
      | sand\s+lake
      | averill\s+park
      | boght\s+corners
      | newtonville
      | \bcoeymans\b
      | helderberg
      | empire\s+state\s+plaza
      | albany\s+capital\s+center
      | university\s+at\s+albany
      | \bualbany\b
      | suny\s+albany
      | \bi-?787\b
      | \bon\s+787\b
      | local\s*518
      | \#518\b
      | upstate\s+ny
      | reddit\.com/r/albany\b
      | \br/albany\b
      | saratoga\s+springs\s+police
      | saratoga\s+casino
      | times\s+union\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Ambiguous place names that need NY / Capital Region context.
_AMBIGUOUS_PLACE = re.compile(
    r"""
    (?:
        \balbany\b
      | \btroy\b
      | \blatham\b
      | \bmalta\b
      | \bscotia\b
      | \bbethlehem\b
      | \bbrunswick\b
      | \bcharlton\b
      | \bgalway\b
      | \bstillwater\b
      | \bwaterford\b
      | \brotterdam\b
      | \bhalfmoon\b
      | \bcolonie\b
      | saratoga\s+springs
      | \bsaratoga\b
      | round\s+lake
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_NY_CONTEXT = re.compile(
    r"""
    (?:
        \bny\b
      | new\s+york
      | upstate
      | capital\s+(?:region|district)
      | \#ny\b
      | \#upstateny\b
      | hudson\s+valley
      | \#albanyny\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Hard negatives that always win over an otherwise-strong local phrase
# (e.g. "capital region" inside "capital region of Madrid").
_HARD_NEGATIVE_BLOCKS_STRONG = re.compile(
    r"""
    (?:
        albany\s+park
      | new\s+albany
      | national\s+capital\s+region
      | brussels\s+capital\s+region
      | capital\s+region\s+of\s+(?:
            madrid|spain|belgium|brussels|paris|france|berlin|germany|
            tokyo|seoul|beijing|delhi|ottawa|canberra|rome|italy|
            amsterdam|vienna|warsaw|prague|lisbon|athens|dublin
          )\b
      | hauptstadtregion
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Hard negatives: other Albanys / homographs that should never match alone.
# State abbreviations require a comma/space boundary so we do not trip on
# English words like "Albany or …" / "Albany in …".
_HARD_NEGATIVE = re.compile(
    r"""
    (?:
        albany\s+park
      | new\s+albany
      | albany\s*,\s*(?:
            or|oregon|ca|california|ga|georgia|tx|texas|
            mn|minnesota|mo|missouri|ky|kentucky|in|indiana|oh|ohio|
            wy|wyoming|il|illinois|wi|wisconsin|vt|vermont
          )\b
      | albany\s+(?:
            oregon|california|georgia|texas|minnesota|missouri|
            kentucky|indiana|ohio|wyoming|illinois|wisconsin|vermont
          )\b
      | albany\s+road
      | national\s+capital\s+region
      | brussels\s+capital\s+region
      | capital\s+region\s+of\s+(?:
            madrid|spain|belgium|brussels|paris|france|berlin|germany|
            tokyo|seoul|beijing|delhi|ottawa|canberra|rome|italy|
            amsterdam|vienna|warsaw|prague|lisbon|athens|dublin
          )\b
      | hauptstadtregion
      | jc\s+latham
      | saratoga\s+springs\s*,\s*ut\b
      | saratoga\s+springs\s+ut\b
      | colonie\s+de\s+vacances
      | colonie\s+num[eé]rique
      | une\s+colonie
      | m[eê]me\s+colonie
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# "Colonie" as a NY town usually appears with these local cues.
_COLONIE_LOCAL = re.compile(
    r"""
    (?:
        town\s+of\s+colonie
      | colonie(?:\s*,)?\s*ny
      | colonie\s+(?:police|center|senior|school|high|library|fire)
      | colonie\s+senior\s+service
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Upcoming-event phrasing. Alone this never keeps a post; it only unlocks
# high-confidence local venues below (not bare Albany / ambiguous towns).
_EVENT_CUE = re.compile(
    r"""
    \b(?:
        tonight|tomorrow|this\s+weekend|this\s+saturday|this\s+sunday|
        next\s+(?:friday|saturday|sunday|week)|
        doors(?:\s+at|\s+open)?|tickets?|presale|save\s+the\s+date|join\s+us|
        open\s+mic|festival|concert|comedy\s+night|show\s+starts|
        \d{1,2}:\d{2}\s*(?:am|pm)|(?<!\d)\d{1,2}\s*(?:am|pm)\b|
        january|february|march|april|june|july|august|september|
        october|november|december
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Venues that strongly imply Capital Region locality even without a city name.
# Prefer distinctive names; keep SPAC / The Egg narrow to avoid acronym/food FPs.
_LOCAL_EVENT_VENUE = re.compile(
    r"""
    (?:
        \bproctors?\b
      | saratoga\s+performing\s+arts\s+center
      | \bat\s+spac\b
      | \bspac\s+(?:season|lawn|amphitheatre|amphitheater|presents)
      | mvp\s+arena
      | music\s+haven
      | troy\s+savings\s+bank\s+music\s+hall
      | troy\s+music\s+hall
      | cohoes\s+music\s+hall
      | caffe?\s+lena
      | \bat\s+the\s+egg\b
      | albany\s+palace\s+(?:theatre|theater)
      | palace\s+(?:theatre|theater)\s+(?:albany|in\s+albany)
      | capital\s+repertory
      | \bcap\s+rep\b
      | albany\s+civic\s+(?:theater|theatre)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


def extract_alt_text(embed: object | None) -> str:
    """Pull alt text from common Bluesky embed shapes (dict or SDK-like)."""
    if not embed:
        return ''

    chunks: list[str] = []

    if isinstance(embed, dict):
        images = embed.get('images')
        if isinstance(images, list):
            for image in images:
                if not isinstance(image, dict):
                    continue
                alt = image.get('alt')
                if alt:
                    chunks.append(str(alt))
        external = embed.get('external')
        if isinstance(external, dict):
            for key in ('title', 'description'):
                value = external.get(key)
                if value:
                    chunks.append(str(value))
        media = embed.get('media')
        if isinstance(media, dict):
            chunks.append(extract_alt_text(media))
        record = embed.get('record')
        if isinstance(record, dict):
            nested = record.get('record') or record.get('value') or {}
            if isinstance(nested, dict):
                text = nested.get('text')
                if text:
                    chunks.append(str(text))
                chunks.append(extract_alt_text(nested.get('embed')))
        return ' '.join(chunks)

    # SDK model fallbacks
    images = getattr(embed, 'images', None)
    if isinstance(images, list):
        for image in images:
            alt = getattr(image, 'alt', None)
            if alt:
                chunks.append(str(alt))
    external = getattr(embed, 'external', None)
    if external is not None:
        for attr in ('title', 'description'):
            value = getattr(external, attr, None)
            if value:
                chunks.append(str(value))
    return ' '.join(chunks)


def combine_text(text: str = '', *, alt_text: str = '', langs: Iterable[str] | None = None) -> str:
    del langs  # reserved for future language-aware heuristics
    return _normalize(f'{text} {alt_text}')


def _soft_prior_ambiguous(
    author_did: str | None,
    soft_prior_dids: set[str],
    term: str,
) -> MatchResult | None:
    """Keep bare ambiguous places for authors with an earned soft prior."""
    if author_did and author_did in soft_prior_dids:
        return MatchResult(True, f'soft_prior_ambiguous:{term}')
    return None


def _match_local_event(haystack: str) -> MatchResult | None:
    """Keep regional events when a local venue appears with event phrasing."""
    if not _EVENT_CUE.search(haystack):
        return None
    match = _LOCAL_EVENT_VENUE.search(haystack)
    if not match:
        return None
    venue = re.sub(r'\s+', ' ', match.group(0).lower())
    return MatchResult(True, f'event_local_venue:{venue}')


def _classifier_keep(
    haystack: str,
    *,
    term: str | None,
    model: ClassifierModel | None,
) -> MatchResult | None:
    """Second-stage keep for ambiguous / event-near-miss leftovers."""
    decision = classify_candidate(
        haystack,
        term=term,
        has_event_cue=bool(_EVENT_CUE.search(haystack)),
        has_local_venue=bool(_LOCAL_EVENT_VENUE.search(haystack)),
        model=model,
    )
    if decision is None:
        return None
    return MatchResult(True, decision.reason)


def match_post(
    text: str,
    *,
    alt_text: str = '',
    author_did: str | None = None,
    author_handle: str | None = None,
    allowlist_dids: set[str] | None = None,
    allowlist_handles: set[str] | None = None,
    soft_prior_dids: set[str] | None = None,
    classifier_model: ClassifierModel | None = None,
) -> MatchResult:
    """Return whether a post belongs in the Capital Region feed.

    Decision order: allowlist → hard negative / strong regex floor → soft prior
    → ambiguous-case classifier → drop. ``classifier_model`` is for tests;
    production uses the checked-in weights in ``data/models/``.
    """
    allowlist_dids = allowlist_dids or set()
    allowlist_handles = {h.lower() for h in (allowlist_handles or set())}
    soft_prior_dids = soft_prior_dids or set()

    if author_did and author_did in allowlist_dids:
        return MatchResult(True, 'allowlist_did')
    if author_handle and author_handle.lower() in allowlist_handles:
        return MatchResult(True, 'allowlist_handle')

    haystack = combine_text(text, alt_text=alt_text)
    if not haystack:
        return MatchResult(False, 'empty')

    if _HARD_NEGATIVE.search(haystack):
        # Strong NY phrasing can still win over a hard negative only when it is
        # clearly local (e.g. quoting "New Albany" while talking about Albany, NY).
        if _STRONG_POSITIVE.search(haystack) and not _HARD_NEGATIVE_BLOCKS_STRONG.search(haystack):
            return MatchResult(True, 'strong_positive_over_negative')
        return MatchResult(False, 'hard_negative')

    if _STRONG_POSITIVE.search(haystack):
        return MatchResult(True, 'strong_positive')

    if _COLONIE_LOCAL.search(haystack):
        return MatchResult(True, 'colonie_local')

    event_match = _match_local_event(haystack)
    if event_match is not None:
        return event_match

    ambiguous_hits = _AMBIGUOUS_PLACE.findall(haystack)
    if ambiguous_hits:
        # Normalize to compare distinct place tokens (e.g. Albany + Troy).
        distinct = {re.sub(r'\s+', ' ', h.lower()) for h in ambiguous_hits}
        if len(distinct) >= 2:
            return MatchResult(True, 'multi_local_places')

        term = next(iter(distinct))
        # Bare "albany" is the noisiest token; require NY/local context.
        if term == 'albany':
            if _NY_CONTEXT.search(haystack):
                return MatchResult(True, 'albany_with_ny_context')
            if _STRONG_POSITIVE.search(haystack):
                return MatchResult(True, 'albany_with_local_cue')
            prior = _soft_prior_ambiguous(author_did, soft_prior_dids, term)
            if prior:
                return prior
            clf = _classifier_keep(haystack, term=term, model=classifier_model)
            return clf if clf else MatchResult(False, 'bare_albany')

        if term == 'colonie':
            # Avoid French "colonie" without local cues (handled above / hard neg).
            if _NY_CONTEXT.search(haystack) or _COLONIE_LOCAL.search(haystack):
                return MatchResult(True, 'colonie_with_context')
            prior = _soft_prior_ambiguous(author_did, soft_prior_dids, term)
            if prior:
                return prior
            clf = _classifier_keep(haystack, term=term, model=classifier_model)
            return clf if clf else MatchResult(False, 'bare_colonie')

        if _NY_CONTEXT.search(haystack) or _STRONG_POSITIVE.search(haystack):
            return MatchResult(True, f'ambiguous_with_context:{term}')
        prior = _soft_prior_ambiguous(author_did, soft_prior_dids, term)
        if prior:
            return prior
        clf = _classifier_keep(haystack, term=term, model=classifier_model)
        return clf if clf else MatchResult(False, f'ambiguous_no_context:{term}')

    clf = _classifier_keep(haystack, term=None, model=classifier_model)
    return clf if clf else MatchResult(False, 'no_match')
