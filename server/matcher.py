"""Capital Region post matcher.

Designed to fix the main SkyFeed false positives:
- Albany Park (Chicago), New Albany (MS/IN), other U.S. Albanys
- French "colonie", NFL "JC Latham", Saratoga Springs UT
- Bare town names without NY / local context
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional


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


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


def extract_alt_text(embed: Optional[dict]) -> str:
    """Pull alt text from common Bluesky embed shapes (dict or SDK-like)."""
    if not embed:
        return ''

    chunks: list[str] = []

    if isinstance(embed, dict):
        images = embed.get('images') or []
        for image in images:
            if isinstance(image, dict) and image.get('alt'):
                chunks.append(str(image['alt']))
        external = embed.get('external')
        if isinstance(external, dict):
            for key in ('title', 'description'):
                if external.get(key):
                    chunks.append(str(external[key]))
        media = embed.get('media')
        if isinstance(media, dict):
            chunks.append(extract_alt_text(media))
        record = embed.get('record')
        if isinstance(record, dict):
            nested = record.get('record') or record.get('value') or {}
            if isinstance(nested, dict):
                if nested.get('text'):
                    chunks.append(str(nested['text']))
                chunks.append(extract_alt_text(nested.get('embed')))
        return ' '.join(chunks)

    # SDK model fallbacks
    images = getattr(embed, 'images', None) or []
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


def combine_text(text: str = '', *, alt_text: str = '', langs: Optional[Iterable[str]] = None) -> str:
    del langs  # reserved for future language-aware heuristics
    return _normalize(f'{text} {alt_text}')


def match_post(
    text: str,
    *,
    alt_text: str = '',
    author_did: Optional[str] = None,
    author_handle: Optional[str] = None,
    allowlist_dids: Optional[set[str]] = None,
    allowlist_handles: Optional[set[str]] = None,
) -> MatchResult:
    """Return whether a post belongs in the Capital Region feed."""
    allowlist_dids = allowlist_dids or set()
    allowlist_handles = {h.lower() for h in (allowlist_handles or set())}

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
        if _STRONG_POSITIVE.search(haystack) and not re.search(
            r'albany\s+park|new\s+albany|national\s+capital\s+region',
            haystack,
            re.IGNORECASE,
        ):
            return MatchResult(True, 'strong_positive_over_negative')
        return MatchResult(False, 'hard_negative')

    if _STRONG_POSITIVE.search(haystack):
        return MatchResult(True, 'strong_positive')

    if _COLONIE_LOCAL.search(haystack):
        return MatchResult(True, 'colonie_local')

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
            return MatchResult(False, 'bare_albany')

        if term == 'colonie':
            # Avoid French "colonie" without local cues (handled above / hard neg).
            if _NY_CONTEXT.search(haystack) or _COLONIE_LOCAL.search(haystack):
                return MatchResult(True, 'colonie_with_context')
            return MatchResult(False, 'bare_colonie')

        if _NY_CONTEXT.search(haystack) or _STRONG_POSITIVE.search(haystack):
            return MatchResult(True, f'ambiguous_with_context:{term}')
        return MatchResult(False, f'ambiguous_no_context:{term}')

    return MatchResult(False, 'no_match')
