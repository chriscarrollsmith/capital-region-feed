"""Capital Region post matcher.

Rejects the main SkyFeed false positives:
- Albany Park (Chicago), New Albany (MS/IN), other U.S. Albanys
- French "colonie", NFL "JC Latham", Saratoga Springs UT
- Bare town names without NY / local context

Also aims for recall without placenames via author allowlists, soft author
priors (earned from repeated strong local matches), and event/venue cues
(local venue + upcoming-event phrasing). A checked-in gazetteer resolves
known place homographs to Capital Region vs other-region entities. Jetstream
``langs`` can drop French *colonie* without NY cues. Ambiguous leftovers and
event near-misses route to a small linear classifier. Precision stays strict
for ambiguous bare names unless a soft prior or classifier keep applies; see
README matching policy and BACKLOG.md.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from server.classifier import ClassifierModel, classify_candidate
from server.gazetteer import Gazetteer, default_gazetteer


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
      # Word-boundary after region/district — "capital regional" (ES/LatAm) is not NY.
      | (?:new\s+york(?:'s)?\s+)?capital\s+(?:region|district)\b
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
      | \bmenands\b
      | loudonville
      | slingerlands
      | voorheesville
      | schaghticoke
      | hoosick\s+falls
      | wynantskill
      | poestenkill
      | \bschodack\b
      | \bduanesburg\b
      | \bdelanson\b
      | east\s+greenbush
      | north\s+greenbush
      | mechanicville
      | burnt\s+hills
      | ballston\s+spa
      | clifton\s+park
      | averill\s+park
      | boght\s+corners
      | newtonville
      | \bcoeymans\b
      | helderberg
      # Cap Region Altamont cues only — bare "Altamont" stays ambiguous (CA festival / band names).
      | \baltamont-based\b
      | altamont\s+fair
      | altamont(?:\s*,)?\s*ny\b
      | empire\s+state\s+plaza
      | albany\s+capital\s+center
      | university\s+at\s+albany
      | \bualbany\b
      | suny\s+albany
      | \bi-?787\b
      | \bon\s+787\b
      | local\s*518
      # Prefer #518ny / #518area — bare #518 collides with train/jersey numbers.
      | \#518(?:ny|area)\b
      | reddit\.com/r/albany\b
      | \br/albany\b
      | saratoga\s+springs\s+police
      | saratoga\s+casino
      | saratoga\s+race\s+course
      # Common phrasing inserts "Springs" before Performing Arts Center.
      | saratoga(?:\s+springs)?\s+performing\s+arts\s+center
      # Saratoga Race Course training facility (tourism / race-day copy).
      | oklahoma\s+training\s+track
      | national\s+museum\s+of\s+racing
      # Horse-racing debut copy often omits "Race Course".
      | debut\s+at\s+saratoga\b
      # Hashtag #SPAC next to Saratoga (avoid bare SPAC / financial SPACs).
      | \#spac\b[\s\S]{0,200}\bsaratoga\b
      | \bsaratoga\b[\s\S]{0,200}\#spac\b
      | times\s+union\b
      # Town of New Scotland — not "a new Scotland" / "New Scotland Shirt".
      | new\s+scotland(?:\s*,?\s*ny\b|\s+town\b)
      # Distinctive Cap Region named events (not bare "Albany this weekend").
      | \beufuria\b
      | black\s+paw-?rade
      | alive\s+at\s+5\s+after\s+party
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Ambiguous place names that need NY / Capital Region context.
# Includes Cap Region micro-toponyms that collide with personal names,
# other U.S./world places, or common phrases (e.g. "green islands").
# ``troy`` ignores email local-parts (``troy@…``) and troy weight (oz / "10.8 troy").
_AMBIGUOUS_PLACE = re.compile(
    r"""
    (?:
        \balbany\b
      | (?<!\d\s)\btroy\b(?!@)(?!\s*(?:oz|ounces?|ozt|weight)\b)
      | \blatham\b
      | \bmalta\b
      # Town of Scotia — not the province inside "Nova Scotia".
      | (?<!nova\s)\bscotia\b
      | \bbethlehem\b
      # Town of Brunswick — not the province inside "New Brunswick".
      | (?<!new\s)\bbrunswick\b
      | \bgalway\b
      | \bstillwater\b
      | \bwaterford\b
      | \brotterdam\b
      | \bhalfmoon\b
      | \bcolonie\b
      | saratoga\s+springs
      | \bsaratoga\b
      | round\s+lake
      | \bdelmar\b
      | \bravena\b
      | \baltamont\b
      | sand\s+lake\b
      | green\s+island\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Collision micro-toponyms: do not unlock multi_local_places by pairing with
# another ambiguous token alone (e.g. #Saratoga + #DelMar racing hashtags).
_MULTI_LOCAL_EXCLUDED = frozenset(
    {
        'delmar',
        'ravena',
        'altamont',
        'sand lake',
        'green island',
    }
)

# "New York Times/Post/…" mastheads are national media names, not place context.
# Also reject abbreviated ``ny times`` (including NBSP variants via ``\s``).
_NY_CONTEXT = re.compile(
    r"""
    (?:
        \bny\b(?!\s*times\b)
      # Reject handle TLDs like @socialists.nyc (dot before nyc).
      | (?<!\.)\bnyc\b
      | new\s+york(?!\s+(?:
            times|post|daily\s+news|magazine|observer|herald|metro|sun
          )\b)
      | upstate
      | capital\s+(?:region|district)\b
      | \#ny\b
      | \#upstateny\b
      | upstate\s+ny\b
      | hudson\s+valley
      | \#albanyny\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# @handles can embed place-like tokens (nokings-albany, socialists.nyc) that
# must not unlock ambiguous-place keeps. Strong positives still see full text.
# Require a non-word char before @ so email local-parts (troy@example.com) stay
# intact for the ``troy(?!@)`` ambiguous-place guard.
_HANDLE_MENTION = re.compile(r'(?<![\w.])@[\w.-]+', flags=re.UNICODE)

# Hard negatives that always win over an otherwise-strong local phrase
# (e.g. "capital region" inside "capital region of Madrid").
# "New Albany Bus Station" is Albany NY's proposed terminal, not New Albany IN/MS.
_HARD_NEGATIVE_BLOCKS_STRONG = re.compile(
    r"""
    (?:
        albany\s+park
      | new\s+albany(?!\s+bus\s+(?:station|terminal|depot))
      | national\s+capital\s+region
      | brussels\s+capital\s+region
      | canadian\s+capital\s+region
      # Other-state / non-NY newsroom jargon (e.g. Jackson MS bureau).
      | capital\s+region\s+bureau\b
      | capital\s+region\s+of\s+(?:
            madrid|spain|belgium|brussels|paris|france|berlin|germany|
            tokyo|seoul|beijing|delhi|ottawa|canberra|rome|italy|
            amsterdam|vienna|warsaw|prague|lisbon|athens|dublin|
            canada|denmark|copenhagen|mississippi
          )\b
      | hauptstadtregion
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Canadian "capital region" cues (Ottawa national, Victoria BC / #yyj).
_CANADIAN_GEO_CUE = (
    r'\bcanada\b|\bcanadian\b|\bottawa\b|\#canadian\w*|'
    r'\#yyj\b|\#bcpoli\b|british\s+columbia|\blangford\b|'
    r'victoria(?:\s*,?\s*bc\b)'
)

# Ottawa / Canada / BC "capital region" co-occurring with Canadian cues (not NY).
_CANADIAN_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        canadian\s+capital\s+region
      | capital\s+region\s+of\s+(?:canada|ottawa)\b
      | capital\s+region\b[\s\S]{{0,160}}(?:{_CANADIAN_GEO_CUE})
      | (?:{_CANADIAN_GEO_CUE})[\s\S]{{0,160}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Maryland / DC-metro "capital region" (Prince George's, MoCo MD tourism, etc.).
# Avoid bare "montgomery county" — NYS Montgomery County appears in ALY weather.
_MD_DC_GEO_CUE = (
    # AppView cards often use a curly apostrophe in "George's".
    r"prince\s+george['\u2019]?s|\bmaryland\b|\#md(?:wx|politics|gov)\b|"
    r'washington(?:\s*,?\s*d\.?c\.?\b)|(?<![\w.])dc\s+metro\b|'
    r'\bdmv\b|silver\s+spring|\bbethesda\b|\brockville\b'
)

# MD/DC "capital region" co-occurring with Maryland / Prince George's cues (not NY).
_MD_DC_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        capital\s+region\b[\s\S]{{0,160}}(?:{_MD_DC_GEO_CUE})
      | (?:{_MD_DC_GEO_CUE})[\s\S]{{0,160}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Wisconsin East Troy / Waterford weather (and similar) must not unlock
# multi_local_places via Cap Region town-name collisions.
_WI_TROY_WATERFORD = re.compile(
    r"""
    (?:
        \#wiwx\b
      | \bwisconsin\b
      | \beast\s+troy\b
      | \be\s+troy\b
      | troy\s+center\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Galway Ireland (university / county / League of Ireland clubs).
_GALWAY_IRELAND = re.compile(
    r"""
    (?:
        university\s+of\s+galway
      | galway\s*,?\s*ireland\b
      | galway\s+united\b
      | galway\s+fc\b
      | league\s+of\s+ireland
      | county\s+mayo\b
      | cois\s+coiribe
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bethlehem, Pennsylvania (SteelStacks / Lehigh Valley) — not Town of Bethlehem NY.
_BETHLEHEM_PA = re.compile(
    r"""
    (?:
        bethlehem\s*,?\s*(?:pa|pennsylvania)\b
      | bethlehem[\s\S]{0,160}\b(?:pa|pennsylvania|philly|philadelphia)\b
      | \b(?:pa|pennsylvania|philly|philadelphia)\b[\s\S]{0,160}bethlehem
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
      # "New Albany Bus Station" is Albany NY's proposed terminal, not New Albany IN/MS.
      | new\s+albany(?!\s+bus\s+(?:station|terminal|depot))
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
      # California city / racetrack (not Delmar, NY).
      | \bdel\s+mar\b
      # Street name elsewhere (e.g. Rochester) — not the Town of Delmar.
      | delmar\s+st(?:reet)?\b
      # National politician, not Troy NY.
      | \btroy\s+jackson\b
      | national\s+capital\s+region
      | brussels\s+capital\s+region
      | canadian\s+capital\s+region
      | capital\s+region\s+bureau\b
      | capital\s+region\s+of\s+(?:
            madrid|spain|belgium|brussels|paris|france|berlin|germany|
            tokyo|seoul|beijing|delhi|ottawa|canberra|rome|italy|
            amsterdam|vienna|warsaw|prague|lisbon|athens|dublin|
            canada|denmark|copenhagen|mississippi
          )\b
      | hauptstadtregion
      | jc\s+latham
      | saratoga\s+springs\s*,\s*ut\b
      | saratoga\s+springs\s+ut\b
      | colonie\s+de\s+vacances
      | colonie\s+num[eé]rique
      | une\s+colonie
      | m[eê]me\s+colonie
      | university\s+of\s+galway
      | galway\s*,?\s*ireland\b
      | galway\s+united\b
      | galway\s+fc\b
      | league\s+of\s+ireland
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
        # Require "doors at/open" — bare "doors" matches doorway photography.
        doors(?:\s+at|\s+open)|tickets?|presale|save\s+the\s+date|join\s+us|
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
        # Schenectady's Proctors theatre — not the surname "Proctor".
        \bproctors\b
      | saratoga(?:\s+springs)?\s+performing\s+arts\s+center
      | \bat\s+spac\b
      | \bspac\s+(?:season|lawn|amphitheatre|amphitheater|presents)
      | national\s+museum\s+of\s+racing
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


# Link-card / quote bodies can be full articles; matching on the whole blob
# pulls buried tour-stop footnotes (e.g. "Albany, New York" in a Chicago review).
_EMBED_TEXT_MAX = 320


def _clip_embed_text(value: object, *, max_len: int = _EMBED_TEXT_MAX) -> str:
    text = str(value or '').strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip()


def extract_alt_text(embed: object | None) -> str:
    """Pull alt text from common Bluesky embed shapes (dict or SDK-like).

    External descriptions and quoted record text are length-capped so deep
    article body does not dominate matching. Titles and image alts are kept
    in full when short, and clipped at the same cap when longer.
    """
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
                    chunks.append(_clip_embed_text(alt))
        external = embed.get('external')
        if isinstance(external, dict):
            title = external.get('title')
            if title:
                chunks.append(_clip_embed_text(title))
            description = external.get('description')
            if description:
                chunks.append(_clip_embed_text(description))
        media = embed.get('media')
        if isinstance(media, dict):
            chunks.append(extract_alt_text(media))
        record = embed.get('record')
        if isinstance(record, dict):
            nested = record.get('record') or record.get('value') or {}
            if isinstance(nested, dict):
                text = nested.get('text')
                if text:
                    chunks.append(_clip_embed_text(text))
                chunks.append(extract_alt_text(nested.get('embed')))
        return ' '.join(chunk for chunk in chunks if chunk)

    # SDK model fallbacks
    images = getattr(embed, 'images', None)
    if isinstance(images, list):
        for image in images:
            alt = getattr(image, 'alt', None)
            if alt:
                chunks.append(_clip_embed_text(alt))
    external = getattr(embed, 'external', None)
    if external is not None:
        title = getattr(external, 'title', None)
        if title:
            chunks.append(_clip_embed_text(title))
        description = getattr(external, 'description', None)
        if description:
            chunks.append(_clip_embed_text(description))
    return ' '.join(chunk for chunk in chunks if chunk)


def combine_text(text: str = '', *, alt_text: str = '', langs: Iterable[str] | None = None) -> str:
    del langs  # language tags are applied in match_post, not folded into text
    return _normalize(f'{text} {alt_text}')


def _normalize_langs(langs: Iterable[str] | None) -> list[str]:
    if not langs:
        return []
    out: list[str] = []
    for lang in langs:
        token = str(lang or '').strip().lower().replace('_', '-')
        if token:
            out.append(token)
    return out


def _has_lang_prefix(langs: list[str], prefix: str) -> bool:
    return any(lang == prefix or lang.startswith(f'{prefix}-') for lang in langs)


_ALBANY_COUNTY_WY = re.compile(
    r"""
    (?:
        albany\s+county(?:\s*,)?\s*(?:wy|wyoming)\b
      # NWS bots tag state as [WY]; Laramie Valley is WY-only context.
      | albany\s+county[\s\S]{0,240}(?:\bwyoming\b|\#wy\b|\[wy\]|\blaramie\b)
      | (?:\#wy\b|\bwyoming\b|\[wy\]|\blaramie\b)[\s\S]{0,240}albany\s+county
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _albany_county_wy_conflict(haystack: str) -> bool:
    """True when Albany County cues point at Wyoming, not NY."""
    if not _ALBANY_COUNTY_WY.search(haystack):
        return False
    # Explicit NY state context wins (comparison posts, etc.).
    if re.search(r'\b(?:ny|new\s+york)\b', haystack, flags=re.IGNORECASE):
        return False
    return True


def _ny_capital_region_context(haystack: str) -> bool:
    """True when haystack has explicit NY Cap Region anchors."""
    return (
        re.search(
            r"""
            (?:
                \b(?:ny|new\s+york)\b
              | hudson\s+valley
              | \#albanyny\b
              | albany\s*,?\s*(?:ny|new\s+york)
              | schenectady
              | (?<!\d\s)\btroy\b(?!@)(?!\s*(?:oz|ounces?|ozt|weight)\b)
            )
            """,
            haystack,
            flags=re.IGNORECASE | re.VERBOSE,
        )
        is not None
    )


def _canadian_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to Ottawa/Canada/BC, not NY."""
    if not _CANADIAN_CAPITAL_REGION.search(haystack):
        return False
    # Explicit NY Cap Region context wins over Canadian co-mentions.
    return not _ny_capital_region_context(haystack)


def _md_dc_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to MD/DC metro, not NY."""
    if not _MD_DC_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _wi_troy_waterford_conflict(haystack: str, distinct: set[str]) -> bool:
    """True when Troy+Waterford (etc.) are Wisconsin East Troy weather, not NY."""
    if not ({'troy', 'waterford'} & distinct):
        return False
    if not _WI_TROY_WATERFORD.search(haystack):
        return False
    if _NY_CONTEXT.search(haystack) or _STRONG_POSITIVE.search(haystack):
        return False
    return True


def _galway_ireland_conflict(haystack: str) -> bool:
    """True when Galway refers to Ireland, not the Town of Galway NY."""
    if not re.search(r'\bgalway\b', haystack, flags=re.IGNORECASE):
        return False
    if not _GALWAY_IRELAND.search(haystack):
        return False
    if re.search(r'galway\s*,?\s*ny\b|town\s+of\s+galway', haystack, flags=re.IGNORECASE):
        return False
    return True


def _bethlehem_pa_conflict(haystack: str) -> bool:
    """True when Bethlehem refers to Pennsylvania, not Town of Bethlehem NY."""
    if not re.search(r'\bbethlehem\b', haystack, flags=re.IGNORECASE):
        return False
    if not _BETHLEHEM_PA.search(haystack):
        return False
    if re.search(
        r'bethlehem\s*,?\s*ny\b|town\s+of\s+bethlehem',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _haystack_without_handles(haystack: str) -> str:
    """Strip @mentions so handle tokens cannot supply place/NY context."""
    return _HANDLE_MENTION.sub(' ', haystack)


def _lang_non_local_colonie(haystack: str, langs: list[str]) -> MatchResult | None:
    """Drop French *colonie* when langs say fr and there is no NY/local cue."""
    if not _has_lang_prefix(langs, 'fr'):
        return None
    # Bilingual posts that also declare English keep the regex path.
    if _has_lang_prefix(langs, 'en'):
        return None
    if not re.search(r'\bcolonie\b', haystack, flags=re.IGNORECASE):
        return None
    if (
        _COLONIE_LOCAL.search(haystack)
        or _NY_CONTEXT.search(haystack)
        or _STRONG_POSITIVE.search(haystack)
    ):
        return None
    return MatchResult(False, 'lang_non_local:fr')


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
    langs: Iterable[str] | None = None,
    author_did: str | None = None,
    author_handle: str | None = None,
    allowlist_dids: set[str] | None = None,
    allowlist_handles: set[str] | None = None,
    soft_prior_dids: set[str] | None = None,
    classifier_model: ClassifierModel | None = None,
    gazetteer: Gazetteer | None = None,
) -> MatchResult:
    """Return whether a post belongs in the Capital Region feed.

    Decision order: allowlist → language gate → gazetteer other-region → hard
    negative / gazetteer local / strong regex floor → soft prior →
    ambiguous-case classifier → drop. ``classifier_model`` / ``gazetteer`` are
    for tests; production uses checked-in weights and ``data/gazetteer/``.
    """
    allowlist_dids = allowlist_dids or set()
    allowlist_handles = {h.lower() for h in (allowlist_handles or set())}
    soft_prior_dids = soft_prior_dids or set()
    places = gazetteer if gazetteer is not None else default_gazetteer()
    lang_tags = _normalize_langs(langs)

    if author_did and author_did in allowlist_dids:
        return MatchResult(True, 'allowlist_did')
    if author_handle and author_handle.lower() in allowlist_handles:
        return MatchResult(True, 'allowlist_handle')

    haystack = combine_text(text, alt_text=alt_text, langs=lang_tags)
    if not haystack:
        return MatchResult(False, 'empty')

    lang_drop = _lang_non_local_colonie(haystack, lang_tags)
    if lang_drop is not None:
        return lang_drop

    entity = places.lookup(haystack)
    if entity is not None and entity.region == 'other':
        return MatchResult(False, f'entity_other:{entity.entity_id}')

    if _HARD_NEGATIVE.search(haystack):
        # Strong NY phrasing can still win over a hard negative only when it is
        # clearly local (e.g. quoting "New Albany" while talking about Albany, NY).
        if _STRONG_POSITIVE.search(haystack) and not _HARD_NEGATIVE_BLOCKS_STRONG.search(haystack):
            return MatchResult(True, 'strong_positive_over_negative')
        return MatchResult(False, 'hard_negative')

    if entity is not None and entity.region == 'capital_ny':
        if entity.entity_id == 'albany_county_ny' and _albany_county_wy_conflict(haystack):
            return MatchResult(False, 'entity_other:albany_county_wy')
        return MatchResult(True, f'entity_local:{entity.entity_id}')

    if _STRONG_POSITIVE.search(haystack):
        # Bare "Albany County" is also a strong token; still drop WY-tagged posts.
        if _albany_county_wy_conflict(haystack):
            return MatchResult(False, 'entity_other:albany_county_wy')
        if _canadian_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:canadian_capital_region')
        if _md_dc_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:md_dc_capital_region')
        return MatchResult(True, 'strong_positive')

    if _COLONIE_LOCAL.search(haystack):
        return MatchResult(True, 'colonie_local')

    event_match = _match_local_event(haystack)
    if event_match is not None:
        return event_match

    # Place / NY-context from body text only — not @handle tokens.
    place_haystack = _haystack_without_handles(haystack)
    ambiguous_hits = _AMBIGUOUS_PLACE.findall(place_haystack)
    if ambiguous_hits:
        # Normalize to compare distinct place tokens (e.g. Albany + Troy).
        distinct = {re.sub(r'\s+', ' ', h.lower()) for h in ambiguous_hits}
        # Ireland Galway (+ Waterford FC scorelines) must not unlock multi-local.
        if _galway_ireland_conflict(haystack) and distinct <= {'galway', 'waterford'}:
            return MatchResult(False, 'hard_negative:galway_ireland')
        multi_eligible = {name for name in distinct if name not in _MULTI_LOCAL_EXCLUDED}
        if len(multi_eligible) >= 2:
            if _wi_troy_waterford_conflict(haystack, multi_eligible):
                return MatchResult(False, 'hard_negative:wi_troy_waterford')
            return MatchResult(True, 'multi_local_places')

        # Prefer a non-collision token when several ambiguous names appear but
        # multi-local did not fire (e.g. Saratoga + DelMar racing tags).
        term = sorted(distinct, key=lambda name: (name in _MULTI_LOCAL_EXCLUDED, name))[0]
        # Bare "albany" is the noisiest token; require NY/local context.
        if term == 'albany':
            if _NY_CONTEXT.search(place_haystack):
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
            if _NY_CONTEXT.search(place_haystack) or _COLONIE_LOCAL.search(haystack):
                return MatchResult(True, 'colonie_with_context')
            prior = _soft_prior_ambiguous(author_did, soft_prior_dids, term)
            if prior:
                return prior
            clf = _classifier_keep(haystack, term=term, model=classifier_model)
            return clf if clf else MatchResult(False, 'bare_colonie')

        if term == 'galway' and _galway_ireland_conflict(haystack):
            return MatchResult(False, 'hard_negative:galway_ireland')

        if term == 'bethlehem' and _bethlehem_pa_conflict(haystack):
            return MatchResult(False, 'hard_negative:bethlehem_pa')

        if _NY_CONTEXT.search(place_haystack) or _STRONG_POSITIVE.search(haystack):
            return MatchResult(True, f'ambiguous_with_context:{term}')
        prior = _soft_prior_ambiguous(author_did, soft_prior_dids, term)
        if prior:
            return prior
        clf = _classifier_keep(haystack, term=term, model=classifier_model)
        return clf if clf else MatchResult(False, f'ambiguous_no_context:{term}')

    clf = _classifier_keep(haystack, term=None, model=classifier_model)
    return clf if clf else MatchResult(False, 'no_match')
