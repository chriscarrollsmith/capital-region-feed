import json
from pathlib import Path
from typing import Any

import pytest
from server.allowlists import load_allowlist_dids, load_allowlist_handles
from server.matcher import extract_alt_text, match_post

ALL_CASES = json.loads(
    (Path(__file__).resolve().parents[1] / 'data' / 'eval_cases.json').read_text(encoding='utf-8')
)
# Gap cases (regression=false) are measured by scripts/eval_filter.py but not
# asserted here — they track known author/event recall misses until backlog
# items close them, without treating precision-gate drops as the only goal.
CASES = [c for c in ALL_CASES if c.get('regression', True)]

ALLOWLIST_HANDLES = load_allowlist_handles()
ALLOWLIST_DIDS = load_allowlist_dids()


@pytest.mark.parametrize('case', CASES, ids=[c['id'] for c in CASES])
def test_eval_case(case: dict[str, Any]) -> None:
    soft_prior_dids: set[str] = set()
    if case.get('soft_prior') and case.get('author_did'):
        soft_prior_dids.add(str(case['author_did']))
    langs = case.get('langs')
    result = match_post(
        case.get('text', ''),
        alt_text=case.get('alt_text', ''),
        langs=langs if isinstance(langs, list) else None,
        author_did=case.get('author_did'),
        author_handle=case.get('author_handle'),
        allowlist_dids=ALLOWLIST_DIDS,
        allowlist_handles=ALLOWLIST_HANDLES,
        soft_prior_dids=soft_prior_dids,
    )
    assert result.matched is bool(case['expected']), (
        f'{case["id"]}: expected={case["expected"]} got={result.matched} '
        f'reason={result.reason} note={case.get("note")}'
    )


def test_soft_prior_unlocks_bare_ambiguous_not_hard_negative() -> None:
    did = 'did:plc:softpriortest0000000000001'
    priors = {did}
    keep = match_post(
        'Dinner in Troy tonight.',
        author_did=did,
        soft_prior_dids=priors,
    )
    assert keep.matched is True
    assert keep.reason == 'soft_prior_ambiguous:troy'

    drop = match_post(
        'Dinner in Troy tonight.',
        author_did=did,
        soft_prior_dids=set(),
    )
    assert drop.matched is False
    assert drop.reason == 'ambiguous_no_context:troy'

    blocked = match_post(
        'Nice day in Albany Park.',
        author_did=did,
        soft_prior_dids=priors,
    )
    assert blocked.matched is False
    assert blocked.reason in {'hard_negative', 'entity_other:albany_park_chicago'}


def test_allowlist_did_matches_without_handle_or_placename() -> None:
    """Production Jetstream path: author DID only, no placename text."""
    assert ALLOWLIST_DIDS, 'allowlist_dids.txt must be populated for production recall'
    did = next(iter(sorted(ALLOWLIST_DIDS)))
    result = match_post(
        'Thanks for reading — more updates tomorrow.',
        author_did=did,
        allowlist_dids=ALLOWLIST_DIDS,
        allowlist_handles=ALLOWLIST_HANDLES,
    )
    assert result.matched is True
    assert result.reason == 'allowlist_did'


def test_event_local_venue_requires_cue_and_venue() -> None:
    keep = match_post('Tickets on sale for Saturday comedy night at Proctors. Doors at 7.')
    assert keep.matched is True
    assert keep.reason.startswith('event_local_venue:')

    no_cue = match_post('Proctors is a beautiful historic building downtown.')
    assert no_cue.matched is False

    off_region = match_post(
        'Tickets on sale for Saturday comedy night at The Fillmore. Doors at 7.'
    )
    assert off_region.matched is False

    bare_albany_event = match_post(
        "Don't miss the Albany Veterans Day Parade this Saturday downtown!"
    )
    assert bare_albany_event.matched is False


def test_named_cap_region_events_and_nyc_context() -> None:
    eufuria = match_post("I'll be suiting and hanging out at Eufuria in Albany this weekend.")
    assert eufuria.matched is True
    assert eufuria.reason == 'strong_positive'

    alive = match_post('Official Alive at 5 After Party in Albany with Lespecial.')
    assert alive.matched is True
    assert alive.reason == 'strong_positive'

    nyc_context = match_post("They've never went bankrupt in Albany: NYC almost did in the '70s.")
    assert nyc_context.matched is True
    assert nyc_context.reason == 'albany_with_ny_context'

    # Generic bare-Albany events still drop (precision gate).
    generic = match_post('See you in Albany this weekend!')
    assert generic.matched is False
    assert generic.reason == 'bare_albany'


def test_extract_alt_text_from_images() -> None:
    embed = {
        '$type': 'app.bsky.embed.images',
        'images': [{'alt': 'Sunset over the Hudson', 'image': {}}],
    }
    assert 'Hudson' in extract_alt_text(embed)


def test_new_york_times_masthead_is_not_ny_context_for_troy() -> None:
    """Person-name Troy + NYT masthead in link-card text must not keep."""
    embed = {
        '$type': 'app.bsky.embed.external',
        'external': {
            'title': 'Democrats pick new Senate candidate in Maine',
            'description': (
                'Troy Jackson Picked to Replace Platner as Democratic Nominee '
                'in Maine Senate Race  The New York Times'
            ),
        },
    }
    alt = extract_alt_text(embed)
    result = match_post(
        'Democrats pick new Senate candidate in Maine - Google News',
        alt_text=alt,
        langs=['en'],
    )
    assert result.matched is False
    assert result.reason == 'ambiguous_no_context:troy'

    keep = match_post('Dinner in Troy, New York tonight.', langs=['en'])
    assert keep.matched is True


def test_lang_gate_drops_french_colonie_not_english_local() -> None:
    fr = match_post(
        'La colonie organise une sortie demain matin.',
        langs=['fr'],
    )
    assert fr.matched is False
    assert fr.reason == 'lang_non_local:fr'

    # Bilingual EN+FR keeps the regex/entity path (Colonie NY police).
    bilingual = match_post(
        'Colonie Police responded to a crash on Central Ave this morning.',
        langs=['fr', 'en'],
    )
    assert bilingual.matched is True

    english = match_post(
        'Colonie Police responded to a crash on Central Ave this morning.',
        langs=['en'],
    )
    assert english.matched is True
