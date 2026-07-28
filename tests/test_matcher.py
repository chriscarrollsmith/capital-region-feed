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


def test_hash_518_and_new_scotland_precision() -> None:
    """Bare #518 / New Scotland phrasing must not over-keep off-region posts."""
    assert match_post('Union Pacific West: #518 Per Metra realtime data.').matched is False
    assert match_post('Food truck Friday in the lot — come hang #518ny').matched is True
    assert match_post('Local 518 Music Fest this Saturday.').matched is True

    assert match_post('a new voice for a new Scotland radio schedule').matched is False
    assert (
        match_post(
            'Watch this',
            alt_text='Smoker with Blue White TShirt, New Scotland Shirt; music video',
        ).matched
        is False
    )
    assert match_post('Town board meets in New Scotland, NY on Tuesday.').matched is True

    # Upstate NY alone is broader than the Capital Region.
    assert (
        match_post(
            'So relieved they backed down.',
            alt_text='After backlash, Upstate NY school district pauses robot plan',
        ).matched
        is False
    )
    assert match_post('Albany in upstate is prettier in the fall').matched is True


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


def test_capital_regional_spanish_is_not_strong_positive() -> None:
    """Spanish 'capital regional' must not match capital region\\b."""
    result = match_post(
        'durante el estallido en cada capital regional estaba copada por pacos. '
        'En todo Chile había un número impresionante de pacos.'
    )
    assert result.matched is False


def test_canadian_capital_region_is_hard_negative() -> None:
    result = match_post(
        'The footnote is that the capital region does have a 70mm IMAX screen, '
        'but the museum of history is useless. #CanadianInnovation',
        alt_text=(
            'Q&A: How The Odyssey new IMAX cameras were designed in Canada | BetaKit '
            'IMAX Theatres global president says the Canadian film company wants to redesign.'
        ),
    )
    assert result.matched is False

    # Victoria BC / #yyj talk-radio "capital region" is not NY.
    bc = match_post('Zooming out for a birds-eye view of capital region politics. #yyj #BCpoli')
    assert bc.matched is False

    # NY Capital Region + Canada mention in passing can still keep.
    keep = match_post(
        'Capital Region exporters shipped goods to Canada through the Port of Albany (#AlbanyNY).'
    )
    assert keep.matched is True


def test_other_capital_region_phrases_are_hard_negatives() -> None:
    assert (
        match_post(
            'Sporty Group is hiring in Copenhagen, Capital Region Of Denmark, Denmark'
        ).matched
        is False
    )
    assert (
        match_post(
            'Grace Marion in our Capital Region Bureau and '
            'Jaylin Smith as our Delta Bureau reporter.'
        ).matched
        is False
    )


def test_new_brunswick_nova_scotia_not_multi_local() -> None:
    assert (
        match_post(
            'Police watchdog in New Brunswick and Nova Scotia is hiring an Indigenous investigator.'
        ).matched
        is False
    )
    # Town of Scotia / Brunswick NY still keep with NY context.
    assert match_post('Town board meets in Scotia, NY tonight.').matched is True
    assert match_post('Brunswick, NY planning board agenda posted.').matched is True


def test_wisconsin_east_troy_waterford_not_multi_local() -> None:
    assert (
        match_post(
            'Severe Thunderstorm Near E Troy Moving E. Locations Impacted Include '
            'E Troy, Wind Lake, Rochester, Waterford North, Troy Center. #wiwx'
        ).matched
        is False
    )
    # Cap Region Troy + Waterford still keep.
    assert match_post('Drive from Troy to Waterford for the farmers market.').matched is True


def test_galway_ireland_not_town_of_galway() -> None:
    assert (
        match_post(
            'Protecting human rights in New York and around the world as well.',
            alt_text="University of Galway's online publication Cois Coiribe.",
        ).matched
        is False
    )
    assert match_post('Town of Galway, NY board meets Thursday.').matched is True


def test_saratoga_race_course_and_spac_are_strong_positives() -> None:
    assert match_post('What to wear to Saratoga Race Course this August.').matched is True
    assert (
        match_post(
            'SPAC is where unforgettable moments happen.',
            alt_text='Saratoga Performing Arts Center (SPAC): A Music Hotspot',
        ).matched
        is True
    )
    assert (
        match_post(
            'Best view of morning works.',
            alt_text="Whitney Viewing Stand at Saratoga's Oklahoma Training Track",
        ).matched
        is True
    )


def test_handle_mentions_do_not_supply_albany_or_nyc_context() -> None:
    """@…albany… plus @….nyc must not keep a Portland/national share post."""
    result = match_post(
        'Share, share, share @portlanddsa.bsky.social @socialists.nyc '
        '#pdx #nokings @indivisible-oregon.bsky.social @nokings-albany.bsky.social'
    )
    assert result.matched is False

    # @eufuria.org remains a strong-positive path (full haystack).
    eufuria = match_post('Made it back home from @eufuria.org without issue.')
    assert eufuria.matched is True
    assert eufuria.reason == 'strong_positive'

    # Stripping mentions must not turn troy@email into a bare Troy place hit.
    assert (
        match_post(
            'Thanks to my friend in Mount Sinai, New York. Order today — email troy@example.com'
        ).matched
        is False
    )


def test_extract_alt_text_from_images() -> None:
    embed = {
        '$type': 'app.bsky.embed.images',
        'images': [{'alt': 'Sunset over the Hudson', 'image': {}}],
    }
    assert 'Hudson' in extract_alt_text(embed)


def test_extract_alt_text_caps_external_description() -> None:
    buried = (
        'Opening graphs about a Chicago concert. ' + ('word ' * 80) + 'Albany, New York tour stop.'
    )
    embed = {
        '$type': 'app.bsky.embed.external',
        'external': {
            'title': 'Review: Benson Boone in Chicago',
            'description': buried,
        },
    }
    alt = extract_alt_text(embed)
    assert 'Benson Boone' in alt
    assert 'Albany, New York' not in alt
    assert match_post('', alt_text=alt).matched is False


def test_collision_toponyms_need_ny_context() -> None:
    assert match_post('Lady Ravena rates for Edinburgh.').matched is False
    assert match_post('Mesonet station SNLW4 Sand Lake. #wywx').matched is False
    assert match_post('Near Green Island on the Jersey shore.').matched is False
    assert match_post('lush, green islands under a cloudy sky').matched is False
    assert match_post('#DelMar Race 10 projected odds').matched is False
    assert match_post('Post time from Del Mar this afternoon.').matched is False
    assert match_post('Now playing by The Lords of Altamont').matched is False
    assert match_post('#horseracing #saratoga #delmar #delmarthoroughbredclub').matched is False
    assert (
        match_post(
            'Susan Collins cash edge in Maine – ny times. Troy Jackson wins the nod.'
        ).matched
        is False
    )
    assert match_post('76 Delmar St Rochester, NY Single-family home').matched is False
    assert (
        match_post('Friend in Mount Sinai, New York. Email troy@example.com for prints.').matched
        is False
    )

    assert match_post('Meeting in Green Island, NY tonight.').matched is True
    assert match_post('Fire on Route 43 in Sand Lake, NY.').matched is True
    assert match_post('Altamont-based duo releases a new album.').matched is True
    assert match_post('See you at the Altamont Fair this August.').matched is True
    assert match_post('Day trip from Albany to Troy for the market.').matched is True


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
    assert result.reason in {'ambiguous_no_context:troy', 'hard_negative'}

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
