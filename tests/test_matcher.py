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

    # Greater Victoria / Livable CRD / Times Colonist "capital region" is not NY.
    victoria = match_post(
        'Mayoral candidates should answer transit surveys.',
        alt_text=(
            'Coalition of transit advocates to grade Greater Victoria candidates. '
            'Livable CRD plans to survey capital region candidates on housing.'
        ),
    )
    assert victoria.matched is False
    assert victoria.reason == 'hard_negative:canadian_capital_region'

    # RCAF Snowbirds flyovers / Times Colonist author handle.
    snowbirds = match_post(
        '',
        alt_text=(
            'Parkland Secondary grad Capt. Brendan Pellow will return to the capital '
            'region with the Snowbirds team on Monday.'
        ),
    )
    assert snowbirds.matched is False
    assert snowbirds.reason == 'hard_negative:canadian_capital_region'

    tc_handle = match_post(
        'Flyover returns to the capital region on Monday.',
        author_handle='timescolonist.bsky.social',
    )
    assert tc_handle.matched is False

    # NY Capital Region + Canada mention in passing can still keep.
    keep = match_post(
        'Capital Region exporters shipped goods to Canada through the Port of Albany (#AlbanyNY).'
    )
    assert keep.matched is True


def test_md_dc_capital_region_is_hard_negative() -> None:
    md = match_post(
        '7 things to do in the capital region, from Karol G to wine and jazz',
        alt_text=(
            "Things to do across Montgomery and Prince George's counties this week "
            'include festivals showcasing books, wine and jazz.'
        ),
    )
    assert md.matched is False
    assert md.reason == 'hard_negative:md_dc_capital_region'

    # Curly apostrophe as seen in AppView external descriptions.
    curly = match_post(
        '7 things to do in the capital region, from Karol G to wine and jazz',
        alt_text=(
            'Things to do across Montgomery and Prince George\u2019s counties this week '
            'include festivals showcasing books, wine and jazz.'
        ),
    )
    assert curly.matched is False

    # NY Capital Region keeps even if Maryland is mentioned in passing.
    keep = match_post(
        'Capital Region students visited museums in Maryland before returning to #AlbanyNY.'
    )
    assert keep.matched is True


def test_louisiana_capital_region_is_hard_negative() -> None:
    la = match_post(
        'Governments across the Capital Region, including East Baton Rouge Parish, '
        'are preparing shelters ahead of the storm.'
    )
    assert la.matched is False
    assert la.reason == 'hard_negative:louisiana_capital_region'

    # WBRZ weather copy often omits Baton Rouge; gate on author handle.
    wbrz = match_post(
        'Storms are expected to move through the Capital Region overnight, '
        'but most of the rain should be out by sunrise.',
        author_handle='wbrz-mirror.bsky.social',
    )
    assert wbrz.matched is False
    assert wbrz.reason == 'hard_negative:louisiana_capital_region'

    keep = match_post(
        'Capital Region exporters shipped goods to Louisiana through the Port of Albany '
        '(#AlbanyNY).'
    )
    assert keep.matched is True


def test_pennsylvania_capital_region_is_hard_negative() -> None:
    pa = match_post(
        'Capital Region Water replaces manholes in Harrisburg through March.',
    )
    assert pa.matched is False

    forum = match_post(
        'Spent Sunday at the Pennsylvania Capital Region Stands Up forum on judicial retention.'
    )
    assert forum.matched is False

    keep = match_post('Capital Region students visited Harrisburg before returning to #AlbanyNY.')
    assert keep.matched is True


def test_malta_europe_ais_not_multi_local() -> None:
    ais = match_post(
        'VesselAlert\nName: PRYSMIAN MARCO POLO\nMMSI: 249023000\n'
        'Callsign: 9HA6070\nType: Other\nFlag: Malta\n'
        'Dest.: ROTTERDAM\nSpeed: 6.9 kts'
    )
    assert ais.matched is False
    assert ais.reason == 'hard_negative:malta_europe'

    # Town of Malta / Rotterdam NY still keep with NY context.
    assert match_post('Malta, NY town board meets about the solar farm tonight.').matched is True
    assert match_post('Rotterdam, NY fire department open house this weekend.').matched is True


def test_troy_weight_is_not_troy_ny() -> None:
    assert (
        match_post(
            'Silver ecclesiastical chalice with Nelson & Nelson, NYC',
            alt_text='Very good condition. Weight: 10.8 troy. French and Dutch import marks.',
        ).matched
        is False
    )
    assert match_post('Antique spoon listed at 2 troy ounces — shipping from NYC.').matched is False
    # Real Troy NY + NYC still keeps.
    assert match_post('Heading from Troy to NYC for the weekend show.').matched is True


def test_new_albany_bus_station_does_not_block_empire_state_plaza() -> None:
    result = match_post(
        "Could the Empire State Plaza play host to Albany's next bus terminal?",
        alt_text="State Says 'Not So Fast' On New Albany Bus Station - Streetsblog Empire State",
    )
    assert result.matched is True
    # True New Albany IN/MS still drops.
    assert match_post('Weekend plans in New Albany, Indiana.').matched is False


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
    # Tourism itineraries: Ireland + Galway + NYC must not unlock Town of Galway NY.
    tourism = match_post(
        'Ireland is one of the safest places you will visit. NYC to Dublin is under '
        '6 hours. Start with Dingle, Galway, Kinsale. #VisitIreland #WildAtlanticWay'
    )
    assert tourism.matched is False
    assert tourism.reason == 'hard_negative:galway_ireland'
    assert match_post('Town of Galway, NY board meets Thursday.').matched is True


def test_clifton_park_uk_cricket_not_ny() -> None:
    cricket = match_post(
        "Yorkshire make light work of Durham on Clifton Park's grand occasion",
        alt_text=(
            "Yorkshire make light work of Durham on Clifton Park's grand occasion | The Cricketer"
        ),
    )
    assert cricket.matched is False
    assert cricket.reason == 'hard_negative:clifton_park_uk'

    # Rotherham Show / .gov.uk Clifton Park is England, not NY.
    rotherham = match_post(
        'The Rotherham Show returns to Clifton Park, with one of its most vibrant programmes yet. '
        'www.rotherham.gov.uk/rotherham-show'
    )
    assert rotherham.matched is False
    assert rotherham.reason == 'hard_negative:clifton_park_uk'

    assert match_post('Water main break on Carlton Road in Clifton Park, NY.').matched is True


def test_california_capital_region_is_hard_negative() -> None:
    sac = match_post(
        'In recent years, numerous social clubs geared toward women in the capital region '
        'have sprung up.',
        alt_text='Sacramento meet-up clubs help women form community in adulthood.',
    )
    assert sac.matched is False
    assert sac.reason == 'hard_negative:california_capital_region'

    # SacBee cards may omit Sacramento in short body copy.
    handle = match_post(
        'Are social clubs in the capital region creating true friendships?',
        author_handle='sacbee.com',
    )
    assert handle.matched is False
    assert handle.reason == 'hard_negative:california_capital_region'

    keep = match_post('Capital Region students visited Sacramento before returning to #AlbanyNY.')
    assert keep.matched is True


def test_korea_capital_region_is_hard_negative() -> None:
    seoul = match_post(
        "Seoul and parts of Gyeonggi Province came under the capital region's first "
        'Heat Wave Emergency Warning on Monday, as record-breaking heat in southeastern '
        'South Korea spread westward.'
    )
    assert seoul.matched is False
    assert seoul.reason == 'hard_negative:korea_capital_region'

    keep = match_post(
        'Capital Region exporters shipped goods to Seoul through the Port of Albany (#AlbanyNY).'
    )
    assert keep.matched is True


def test_saratoga_venue_cues_without_ny() -> None:
    """Caffe Lena / High Rock Park imply Saratoga Springs NY even without ', NY'."""
    caffe = match_post('Got to see Rory Block over the weekend at Caffe Lena in Saratoga Springs.')
    assert caffe.matched is True

    high_rock = match_post(
        'We are popping up again in SARATOGA SPRINGS in September at High Rock Park Pavilions!'
    )
    assert high_rock.matched is True


def test_ny_dot_abbrev_and_nys_are_ny_context() -> None:
    """Wire datelines use N.Y.; locals often write NYS for New York State."""
    assert match_post('SARATOGA SPRINGS, N.Y. — Local Knowledge wins the Amsterdam Stakes.').matched
    assert match_post('ALBANY, N.Y. (WRGB) — Lawmakers met in the state Capitol.').matched
    assert match_post('MALTA, N.Y. (WNYT) – Deputies tracked a missing person.').matched
    assert match_post(
        'I moved all the way to Albany and I am still far from Rochester, best city in NYS.'
    ).matched


def test_troy_hyphenated_name_not_troy_ny() -> None:
    """Hyphenated troy- names/domains must not unlock via New York art titles."""
    assert (
        match_post(
            'Snow in New York print sold. See troy-caperton.pixels.com — email troy@example.com'
        ).matched
        is False
    )
    assert match_post('Heading from Troy to NYC for the weekend show.').matched is True


def test_galway_united_waterford_not_multi_local() -> None:
    ireland = 'Galway United 0-0 Waterford\n\nGalway United:\n-\n\nWaterford:\n-'
    assert match_post(ireland).matched is False
    assert match_post('Drive from Galway to Waterford for the farmers market.').matched is True


def test_bethlehem_pa_not_town_of_bethlehem_ny() -> None:
    assert (
        match_post(
            'Two shows in PA and one in NYC.\n\nFri 8/21 - Bethlehem, PA\n'
            'Sat 8/22 - Philly, PA\nSun 8/23 - Brooklyn, NY'
        ).matched
        is False
    )
    assert match_post('Town of Bethlehem, NY board meeting tonight.').matched is True


def test_proctors_theatre_not_surname_proctor() -> None:
    assert (
        match_post(
            'Join us for Deep Dives on space exploration this September.',
            alt_text='A conversation with Mary Robinette Kowal and Dr. Sian Proctor.',
        ).matched
        is False
    )
    assert match_post('Comedy night at Proctors this Saturday — tickets on sale.').matched is True


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
            'this week’s grateful deadcast visits saratoga springs ’85',
            alt_text='Grateful Dead at Saratoga Springs Performing Arts Center',
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
    assert (
        match_post(
            "This weekend marks the band's 50th show at #SPAC. "
            'Thanks for coming to Saratoga year after year.'
        ).matched
        is True
    )
    assert (
        match_post(
            'Casino Night at the National Museum of Racing and Hall of Fame in Saratoga Springs.'
        ).matched
        is True
    )
    assert match_post('The filly is expected to make her debut at Saratoga.').matched is True


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
