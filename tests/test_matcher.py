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

    # DC Snipers exhibit copy — "DC" alone is not a geo cue, but "DC Snipers" is.
    snipers = match_post(
        'It’s been nearly 24 years since the DC Snipers terrorized the capital region. '
        'Without Warning, an exhibit at the National Law Enforcement Museum.'
    )
    assert snipers.matched is False
    assert snipers.reason == 'hard_negative:md_dc_capital_region'

    # MoCo community media often omits Maryland in the body.
    mymc = match_post(
        'It’s been nearly 24 years since snipers terrorized the capital region.',
        author_handle='mymcmedia.bsky.social',
    )
    assert mymc.matched is False
    assert mymc.reason == 'hard_negative:md_dc_capital_region'

    # NY Capital Region keeps even if Maryland is mentioned in passing.
    keep = match_post(
        'Capital Region students visited museums in Maryland before returning to #AlbanyNY.'
    )
    assert keep.matched is True


def test_png_national_capital_district_is_hard_negative() -> None:
    png = match_post(
        'The National Capital District Provincial Health Authority (NCDPHA) has submitted '
        'its 2027 Annual Budget for Port Moresby and Motu Koitabu areas.'
    )
    assert png.matched is False

    keep = match_post(
        'Capital District aid groups sent supplies after floods in Port Moresby (#AlbanyNY).'
    )
    assert keep.matched is True


def test_troy_avenue_brooklyn_not_troy_ny() -> None:
    crown = match_post(
        'A major new development planned for the corner of Troy Avenue and East New York '
        'Avenue is prompting questions among Crown Heights residents.'
    )
    assert crown.matched is False
    assert crown.reason == 'hard_negative'

    keep = match_post('Architecture walking tour on River Street in Troy, NY.')
    assert keep.matched is True


def test_bay_area_albany_not_albany_ny() -> None:
    bay = match_post(
        'like obviously piedmont and atherton and albany should not exist. but we should '
        'seriously consider large-scale consolidation in the bay area like new york city '
        'did in 1898'
    )
    assert bay.matched is False
    assert bay.reason == 'hard_negative:albany_bay_area'

    keep = match_post('#AlbanyNY officials toured Bay Area transit projects before returning home.')
    assert keep.matched is True


def test_bay_area_albany_saratoga_multi_local_not_ny() -> None:
    bay = match_post(
        'various city/county consolidation concepts',
        alt_text=(
            'cupertino+saratoga+monte sereno+los gatos\n'
            'albany+berkeley+emeryville\noakland+piedmont'
        ),
    )
    assert bay.matched is False
    assert bay.reason == 'hard_negative:albany_bay_area'

    keep = match_post('Drive from Albany to Saratoga Springs for the races.')
    assert keep.matched is True


def test_stillwater_film_not_stillwater_ny() -> None:
    film = match_post(
        '',
        alt_text=(
            "Tom McCarthy's 'A Statement' to World Premiere at 64th New York Film Festival. "
            'It\'s been five years since "Stillwater," starring Matt Damon.'
        ),
    )
    assert film.matched is False
    assert film.reason == 'hard_negative:stillwater_film'

    keep = match_post('Stillwater, NY town board meets Tuesday.')
    assert keep.matched is True


def test_stillwater_road_lewis_co_not_stillwater_ny() -> None:
    road = match_post(
        '11 ESE Croghan [Lewis Co, NY] 911 Call Center reports Tstm Wnd Dmg — '
        'Multiple trees on wires on Stillwater Road.',
        author_handle='buf.nws-bot.us',
    )
    assert road.matched is False
    assert road.reason == 'hard_negative:stillwater_road'

    keep = match_post('Farmers market returns to Stillwater, NY this Saturday.')
    assert keep.matched is True


def test_troy_michigan_with_ny_context_not_troy_ny() -> None:
    odyssey = match_post(
        'a couple’s odyssey from Troy Michigan to Ithaca New York and their separate '
        'adventures after one of them seeks asylum in Canada once they hit Sarnia.'
    )
    assert odyssey.matched is False

    detroit = match_post(
        'New York City • August 25–28\nLas Vegas • September 10–12\nDetroit/Troy • September 28–30'
    )
    assert detroit.matched is False

    keep = match_post('Concert in Troy, NY tonight at the Music Hall.')
    assert keep.matched is True


def test_indiana_albany_saratoga_weather_not_multi_local() -> None:
    indiana = match_post(
        'Severe Thunderstorm Near Albany or 7 Miles NE of Muncie Moving SE At 50 MPH. '
        'Locations Impacted Include Muncie, Winchester, Union City, Albany, Eaton, '
        'Parker City, Farmland, Lynn, Selma, Ridgeville, Saratoga, Modoc & '
        'Ball State University. #inwx Details'
    )
    assert indiana.matched is False
    assert indiana.reason == 'hard_negative:indiana_albany_saratoga'

    keep = match_post('Drive from Albany to Saratoga Springs for the races.')
    assert keep.matched is True


def test_van_helderbergh_not_helderberg_escarpment() -> None:
    sculptor = match_post(
        'Pulpit of the Small Beguinage of Ghent, executed by Jan Baptist van Helderbergh, '
        '1731–1732.'
    )
    assert sculptor.matched is False

    keep = match_post('Hike the Helderberg Escarpment this weekend near #AlbanyNY.')
    assert keep.matched is True


def test_saratoga_amtrak_spac_jazz_without_ny_token() -> None:
    assert match_post('UNSAFE DE-BOARDING by Amtrak in Saratoga Springs').matched is True
    assert match_post('Show at SPAC in Saratoga Springs this Friday').matched is True
    assert (
        match_post(
            'From the Saratoga Jazz Festival to the Toying Around Block Party in Johnstown'
        ).matched
        is True
    )


def test_saratoga_avenue_subway_not_saratoga_springs() -> None:
    mta = match_post(
        'Uptown 3 trains are running with delays after we requested NYPD assistance for an '
        'unauthorized person on the tracks at Saratoga Av. #nyc #mta #subway'
    )
    assert mta.matched is False
    assert mta.reason == 'hard_negative'

    keep = match_post('March On Washington meetup in Saratoga New York this Thursday.')
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

    # wbrznews2 (prefix, not a bare \bwbrz\b token) and Ascension Parish towns.
    wbrznews = match_post(
        'First day of qualifying ends for US House, local races across Capital Region.',
        author_handle='wbrznews2.bsky.social',
    )
    assert wbrznews.matched is False
    assert wbrznews.reason == 'hard_negative:louisiana_capital_region'

    ascension = match_post(
        'From St. Amant to Sorrento to Prairieville, we deliver portable storage across '
        'Ascension and the wider Capital Region.'
    )
    assert ascension.matched is False
    assert ascension.reason == 'hard_negative:louisiana_capital_region'

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


def test_bogota_capital_district_is_hard_negative() -> None:
    bogota = match_post(
        '98-0002 Landed near Bogota, Bogotá Capital District, Colombia. Apx. flt. time 0 min.',
        author_handle='usairforcevip.bsky.social',
    )
    assert bogota.matched is False

    keep = match_post(
        'Capital District aid groups sent supplies after floods in Colombia (#AlbanyNY).'
    )
    assert keep.matched is True


def test_maryland_banner_capital_region_is_hard_negative() -> None:
    olney = match_post(
        '7 things to do in the capital region, including a free day of theater.',
        alt_text=(
            'Weekend guide for the capital region, including Olney Theatre Center '
            'and a water lantern festival in National Harbor.'
        ),
        author_handle='bannerpgcounty.bsky.social',
    )
    assert olney.matched is False
    assert olney.reason == 'hard_negative:md_dc_capital_region'

    handle_only = match_post(
        '7 things to do in the capital region this weekend.',
        author_handle='bannermoco.bsky.social',
    )
    assert handle_only.matched is False
    assert handle_only.reason == 'hard_negative:md_dc_capital_region'

    keep = match_post('Capital Region exporters shipped goods through Maryland to #AlbanyNY.')
    assert keep.matched is True


def test_cfax_canadian_capital_region_long_window() -> None:
    # #yyj sits ~196 chars after "capital region" — beyond the old 160-char window.
    cfax = match_post(
        'Explore debates in the capital region with local callers weighing deer '
        'immunocontraception efforts, downtown street disorder, the role of advocacy '
        'groups in municipal politics and more. #yyj #BCpoli',
        author_handle='cfax1070.bsky.social',
    )
    assert cfax.matched is False
    assert cfax.reason == 'hard_negative:canadian_capital_region'

    handle_only = match_post(
        'Callers weigh in on deer control across the capital region this hour.',
        author_handle='cfax1070.bsky.social',
    )
    assert handle_only.matched is False
    assert handle_only.reason == 'hard_negative:canadian_capital_region'


def test_virginia_capital_region_is_hard_negative() -> None:
    va = match_post(
        'She notes the harm this merger can have for people across the Capital region.',
        alt_text='Opinion | Abigail Spanberger: Why I’m intervening in the Dominion-NextEra merger',
    )
    assert va.matched is False
    assert va.reason == 'hard_negative:virginia_capital_region'

    keep = match_post('Capital Region students visited Richmond before returning to #AlbanyNY.')
    assert keep.matched is True


def test_virginia_capital_district_hs_athletics_is_hard_negative() -> None:
    vhs = match_post(
        'Breaking down the Capital District: Season previews for Armstrong, Atlee, '
        'Varina, Highland Springs, Patrick Henry (Ashland), Mechanicsville, Henrico '
        'and Hanover.'
    )
    assert vhs.matched is False
    assert vhs.reason == 'hard_negative:virginia_capital_district'

    # Hashtag-only VA/MD "Capital Region" sweeps (word boundaries miss #VirginiaNews).
    sweep = match_post(
        'Huge Number of Illegal Immigrants Arrested in Massive Capital Region Sweep '
        '#CapitalRegion #VirginiaNews #MarylandNews',
        alt_text='Govs Spanberger, Moore Going to Be Big Mad As ICE Swoops Down on VA and MD',
    )
    assert sweep.matched is False
    assert sweep.reason in {
        'hard_negative:virginia_capital_region',
        'hard_negative:md_dc_capital_region',
    }

    keep = match_post(
        'Capital District students from Henrico County visited #AlbanyNY on a class trip.'
    )
    assert keep.matched is True


def test_germany_berlin_brandenburg_capital_region_is_hard_negative() -> None:
    de = match_post(
        'Announced at the Medienboard Berlin-Brandenburg Sundowner, Berlin State '
        'Secretary Michael Biel has revealed that the Berlin-Brandenburg capital '
        'region will have one million euros more for games funding.'
    )
    assert de.matched is False
    assert de.reason == 'hard_negative'

    keep = match_post('Capital Region orchestra plays Berlin repertoire this weekend in #AlbanyNY.')
    assert keep.matched is True


def test_louisiana_crpc_capital_region_planning_commission_is_hard_negative() -> None:
    crpc = match_post(
        'Gonzales council just greenlit a crucial $1.35 million grant application '
        'to enhance traffic signals on Highway 44. #GonzalesAscensionParish #LA',
        alt_text=(
            'Council authorized submission of a Capital Region Planning Commission '
            'carbon-reduction grant application to upgrade three traffic signals.'
        ),
    )
    assert crpc.matched is False
    assert crpc.reason == 'hard_negative'

    keep = match_post(
        'Capital Region exporters shipped goods to Louisiana through the Port of Albany '
        '(#AlbanyNY).'
    )
    assert keep.matched is True


def test_aircraft_type_suffix_ny_does_not_unlock_malta() -> None:
    jet = match_post(
        'A321-271NY, Wizz Air Malta, D-AVYQ, 9H-XLG (MSN 13123) | Fourth Flight '
        'XFW-XFW - Customer Acceptance Flight'
    )
    assert jet.matched is False
    assert jet.reason in {'hard_negative:malta_europe', 'ambiguous_no_context:malta'}

    keep = match_post('Malta, NY town board meets about the solar farm tonight.')
    assert keep.matched is True


def test_brunswick_pike_nj_is_hard_negative() -> None:
    pike = match_post(
        'BLUE BOX: 2542 BRUNSWICK PIKE, LAWRENCEVILLE NJ 08648 '
        'BLUE BOX: 1342 CENTRAL AVE, FAR ROCKAWAY NY 11691'
    )
    assert pike.matched is False
    assert pike.reason == 'hard_negative'

    keep = match_post('Town of Brunswick NY planning board meets Thursday.')
    assert keep.matched is True


def test_saratoga_maiden_watch_and_ccc_are_strong_positive() -> None:
    maiden = match_post(
        'American History and Forever Carina lead this week’s Maiden Watch after '
        'Aug. 22 maiden special weight races at Saratoga.'
    )
    assert maiden.matched is True
    assert maiden.reason == 'strong_positive'

    ccc = match_post(
        'In 1939, the CCC arrived at Saratoga. Young men cleared vegetation, '
        'removed fences, built roads and trails.'
    )
    assert ccc.matched is True
    assert ccc.reason == 'strong_positive'


def test_disney_saratoga_springs_not_multi_local() -> None:
    disney = match_post(
        'Room-by-room tour of the new Treehouse Villas at Saratoga Springs is out.\n'
        'https://www.wdwmagic.com/resorts/Treehouse-Villas-at-Disneys-Saratoga-Springs'
        '-Resort-and-Spa/news/06Aug2026.htm',
        author_handle='wdwmagic.bsky.social',
    )
    assert disney.matched is False
    assert disney.reason == 'hard_negative:disney_saratoga'

    resort = match_post("Staying at Disney's Saratoga Springs Resort next month.")
    assert resort.matched is False

    keep = match_post('Saratoga Springs, NY weekend at the Race Course.')
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
    # "Irish" (not only "Ireland") + Galway + New York must still drop.
    irish = match_post(
        'My latest adventures in and around Galway.',
        alt_text=(
            'Day Tripping from Galway | My Irish road trip envy started with a '
            'Facebook post I saw while still living in New York.'
        ),
    )
    assert irish.matched is False
    assert irish.reason == 'hard_negative:galway_ireland'
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

    # Council Watersplash promos often omit "Rotherham" in the body.
    watersplash = match_post(
        "If you're planning a visit to Clifton Park during this school holidays, "
        "don't forget that the Watersplash is open.",
        author_handle='rotherhamcouncil.bsky.social',
    )
    assert watersplash.matched is False
    assert watersplash.reason == 'hard_negative:clifton_park_uk'

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


def test_ukraine_capital_region_is_hard_negative() -> None:
    wire = match_post(
        'Russian missile and drone barrage in Ukrainian capital region kills at least 15'
    )
    assert wire.matched is False

    kyiv = match_post('Overnight strikes in the capital region near Kyiv killed 15 people.')
    assert kyiv.matched is False
    assert kyiv.reason == 'hard_negative:ukraine_capital_region'

    keep = match_post('Capital Region aid groups sent medical supplies to Kyiv from #AlbanyNY.')
    assert keep.matched is True


def test_russia_capital_region_is_hard_negative() -> None:
    russia = match_post(
        'Escalating drone campaign against Russian military infrastructure.\n'
        'Russia: Inland sabotage routes now threaten the capital region.'
    )
    assert russia.matched is False
    assert russia.reason == 'hard_negative:russia_capital_region'

    keep = match_post(
        'Capital Region aid groups shipped medical supplies to Moscow from #AlbanyNY.'
    )
    assert keep.matched is True


def test_malta_jfk_tourism_not_malta_ny() -> None:
    tourism = match_post(
        'eturbonews.com/endless-summ...',
        alt_text=(
            "Malta's Endless Summer for Americans goes until October 23. "
            "Delta's Nonstop New York JFK- Malta service. Malta Tourism USA explains."
        ),
    )
    assert tourism.matched is False
    assert tourism.reason == 'hard_negative:malta_europe'

    keep = match_post('Hiring store associates in Malta, NY this fall.')
    assert keep.matched is True


def test_saratoga_battlefield_and_racing_strong_positives() -> None:
    assert (
        match_post('Hear 18th-century fife & drum at Saratoga Battlefield this weekend.').matched
        is True
    )
    assert (
        match_post(
            "Ancient Egypt is the top pick in Saturday's $500,000 Christophe Clement at Saratoga."
        ).matched
        is True
    )
    assert (
        match_post('Survie makes her second Saratoga start after winning the Glens Falls.').matched
        is True
    )
    assert match_post('#Saratoga 8/14/26 Race 9 - Smart and Fancy projected odds').matched is True


def test_sudan_capital_region_is_hard_negative() -> None:
    sudan = match_post(
        "For more than a century, the Sunut Forest was an oasis in Khartoum, Sudan's capital.",
        alt_text=(
            "Three years into a brutal civil war, the country's capital region is awash "
            'in rubble, sewage and bodies.'
        ),
    )
    assert sudan.matched is False
    assert sudan.reason == 'hard_negative:sudan_capital_region'

    keep = match_post(
        'Capital Region aid groups shipped medical supplies to Khartoum from #AlbanyNY.'
    )
    assert keep.matched is True


def test_brussels_capital_region_hyphen_is_hard_negative() -> None:
    hyphen = match_post(
        'The Iris Festival celebrates the Brussels-Capital Region each year around 8 May.'
    )
    assert hyphen.matched is False

    spaced = match_post('Tourism board promotes the Brussels Capital Region this spring.')
    assert spaced.matched is False


def test_schenectady_hashtag_stuffing_is_not_strong_positive() -> None:
    """Compound hashtags must not unlock bare Schenectady strong positives."""
    spam = match_post(
        'Cleaning & Organization Cuts Cleanup Costs by 45%\n'
        '#schenectadyparkcleanup #tennisclubvolunteerguide #juneteenthcommunityevent'
    )
    assert spam.matched is False

    assert match_post('Schenectady City Council meets Tuesday at City Hall.').matched is True
    assert match_post('Computer repair service in Schenectady, NY.').matched is True


def test_scotia_montreal_not_village_of_scotia() -> None:
    osheaga = match_post(
        'New York five-piece WHATMORE introduced Osheaga audiences on the '
        'Scotia Forest Stage at Parc Jean-Drapeau.'
    )
    assert osheaga.matched is False
    assert osheaga.reason == 'hard_negative:scotia_montreal'

    cinema = match_post(
        'Wacky hijinks in modern day New York.',
        alt_text='Tuesday, August 4th, 2026 - 10:00 PM Cinéma Banque Scotia Montréal',
    )
    assert cinema.matched is False
    assert cinema.reason == 'hard_negative:scotia_montreal'

    assert match_post('Scotia, NY fire department open house this Saturday.').matched is True


def test_watervliet_mi_not_watervliet_ny() -> None:
    mi = match_post('RN Acute Inpatient Rehab - Watervliet, MI Job listing')
    assert mi.matched is False
    assert mi.reason in {'entity_other:watervliet_mi', 'hard_negative'}

    assert match_post('Watervliet, NY water main break on 19th Street.').matched is True


def test_the_egg_with_albany_is_strong_positive() -> None:
    """The Egg + Albany keeps even without explicit NY (tour footnotes, PT copy)."""
    assert match_post('show especial para familiares e amigos no The Egg, em Albany.').matched
    assert match_post('Family-and-friends show at The Egg in Albany.').matched


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


def test_waterford_ct_not_waterford_ny() -> None:
    assert (
        match_post(
            'Waterford town beach, meets rocky area, where fisherman often sit patiently. '
            'Ferry from New London CT. to Long Island NY, in the background.'
        ).matched
        is False
    )
    assert match_post('Town of Waterford, NY board meeting tonight.').matched is True


def test_donna_troy_not_troy_ny() -> None:
    assert (
        match_post(
            "GREEN LANTERN #68 | Nov '95\n"
            'Can Kyle and Donna Troy stop Mr. Freeze before he puts the city of '
            'New York on ice forever?'
        ).matched
        is False
    )
    assert match_post('Dinner in Troy, New York tonight.').matched is True


def test_loudonville_ohio_not_loudonville_ny() -> None:
    assert (
        match_post('Advantage Air Heating & Cooling HVAC Contractor in Loudonville, OH').matched
        is False
    )
    hashtag = match_post(
        '#Ohio voter alert #Parma #Brunswick #Wooster #Loudonville #Massillon #Canton'
    )
    assert hashtag.matched is False
    assert hashtag.reason == 'hard_negative:loudonville_oh'
    assert match_post('Road work begins in Loudonville near Albany this week.').matched is True
    # Ohio high-school soccer (Waynedale / Golden Bears) without ", OH".
    soccer = match_post(
        'In a thrilling season opener, the Golden Bears emerged victorious against '
        'Loudonville with a 1-0 win.'
    )
    assert soccer.matched is False
    assert soccer.reason == 'hard_negative:loudonville_oh'


def test_ottawa_citizen_capital_region_not_ny() -> None:
    assert (
        match_post(
            'Busy overnight shift for OPP highway patrol in capital region '
            'ottawacitizen.com/news/busy-shift',
            author_handle='ottawacitizen.com',
        ).matched
        is False
    )
    assert match_post("New York's Capital Region flash flood warning until 10pm.").matched is True


def test_albany_ga_radio_market_not_multi_local() -> None:
    nielsen = (
        "Today's markets include Nielsen Spring ratings for Albany GA, Ann Arbor, "
        'Beaumont/Port Arthur, Bloomington IL, Brunswick, Dothan, and Savannah.'
    )
    result = match_post(nielsen)
    assert result.matched is False
    assert result.reason in {'entity_other:albany_georgia', 'hard_negative'}
    assert match_post('Drive from Albany to Brunswick for the farmers market.').matched is True


def test_reinvent_albany_nyc_advocacy_not_capital_region() -> None:
    assert (
        match_post(
            'reinventalbany.org/2026/08/mayoral-election-systems',
            alt_text=(
                'Mayoral Election Systems in the 50 Largest U.S. Cities - Reinvent Albany '
                'looked at how the 50 most populous cities select their mayors. '
                'New York City’s election process is unique.'
            ),
        ).matched
        is False
    )
    assert match_post('Moved from NYC to Albany for outdoor activities.').matched is True


def test_iceland_capital_region_is_hard_negative() -> None:
    quiet = match_post(
        'Police record 56 cases during relatively quiet Saturday #Iceland #police #reykjavik',
        alt_text=(
            'Police in the capital region recorded 56 cases between 5am and 5pm on Saturday, '
            'mbl.is reported.'
        ),
    )
    assert quiet.matched is False
    assert quiet.reason == 'hard_negative:iceland_capital_region'

    pride = match_post(
        'Police step up security for Pride parade #police #Prideparade #reykjavik',
        alt_text=(
            'Árni Friðleifsson, deputy chief of the traffic division of the Capital Region '
            'Police, told mbl.is.'
        ),
    )
    assert pride.matched is False
    assert pride.reason == 'hard_negative:iceland_capital_region'

    # mbl.is "Capital District Fire and Rescue" uses district, not region.
    crash = match_post(
        'One taken to hospital after two-car crash #police #reykjavik #trafficaccident',
        alt_text=(
            'One person was taken to hospital after a two-car crash at Miklabraut and '
            'Grensásvegur late Tuesday night, mbl.is reported. According to the Capital '
            'District Fire and Rescue Service, two ambulances were sent to the scene.'
        ),
    )
    assert crash.matched is False

    keep = match_post('Capital Region exporters shipped fish to Reykjavik from #AlbanyNY.')
    assert keep.matched is True


def test_egg_kansas_city_art_garden_is_hard_negative() -> None:
    kc = match_post(
        'Bottom’s Up Festival at The Egg and Art Garden KC',
        alt_text=(
            'The second annual Bottoms Up festival was greeted by a sunny early summer '
            'weekend in Northeast Kansas City.'
        ),
    )
    assert kc.matched is False

    keep = match_post('Jazz night at The Egg in downtown #AlbanyNY this Friday.')
    assert keep.matched is True
    assert keep.reason == 'strong_positive'


def test_finland_capital_region_is_hard_negative() -> None:
    hsl = match_post(
        'HSL introduces winter timetables with more trains and buses',
        alt_text=(
            'Helsinki Region Transport (HSL) will introduce its winter timetable from '
            '10 August, bringing more frequent public transport services across the '
            'capital region.'
        ),
    )
    assert hsl.matched is False
    assert hsl.reason == 'hard_negative:finland_capital_region'

    keep = match_post('Capital Region exporters shipped goods to Helsinki from #AlbanyNY.')
    assert keep.matched is True


def test_australia_capital_region_cancer_relief_is_hard_negative() -> None:
    charity = match_post(
        'Rise Above celebrates 40 years of helping cancer patients',
        alt_text=(
            'Rise Above – Capital Region Cancer Relief will celebrate 40 years of supporting '
            'cancer patients and their families with a free event at the Royal Hotel Queanbeyan.'
        ),
    )
    assert charity.matched is False
    assert charity.reason == 'hard_negative:australia_capital_region'

    keep = match_post('Capital Region aid groups sent supplies to Canberra from #AlbanyNY.')
    assert keep.matched is True


def test_georgia_atlanta_capital_region_is_hard_negative() -> None:
    ice = match_post(
        'PHOTOS: 1,200 Illegal Aliens Arrested in Georgia #BorderCrisis',
        alt_text=(
            'Federal authorities rounded up more than 1,200 people during a major operation '
            'in the state of Georgia. ICE announced Operation Safe Community – Atlanta, '
            'which was carried out statewide but focused on the capital region.'
        ),
    )
    assert ice.matched is False
    assert ice.reason == 'hard_negative:georgia_atlanta_capital_region'

    keep = match_post('Capital Region exporters shipped goods to Atlanta from #AlbanyNY.')
    assert keep.matched is True


def test_malta_gozo_film_troy_title_not_multi_local() -> None:
    gozo = match_post(
        'The Guardian piece on Malta as a film magnet notes Madame Blanc was filmed on Gozo. '
        'Over 100 productions have shot here, from Troy to The Count of Monte Cristo.',
        alt_text='The beautiful cove of Mgarr ix-Xini, where Two Weeks in August filmed.',
    )
    assert gozo.matched is False
    assert gozo.reason == 'hard_negative:malta_europe'

    assert (
        match_post('Town of Malta planning board meets with Troy officials tonight.').matched
        is True
    )


def test_rotterdam_netherlands_oda_new_york_not_ambiguous() -> None:
    tower = match_post(
        'POST Rotterdam Tower / ODA New York - https://www.archdaily.com/1181776/post-rotterdam',
        alt_text='Completed in 2026 in Rotterdam, The Netherlands. Images by Ossip van Duivenbode.',
    )
    assert tower.matched is False
    assert tower.reason == 'hard_negative:malta_europe'

    assert (
        match_post('Town of Rotterdam meeting; see also Troy City Council agenda.').matched is True
    )


def test_rotterdam_new_york_pizza_dutch_chain_not_ny() -> None:
    pizza = match_post(
        'Twee Rotterdamse vestigingen van New York Pizza failliet',
        alt_text='https://dagblad010.nl/rotterdam/twee-rotterdamse-vestigingen-van-new-york-pizza',
    )
    assert pizza.matched is False
    assert pizza.reason == 'hard_negative:malta_europe'

    rijnmond = match_post(
        'New York Pizza wil na faillissement doorgaan met Rotterdamse vestigingen #rijnmond',
        alt_text=(
            'De vestigingen van pizzaketen New York Pizza in Rotterdam-Keizerswaard '
            'zijn failliet verklaard.'
        ),
    )
    assert rijnmond.matched is False
    assert rijnmond.reason == 'hard_negative:malta_europe'

    assert match_post('Rotterdam, NY fire department open house this weekend.').matched is True


def test_loi_galway_waterford_fixture_list_not_multi_local() -> None:
    loi = match_post(
        '#DerryCityFC have 9 League fixtures remaining. Home against Rovers, Galway, '
        'St Pats and Dundalk and away to Bohs, Waterford, Shels, Drogheda, Sligo.'
    )
    assert loi.matched is False
    assert loi.reason == 'hard_negative:galway_ireland'


def test_aging_albany_ny_pubmed_journal_not_local() -> None:
    journal = match_post(
        'Aging is indexed by PubMed/Medline abbreviated as “Aging (Albany NY)”, '
        'PubMed Central, and Web of Science.'
    )
    assert journal.matched is False
    assert journal.reason == 'hard_negative'


def test_delmar_avenue_street_not_town_of_delmar() -> None:
    usps = match_post(
        'BLUE BOX: 886 MARYVALE DR, CHEEKTOWAGA NY 14225 PO LOBBY: 125 S DELMAR AVE, SALEM IL 62881'
    )
    assert usps.matched is False
    assert usps.reason == 'hard_negative'


def test_japan_capital_district_senryu_is_hard_negative() -> None:
    senryu = match_post(
        '萬歳の足駄に府下の霜柱\n'
        'Celebrating “Banzai!” in wooden clogs—the frost columns of the capital district\n\n'
        '- Kenkabo Inoue\n\n#senryu'
    )
    assert senryu.matched is False
    assert senryu.reason == 'hard_negative:japan_capital_district'

    assert match_post("New York's Capital District flash flood warning until 10pm.").matched is True


def test_burnt_hills_drought_alt_text_not_town() -> None:
    drought = match_post(
        'We still have some green! It is next to a river though.',
        alt_text=(
            'A green playing field with houses and brown drought/burnt hills in the distance'
        ),
    )
    assert drought.matched is False
    assert drought.reason == 'hard_negative:burnt_hills_descriptive'

    assert match_post('Concert tonight in Burnt Hills, NY at the high school.').matched is True


def test_rensselaer_county_roblox_not_local() -> None:
    roblox = match_post(
        'Je parle des jeux #roblox Greenville, Rensselaer County et STU26.',
        alt_text="Chaîne dédiée à l'immersion dans l'univers Roblox !",
    )
    assert roblox.matched is False
    assert roblox.reason == 'hard_negative:rensselaer_roblox'

    assert match_post('Rensselaer County legislature meets Tuesday in Troy.').matched is True


def test_rowonebrand_albany_city_list_not_local() -> None:
    spam = match_post(
        'Row One | Historic Sports Art Prints\n'
        'Montreal | Buffalo | Syracuse, NY | Albany | Rochester | NYC rowonebrand.com'
    )
    assert spam.matched is False

    assert match_post('Moved from NYC to Albany for outdoor activities.').matched is True


def test_e_greenbush_nws_abbreviation_is_strong_positive() -> None:
    alert = match_post(
        'Severe Thunderstorm Near Ravena or 13 Miles S of Delmar Moving E At 30 MPH. '
        'Locations Impacted Include Albany, E Greenbush, Rensselaer, Chatham, Nassau, '
        'Delmar, New Baltimore'
    )
    assert alert.matched is True
    assert match_post('Road work on Route 4 in East Greenbush this week.').matched is True
    assert match_post('N Greenbush fire department open house Saturday.').matched is True


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
    assert (
        match_post(
            'We’re at Sara’s Kitchen in Saratoga Springs. Tomorrow we’ll play the ponies '
            'and go to Boca Bistro.'
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


def test_victoria_bc_island_rail_capital_region_not_ny() -> None:
    """Vancouver Island / Goldstream 'Capital Region' is not NY."""
    island = match_post(
        'Track clearing in the Capital Region from Goldstream to Victoria may be '
        'part of the island rail feasibility study.',
        author_handle='restoreislandrail.bsky.social',
    )
    assert island.matched is False
    assert island.reason == 'hard_negative:canadian_capital_region'

    handle_only = match_post(
        'Sightings of track clearing in the Capital Region this week.',
        author_handle='restoreislandrail.bsky.social',
    )
    assert handle_only.matched is False

    keep = match_post('Track work continues across the Capital Region from Albany to Schenectady.')
    assert keep.matched is True


def test_waterford_crystal_not_waterford_ny() -> None:
    crystal = match_post(
        'New listing up.',
        alt_text=(
            'Waterford Crystal Seahorse Ball Ornament - Etsy. Ships from Huntington Station, NY.'
        ),
    )
    assert crystal.matched is False
    assert match_post('Town of Waterford, NY board meeting tonight.').matched is True


def test_norwegian_ny_does_not_unlock_troy() -> None:
    """Sentence-initial Norwegian/Danish 'Ny' ('New') is not the NY abbreviation."""
    norway = match_post(
        'Ny bok om stortingsvalget 2025. Jeg og Troy bidrar med eget kapittel.',
        alt_text='Norsk politikk er i endring. Tromsø, Trondheim, Bergen og Oslo.',
    )
    assert norway.matched is False
    assert match_post('Dinner in Troy, NY tonight.').matched is True
    assert match_post('Dinner in troy, ny tonight.').matched is True


def test_rensselaer_city_and_rpi_not_indiana() -> None:
    protest = match_post('Honk and wave in Rensselaer, NY at Washington Ave and I-90 exit 7.')
    assert protest.matched is True

    rpi = match_post('Rensselaer Polytechnic Institute hosted a career fair on campus.')
    assert rpi.matched is True
    assert rpi.reason == 'strong_positive'

    indiana = match_post('Hiring CNC operators in Rensselaer, Indiana this fall.')
    assert indiana.matched is False


def test_saratoga_thoroughbred_and_fasig_tipton_without_ny() -> None:
    thoroughbred = match_post(
        'The Thoroughbred Aftercare Alliance summit in Saratoga Springs brought '
        'together retired racehorse groups. ##OTTB'
    )
    assert thoroughbred.matched is True

    fasig = match_post("Cody's Wish led first-crop yearling sires after Fasig-Tipton Saratoga.")
    assert fasig.matched is True

    stakes = match_post(
        'Outfielder returns in the $225,000 Mahony Stakes (G3T) Aug. 16 at Saratoga.'
    )
    assert stakes.matched is True

    # National Del Mar + Saratoga hashtag soup must stay dropped.
    assert match_post('#horseracing #saratoga #delmar #delmarthoroughbredclub').matched is False


def test_i787_requires_highway_prefix_not_study_n() -> None:
    study = match_post(
        'Monteris Medical Announces Landmark Study on 787 Brain Tumor Patients '
        'Treated with NeuroBlate',
        alt_text='MINNETONKA, Minn., Aug. 19, 2026 /PRNewswire/',
    )
    assert study.matched is False

    assert match_post('Crash on I-787 northbound near downtown Albany.').matched is True
    assert match_post('Backup on I-787 south of Madison Ave.').matched is True


def test_troy_aikman_not_troy_ny() -> None:
    aikman = match_post(
        '2026 NFL Preseason Week 2 Schedule, TV, Announcers',
        alt_text=(
            'Thursday Las Vegas @ Houston 8:00PM ESPN Joe Buck and Troy Aikman. '
            'Friday New York Jets @ New England.'
        ),
    )
    assert aikman.matched is False
    assert match_post('Dinner in Troy, New York tonight.').matched is True


def test_saratoga_race_cards_and_albany_region_recall() -> None:
    race = match_post("Wednesday's Bettor Bets play of the day is the 6th race at Saratoga.")
    assert race.matched is True

    stakes_at = match_post(
        'NYSS Statue of Liberty Stakes Preview @ Saratoga | DRF Wednesday Race of the Day'
    )
    assert stakes_at.matched is True

    special = match_post(
        'Brendan Walsh trainees Real Restraint and Title Role are buds! '
        'Shot for The Saratoga Special'
    )
    assert special.matched is True

    battles = match_post(
        'On Aug. 19, 1777, Horatio Gates took command. The Battles of Saratoga '
        'began long before Sept. 19.'
    )
    assert battles.matched is True

    region = match_post(
        '7 Brew Coffee now has five locations in the Albany region and more are on the way.'
    )
    assert region.matched is True


def test_waynedale_handle_loudonville_not_ny() -> None:
    soccer = match_post(
        "Highlights from Tuesday's 1-0 victory over Loudonville.",
        author_handle='waynedalesoccer.bsky.social',
    )
    assert soccer.matched is False
    assert soccer.reason == 'hard_negative:loudonville_oh'
    assert match_post('Road work begins in Loudonville near Albany this week.').matched is True


def test_dc_go_go_capital_region_not_ny() -> None:
    go_go = match_post(
        'Go-go music has an age problem. Plus, more things to do this weekend.',
        alt_text=(
            '7 things to do in the capital region, from a go-go concert to a demolition derby'
        ),
        author_handle='zuriberry.com',
    )
    assert go_go.matched is False
    assert go_go.reason == 'hard_negative:md_dc_capital_region'
    assert match_post("New York's Capital Region flash flood warning until 10pm.").matched is True


def test_rotterdam_film_festival_not_rotterdam_ny() -> None:
    trailer = match_post(
        "NYC's Hidden Libraries Occult Mystery Thriller 'Chronovisor' Trailer",
        alt_text='It premiered at the 2026 Rotterdam Film Festival.',
    )
    assert trailer.matched is False
    assert trailer.reason == 'hard_negative:malta_europe'
    assert match_post('Town board meeting in Rotterdam, NY tonight.').matched is True


def test_troy_nyhammer_not_troy_ny() -> None:
    card = match_post(
        'Pre-match vibe: Brighton visit Tromso in the Conference League Qual.',
        alt_text=(
            'Tromsø IL: Jakob Haugaard, Vetle Skjærvik, Troy Nyhammer\n'
            'Brighton & Hove Albion: Bart Verbruggen'
        ),
    )
    assert card.matched is False
    assert match_post('Dinner in Troy, NY tonight.').matched is True


def test_brunswick_records_not_brunswick_ny() -> None:
    shellac = match_post(
        'The Cotton Pickers - Mishawaka Blues (1925). Recorded in New York, NY 6 Feb. 1925.',
        alt_text='"Mishawaka Blues" The Cotton Pickers (Brunswick, 1925)',
    )
    assert shellac.matched is False
    assert shellac.reason == 'hard_negative:brunswick_records'
    assert match_post('Dinner in Brunswick, NY tonight.').matched is True


def test_albany_business_review_and_rentredi_recall() -> None:
    abr = match_post(
        'Walter Thorne, market president and publisher of the Albany Business Review, '
        'has announced he will exit the role.'
    )
    assert abr.matched is True

    rentredi = match_post(
        "A few years back, the founder of what's now RentRedi in Latham lost out on "
        'an apartment because of paperwork.'
    )
    assert rentredi.matched is True

    race_card = match_post(
        'Play the low takeout Cross Country Pick 5 today!\n'
        'Leg A: Saratoga – Race 5 (3:29 PM ET)\n'
        'Leg B: Horseshoe Indianapolis – Race 6'
    )
    assert race_card.matched is True


def test_denmark_and_alberta_capital_region_not_ny() -> None:
    flight = match_post('09-0540 Took off from Copenhagen, Capital Region, Denmark.')
    assert flight.matched is False
    assert flight.reason == 'hard_negative:denmark_capital_region'

    letbane = match_post(
        'Capital region light rail reaches full opening with public celebration',
        alt_text=(
            'DR reports that the capital region’s light rail system will open between '
            'Ishøj and Lundtofte. The Hovedstadens Letbane festival is at Gladsaxe.'
        ),
    )
    assert letbane.matched is False
    assert letbane.reason == 'hard_negative:denmark_capital_region'

    alberta = match_post(
        'The project would double the energy needs for the entire Capital region.',
        alt_text='ALBERTA FACES OUTRAGE #CdnPoli #AbLeg #AbPoli Nate Glubish',
        author_handle='barbh-ab.bsky.social',
    )
    assert alberta.matched is False
    assert alberta.reason == 'hard_negative:alberta_capital_region'


def test_nps_outside_capital_region_and_victoria_island_peers() -> None:
    nps = match_post(
        'Funding allocated to national parks outside the capital region plunged by $854 million.'
    )
    assert nps.matched is False
    assert nps.reason == 'hard_negative:md_dc_capital_region'

    peers = match_post(
        'The program spoke with Peers Victoria Resources Society.\n'
        'vancouverislandmentalhealthsociety.org/podcast/vict...',
        alt_text='Organization offers variety of programming in the capital region',
    )
    assert peers.matched is False
    assert peers.reason == 'hard_negative:canadian_capital_region'


def test_troy_fautanu_helen_of_troy_and_troy_sc() -> None:
    fautanu = match_post(
        'Pittsburgh’s offense struggled badly against New York, but Troy Fautanu '
        'delivered a positive performance in the loss.'
    )
    assert fautanu.matched is False

    helen = match_post(
        'Helen of Troy, 1993',
        alt_text='Helen of Troy, 1993: Poems —The New York Times Book Review',
    )
    assert helen.matched is False

    gsp = match_post(
        'Severe Thunderstorm Warning by NWS Greenville-Spartanburg SC',
        alt_text=(
            'Southern Greenwood County in Upstate South Carolina... '
            '6 miles east of Troy, moving east at 20 mph.'
        ),
        author_handle='gsp.nws-bot.us',
    )
    assert gsp.matched is False
    assert gsp.reason == 'hard_negative:troy_sc'
    assert match_post('Dinner in Troy, New York tonight.').matched is True


def test_new_york_hotel_rotterdam_not_rotterdam_ny() -> None:
    hotel = match_post(
        "New York's decision to reinvigorate public-sector infrastructure.",
        alt_text='Photo in front of the New York Hotel in Rotterdam.',
    )
    assert hotel.matched is False
    assert hotel.reason == 'hard_negative:malta_europe'
    assert match_post('Town board meeting in Rotterdam, NY tonight.').matched is True


def test_valleycats_egg_crossgates_park_playhouse_saratoga250_recall() -> None:
    assert (
        match_post('Catch the Tri-City ValleyCats at Joseph L. Bruno Stadium tonight.').matched
        is True
    )
    assert match_post('The Egg presents a jazz night Saturday.').matched is True
    assert match_post('7 Brew opening at Crossgates Commons this fall.').matched is True
    assert match_post('Park Playhouse is back in Washington Park with free seats.').matched is True
    assert (
        match_post("Burgoyne's defeat. Read it against Saratoga's landscape. #Saratoga250").matched
        is True
    )


def test_bulgaria_japan_michigan_capital_region_not_ny() -> None:
    bulgaria = match_post(
        'Bulgarian authorities identified a GPS jamming device in Sofia. '
        'The device degraded GPS reception in the capital region.'
    )
    assert bulgaria.matched is False
    assert bulgaria.reason == 'hard_negative:bulgaria_capital_region'

    japan = match_post(
        'A magnitude 5.9 earthquake off the coast of Ibaraki disrupted rail '
        'services across the capital region.',
        alt_text='Magnitude 5.9 earthquake strikes eastern Japan',
    )
    assert japan.matched is False
    assert japan.reason == 'hard_negative:japan_capital_region'

    lansing = match_post(
        'NWS Grand Rapids MI tracked a thunderstorm over Capital Region '
        'International Airport, or near Lansing, moving east at 30 mph.'
    )
    assert lansing.matched is False

    grr = match_post(
        'Special Weather Statement issued by NWS Grand Rapids MI',
        alt_text='Thunderstorm over Capital Region International Airport near Lansing',
        author_handle='grr.nws-bot.us',
    )
    assert grr.matched is False
    assert match_post("Dinner plans in New York's Capital Region tonight.").matched is True


def test_albany_county_wy_not_rescued_by_albany_county_strong() -> None:
    flood = match_post(
        'Flash Flood Warning for Albany, WY #WYwx FFWCYS. '
        'Southeastern Albany County in southeastern Wyoming.',
        alt_text='Flash Flood Warning issued by NWS Cheyenne WY',
    )
    assert flood.matched is False
    assert flood.reason == 'entity_other:albany_county_wy'
    assert match_post('Albany County legislators meet in downtown #AlbanyNY.').matched is True


def test_around_lake_george_not_round_lake() -> None:
    okeeffe = match_post(
        'Yellow Hickory Leaves with Daisy',
        alt_text=(
            'She frequently depicted leaves, inspired by the examples she found on '
            'her walks around Lake George in upstate New York.'
        ),
    )
    assert okeeffe.matched is False
    assert match_post('Farmers market in Round Lake, NY this Saturday.').matched is True


def test_brunswick_tulsa_ok_not_brunswick_ny() -> None:
    jobs = match_post(
        'Summer internships: Software Engineer Intern @ Ambrook NYC; '
        'Software Engineer Intern @ Brunswick Tulsa, OK'
    )
    assert jobs.matched is False
    assert jobs.reason == 'hard_negative:brunswick_ok'
    assert match_post('Zoning hearing in Brunswick, NY next week.').matched is True


def test_saratoga_park_montclair_ca_not_saratoga_ny() -> None:
    park = match_post(
        "Saratoga Park is on the brink of construction — what's next for "
        "Montclair's infrastructure? #MontclairSanBernardinoCounty #CA",
        alt_text='Saratoga Park grant pursuit continues as council reviews capital projects',
    )
    assert park.matched is False
    assert park.reason == 'hard_negative:saratoga_park_ca'
    assert match_post('Morning run in Saratoga Springs, NY.').matched is True


def test_gta_liberty_city_albany_not_albany_ny() -> None:
    gta = match_post(
        'Replaying GTA IV with mods.',
        alt_text=(
            'An "Albany" with Liberty City (New York) plates from Cousin Roman\'s '
            'taxi company in Broker (Brooklyn).'
        ),
    )
    assert gta.matched is False
    assert match_post('ICE activity concerns in #AlbanyNY this week.').matched is True


def test_thacher_harness_spac_orchestra_recall() -> None:
    assert match_post('WildPlay Thacher offers ziplines at Thacher State Park.').matched is True
    assert (
        match_post('17 horses killed in Saratoga Springs Harness Track barn fire.').matched is True
    )
    spac = match_post('Star Wars night with The Philadelphia Orchestra at SPAC.')
    assert spac.matched is True
    assert spac.reason.startswith('event_local_venue')


def test_white_greenbush_not_e_greenbush_abbreviation() -> None:
    madison = match_post(
        'Wrapped up my weekend in Madison with a stop at Greenbush Bakery.',
        alt_text='A white Greenbush Bakery, advertising kosher doughnuts.',
    )
    assert madison.matched is False
    assert match_post('Road work on Route 4 in East Greenbush this week.').matched is True
    assert (
        match_post('Locations Impacted Include Albany, E Greenbush, Rensselaer, Delmar').matched
        is True
    )


def test_helderberg_cape_town_not_helderberg_escarpment() -> None:
    cape = match_post(
        'Congratulations to new members at BP Helderberg in Strand, Cape Town. '
        'Western Cape recruitment team.'
    )
    assert cape.matched is False
    assert match_post('Hike the Helderberg Escarpment this weekend near #AlbanyNY.').matched is True


def test_brussels_slash_capital_region_is_hard_negative() -> None:
    slash = match_post(
        'Brussels authorities keep the yellow drought alert in place across '
        'Brussels/Capital Region despite cooler temperatures.'
    )
    assert slash.matched is False


def test_malta_aircraft_reg_9h_nyc_not_ny_context() -> None:
    flight = match_post(
        'Aterrizaje Vuelo: AXY361A Aeronave: Lineage 1000 (9H-NYC) '
        'Origen: Malta Llegada: Palma de Mallorca (PMI)'
    )
    assert flight.matched is False
    assert flight.reason in {
        'ambiguous_no_context:malta',
        'hard_negative:malta_europe',
    }


def test_nbc4_telemundo_capital_region_is_md_dc() -> None:
    card = match_post(
        'www.thedailybeast.com/nbc-news-anc...',
        alt_text=(
            'NBC News Anchor Quits Live on the Air Joseph Olmo has covered the '
            'capital region for about seven years, with NBC4 and Telemundo 44.'
        ),
    )
    assert card.matched is False
    assert card.reason == 'hard_negative:md_dc_capital_region'


def test_old_albany_post_road_not_albany_ny() -> None:
    listing = match_post(
        'Discover peaceful living at 298 Old Albany Post Road in beautiful '
        'Garrison, NY! #PutnamCounty #HudsonValley'
    )
    assert listing.matched is False
    assert match_post('Walking tour of downtown #AlbanyNY this Saturday.').matched is True


def test_rentredi_hashtag_spam_needs_local_place() -> None:
    spam = match_post(
        '5 Ways Property Management Bleeds Your Budget #rentredi '
        '#aitenantcommunication #propertymanagementproductivity'
    )
    assert spam.matched is False
    assert (
        match_post(
            "A few years back, the founder of what's now RentRedi in Latham "
            'lost out on an apartment because of paperwork.'
        ).matched
        is True
    )


def test_socal_saratoga_hashtag_stuffing_not_race_course() -> None:
    socal = match_post(
        'cynthiapublishing.com/hp_wordpress...\n'
        '#heat #losangeles #socal #horses #horseracing #thoroughbreds '
        '#delmar #saratoga #mountaineerpark',
        alt_text='First Post: roasting Southern California generally',
    )
    assert socal.matched is False
    assert (
        match_post(
            'Check out the likely fields for stakes at Saratoga, Del Mar, '
            'Kentucky Downs and Charles Town.'
        ).matched
        is True
    )


def test_saratoga_meet_and_fort_schuyler_campaign_recall() -> None:
    meet = match_post(
        'Jockey Irad Ortiz, with lingering foot injury, will miss remainder of Saratoga meet'
    )
    assert meet.matched is True

    campaign = match_post(
        'What happened at Fort Schuyler helps explain what happened at Saratoga. '
        'Oriskany, Haudenosaunee political divisions, and Benedict Arnold.'
    )
    assert campaign.matched is True


def test_helderberg_college_capetown_hashtag_not_escarpment() -> None:
    cape = match_post(
        'Unplanned Maintenance - Burst Pipe in Die Wingerd\n'
        'C/O Helderberg College Rd & Hermitage Ave\n'
        '#WaterAndSanitation #CapeTown'
    )
    assert cape.matched is False
    assert match_post('Hike the Helderberg Escarpment this weekend near #AlbanyNY.').matched is True


def test_burnt_hillside_wildfire_alt_not_burnt_hills_ny() -> None:
    fire = match_post(
        "Still a lot of damage from last year's fire on Castle Hill.",
        alt_text=(
            'burnt gorse in front of the Victoria Tower, Castle Hill, Huddersfield '
            'Burnt hillside by the Victoria Tower'
        ),
    )
    assert fire.matched is False
    assert match_post('Concert tonight in Burnt Hills NY at the high school.').matched is True


def test_michigan_hashtag_troy_waterford_not_multi_local() -> None:
    spam = match_post(
        '#Michigan #Detroit #GrandRapids #Warren #SterlingHeights #AnnArbor '
        '#Lansing #Dearborn #Livonia #Troy #FarmingtonHills #Wyoming #Flint '
        '#Kalamazoo #Waterford #Novi #Pontiac #RoyalOak'
    )
    assert spam.matched is False
    assert spam.reason == 'hard_negative:troy_michigan'
    assert match_post('Dinner in Troy NY tonight.').matched is True


def test_saratoga_terrace_binghamton_not_saratoga_ny() -> None:
    terrace = match_post(
        "Binghamton's planning committee reviews the Saratoga Terrace Housing "
        'Development pilot agreement. #BinghamtonCityBroomeCounty #NY'
    )
    assert terrace.matched is False
    assert match_post('Canvass launch at Congress Park in Saratoga Springs NY.').matched is True


def test_parx_thistledown_saratoga_hashtag_stuffing_not_race_course() -> None:
    spam = match_post(
        '#nationaldogday #thoroughbreds #parxracing #horseshoeindy #saratoga '
        '#thistledown #assiniboia #manitoba'
    )
    assert spam.matched is False
    assert (
        match_post(
            'Check out the likely fields for stakes at Saratoga, Del Mar, '
            'Kentucky Downs and Charles Town.'
        ).matched
        is True
    )


def test_woodbine_charlestown_remington_saratoga_hashtag_stuffing() -> None:
    spam = match_post(
        'cynthiapublishing.com/hp_wordpress...\n'
        '#thursday #horses #horseracing #handicapping #longshots #thoroughbreds '
        '#delmar #saratoga #woodbine #charlestownraces #remingtonpark'
    )
    assert spam.matched is False
    assert (
        match_post(
            'Check out the likely fields for stakes at Saratoga, Del Mar, '
            'Kentucky Downs and Charles Town.'
        ).matched
        is True
    )


def test_onca_troy_nymex_not_city_of_troy() -> None:
    gold = match_post(
        'Ouro fecha em leve alta com expectativa por sinalizações do Fed em Jackson Hole',
        alt_text=(
            'Na Comex, divisão de metais da New York Mercantile Exchange (Nymex), '
            'o ouro para dezembro encerrou em alta de 0,23%, a US$ 4 653,30 por onça-troy'
        ),
    )
    assert gold.matched is False
    assert match_post('Dinner in Troy NY tonight.').matched is True


def test_forego_jerkens_grade1_saratoga_and_albany_exec_recall() -> None:
    assert (
        match_post('Mike Welsch previews the Grade 1, $500,000 H. Allen Jerkens Memorial.').matched
        is True
    )
    assert match_post("Book'em Danno looks to go back-to-back in Forego").matched is True
    assert (
        match_post('Chris Gracie is tied to three Grade 1 runners at Saratoga on Aug. 29.').matched
        is True
    )
    assert (
        match_post(
            'LASNNY is hiring a Disability Advocacy Staff Attorney (Albany/Amsterdam).'
        ).matched
        is True
    )
    assert (
        match_post(
            'ALBANY EXEC, DA investigating possible fraud within county employee benefits'
        ).matched
        is True
    )
    assert (
        match_post("Kathy Hochul's Sikorsky just touched down at 4B0 (South Albany).").matched
        is True
    )
    assert (
        match_post(
            'Severe Thunderstorm 7 Miles S of Corinth or 7 Miles NW of Saratoga Springs '
            'Moving E. Locations Impacted Include Corinth, Wilton, Greenfield.'
        ).matched
        is True
    )


def test_long_island_albany_hashtag_stuffing_not_local() -> None:
    spam = match_post(
        'IXCHEL Anxiety Relief #usa #longisland #longislandny #longislandnewyork '
        '#newyork #longislandrealestate #albany #albanyny #Magnesium'
    )
    assert spam.matched is False
    assert (
        match_post('Taking the train from Long Island to #AlbanyNY this weekend.').matched is True
    )


def test_travers_saratoga_feature_and_wins_at_saratoga_recall() -> None:
    travers = match_post(
        'Silent Tactic will try for a first grade 1 win in the $1.25 million '
        'Saratoga feature after the Travers Stakes draw.'
    )
    assert travers.matched is True

    wins = match_post(
        'Javier Castellano reached 6,000 North American wins Aug. 26 at Saratoga, '
        'guiding Starship Lizzy to victory in the finale.'
    )
    assert wins.matched is True


def test_lark_hall_and_albany_mayor_recall() -> None:
    assert match_post('Albany: Helmet @ Lark Hall this Friday.').matched is True
    assert (
        match_post(
            "Albany Mayor leaves for Martha's Vineyard day after city workers hurt "
            'in explosion at city site.'
        ).matched
        is True
    )


def test_sun_times_union_not_times_union() -> None:
    suntimes = match_post(
        'Support the Sun-Times union',
        alt_text='Join us in demanding no layoffs from Chicago Public Media. Sun-Times Guild.',
    )
    assert suntimes.matched is False
    assert (
        match_post('Times Union coverage of the Travers at Saratoga Race Course.').matched is True
    )


def test_newtonville_ma_mbta_not_colonie() -> None:
    villages = match_post(
        'Newton Highlands ≠ Newtonville ≠ West Newton. Know before you search.',
        alt_text="Newton MA's 13 Villages: The Complete Buyer's Guide to Boston's Garden City",
    )
    assert villages.matched is False
    assert villages.reason == 'hard_negative:newtonville_ma'

    mbta = match_post('Train 534 is running 15 minutes late at Newtonville. #MBTA #WorcesterLine')
    assert mbta.matched is False
    assert mbta.reason == 'hard_negative:newtonville_ma'
    assert match_post('Road work on Newtonville Avenue in Colonie near #AlbanyNY.').matched is True


def test_dc_capital_district_not_ny() -> None:
    dc = match_post(
        'Trump has ruined not only the White House, but the entire Capital district, '
        'Washington, D.C., & many of its buildings, monuments, & parks.'
    )
    assert dc.matched is False
    assert dc.reason == 'hard_negative:md_dc_capital_region'
    assert match_post("New York's Capital District sees strong job growth.").matched is True


def test_french_capital_region_not_ny() -> None:
    paris = match_post(
        'France DESTRUCTION',
        alt_text=(
            'Severe weather chaos in Paris! The French capital region was destroyed by a storm.'
        ),
    )
    assert paris.matched is False
    assert paris.reason == 'hard_negative:france_capital_region'
    assert (
        match_post('French exchange students visit the Capital Region this fall. #AlbanyNY').matched
        is True
    )


def test_iceland_kringlan_capital_region_window() -> None:
    iceland = match_post(
        'Long queues form at Kringlan polling station #elections #Iceland #voting',
        alt_text=(
            'mbl.is reported, as many capital-area residents appeared eager to vote. '
            'A similar queue was reported earlier outside the premises of the District '
            'Commissioner of the Capital Region.'
        ),
    )
    assert iceland.matched is False
    assert iceland.reason == 'hard_negative:iceland_capital_region'


def test_green_island_sangha_long_island_not_village() -> None:
    sangha = match_post(
        'Long Island, New York: This Sunday, Green Island Sangha will sit together '
        'at the Mindfulness Center at Adelphi University.'
    )
    assert sangha.matched is False
    assert sangha.reason == 'hard_negative:green_island_other'
    assert match_post('Village of Green Island, NY holds budget hearing.').matched is True


def test_ct_east_hartford_capital_district_not_ny() -> None:
    job = match_post('Roving Personal Banker Capital District - 144783-CT-East Hartford Job')
    assert job.matched is False
    assert job.reason == 'hard_negative:ct_capital_district'


def test_albany_county_library_wyoming_not_ny() -> None:
    lib = match_post(
        'Join us September 29 at 6:30 at the Albany County Library.',
        author_handle='wyomingpublicmedia.bsky.social',
    )
    assert lib.matched is False
    assert lib.reason == 'entity_other:albany_county_wy'
    assert match_post('Albany County legislators meet in downtown #AlbanyNY.').matched is True


def test_brunswick_schools_nyc_not_brunswick_ny() -> None:
    schools = match_post(
        '',
        alt_text=(
            'Kidder student and teacher take Brunswick Schools global at NBA Hoops '
            'championship, June 22-25 in New York City.'
        ),
    )
    assert schools.matched is False
    assert schools.reason == 'hard_negative:brunswick_schools_other'
    assert match_post('Zoning hearing in Brunswick, NY next week.').matched is True


def test_rensselaer_sheriff_and_latham_halfmoon_hq_recall() -> None:
    assert (
        match_post(
            'Caribbean immigration advocates praise lawsuit against Rensselaer sheriff '
            'for violating law against ICE'
        ).matched
        is True
    )
    assert (
        match_post(
            'Banking giant to build new regional HQ in Latham. Spending $23 million on '
            'a new regional office in Latham.'
        ).matched
        is True
    )
    assert (
        match_post(
            'Auto wholesaler buys, renovates new Halfmoon HQ at its new Halfmoon location.'
        ).matched
        is True
    )


def test_boston_mattapan_river_street_not_troy_corridor() -> None:
    boston = match_post(
        'Open Streets Boston returns to Mattapan two weeks from today!\n\n'
        'On Saturday, September 12, Blue Hill Avenue between River Street and '
        'Babson Street will transform into a car-free pedestrian zone.'
    )
    assert boston.matched is False
    assert match_post('Open mic tomorrow on River Street — sign-ups start at 6.').matched is True


def test_bethlehem_pa_steel_unesco_not_town_of_bethlehem() -> None:
    pa = match_post(
        'My birthplace. I grew up there, then moved on to upstate NY.',
        alt_text=(
            'An American steel town with serious Christmas spirit. Bethlehem '
            'showcases its industrial heritage and is home to one of the newest '
            'UNESCO sites in the US.'
        ),
    )
    assert pa.matched is False
    assert pa.reason == 'hard_negative:bethlehem_pa'
    assert match_post('Town of Bethlehem, NY board meeting tonight.').matched is True


def test_troy_person_name_and_jana_not_troy_ny() -> None:
    person = match_post(
        'Thoughts on Jana from Last Page First clique accusations',
        alt_text=(
            'And No Jana and Troy, Just Because My 3rd Cousin Built Malls in '
            'New York State Doesn’t Mean I Know Leslie Wexner.'
        ),
    )
    assert person.matched is False
    assert person.reason == 'hard_negative:troy_person_name'
    assert match_post('Dinner in Troy, New York tonight.').matched is True
    assert match_post('Drive between Albany and Troy, NY this weekend.').matched is True


def test_schenectady_style_cuisine_not_city() -> None:
    food = match_post(
        'We set up at Anaheim and Marine in Wilmington! Try our new '
        'Schenectady-style black bean eggrolls!'
    )
    assert food.matched is False
    assert food.reason == 'hard_negative:schenectady_style'
    assert match_post('Schenectady City Council meets Tuesday at City Hall.').matched is True


def test_albany_state_university_not_ualbany() -> None:
    asu = match_post(
        'How learning a language helps our brains',
        alt_text=(
            'Dr Lou Stelling, professor at Albany State University in New York’s '
            'capital city, gave an excellent talk about language learning.'
        ),
    )
    assert asu.matched is False
    assert asu.reason == 'entity_other:albany_georgia'
    assert (
        match_post('University at Albany professor gave a talk on language learning.').matched
        is True
    )


def test_capital_rep_travers_day_and_saratoga_campaign_recall() -> None:
    assert (
        match_post(
            'Check out #LastAmericanNewspaper at #CapitalRep in #Albany Sept. 25-Oct. 18.'
        ).matched
        is True
    )
    assert (
        match_post("It's Travers Day at Saratoga. The main event is at 6:35 p.m.").matched is True
    )
    assert (
        match_post('Bears Cup in Saratoga was ROCKIN this morning. Travers weekend.').matched
        is True
    )
    assert (
        match_post(
            'Historical markers remind us that the Saratoga Campaign unfolded across '
            'a much larger landscape.'
        ).matched
        is True
    )
    assert (
        match_post('I spoke ahead of the Travers about the lessons of the long meet.').matched
        is True
    )


def test_troy_deeney_mt_kisco_not_troy_ny() -> None:
    deeney = match_post(
        'Why is Troy Deeney at my local in watching the game?  Welcome to Mt Kisco, NY, Troy.'
    )
    assert deeney.matched is False
    assert match_post('Dinner in Troy, New York tonight.').matched is True
    assert match_post('Drive between Albany and Troy, NY this weekend.').matched is True


def test_hungarian_new_brunswick_url_not_brunswick_ny() -> None:
    hu = match_post(
        'New Brunswick: Okostelefon-tilalom az iskolákban szeptembertől\n\n'
        'New Brunswick tartomány új rendeletet vezet be az iskolákban.\n\n'
        'https://itouch.hu/new-brunswick-okostelefon-tilalom-az-iskolakban-szeptembertol/'
    )
    assert hu.matched is False
    assert match_post('Town of Brunswick, NY board meeting Thursday.').matched is True


def test_albany_med_amtrak_alb_saratoga_breeze_and_bjs_rotterdam_recall() -> None:
    assert (
        match_post('The man was flown to Albany Medical Center with serious injuries.').matched
        is True
    )
    assert (
        match_post(
            'Mayfield schools will offer telemedicine through the Albany Med Health System.'
        ).matched
        is True
    )
    assert (
        match_post(
            'AMTRAK Maple Leaf (63) NYP->TWO Alert: Train 63 is currently stopped in '
            'Albany (ALB) due to a mechanical assessment.'
        ).matched
        is True
    )
    # Cascades uses Albany, Oregon station code ALY — must stay dropped.
    assert (
        match_post(
            'AMTRAK Cascades (504) EUG->SEA Alert: delay south of Albany (ALY) '
            'due to a signal outage.'
        ).matched
        is False
    )
    assert (
        match_post(
            'Stirring Words died after a cardiac event while warming up to breeze '
            'Friday at Saratoga.'
        ).matched
        is True
    )
    assert match_post("Golden Tempo's chance for Saratoga immortality has arrived.").matched is True
    assert (
        match_post(
            "BJ's Wholesale Club is gearing up to open a new warehouse in Rotterdam."
        ).matched
        is True
    )


def test_latham_watkins_office_not_latham_hq() -> None:
    firm = match_post(
        "Leading Capital Markets and M&A Partners Strengthen Latham & Watkins' "
        'Hong Kong Office #China #Hong_Kong #Latham_Watkins'
    )
    assert firm.matched is False
    assert firm.reason == 'hard_negative'
    assert (
        match_post(
            'Banking giant to build new regional HQ in Latham. Spending $23 million on '
            'a new regional office in Latham.'
        ).matched
        is True
    )
    assert match_post('Traffic backed up at Latham Circle this afternoon.').matched is True


def test_oregon_south_west_albany_not_south_albany_ny() -> None:
    oregon = match_post(
        '',
        alt_text=(
            'Mid-Willamette Conference football preview: 2026 team outlooks, predicted '
            'order of finish Breaking down the 5A Mid-Willamette, including Dallas, '
            'Corvallis, Crescent Valley, Lebanon, Silverton, South Albany and West Albany'
        ),
    )
    assert oregon.matched is False
    assert oregon.reason == 'hard_negative'
    assert (
        match_post("Kathy Hochul's Sikorsky just touched down at 4B0 (South Albany).").matched
        is True
    )
    assert match_post('Construction starts in South Albany near the airport.').matched is True


def test_lasnny_amtrak_window_saratoga_special_and_1777_recall() -> None:
    assert (
        match_post(
            'Help keep families in their homes. LASNNY is hiring a Foreclosure Prevention '
            'Attorney in Albany to represent homeowners facing foreclosure.'
        ).matched
        is True
    )
    assert (
        match_post(
            'AMTRAK Empire Service (233) NYP->ALB [2026-08-31] Alert:\n'
            'Delay Notification: As of 2:07 PM ET Empire Service Train 233 is operating '
            'approximately 40 minutes late into Albany (ALB) due to rail congestion '
            'along the route.'
        ).matched
        is True
    )
    assert (
        match_post(
            'Twinkle Town, the impressive six-length winner of the Grade 2 Saratoga '
            'Special on Aug. 1 who was scheduled to run in Sunday’s Grade 1 Hopeful.'
        ).matched
        is True
    )
    assert (
        match_post(
            'In 1777, no one knew Saratoga would become a turning point. An American '
            'army was still being built.'
        ).matched
        is True
    )


def test_seattle_times_union_not_times_union() -> None:
    seattle = match_post(
        'Help the Seattle Times union keep AI out of the newsroom.',
        alt_text=(
            'Save Seattle journalism. Members of The Seattle Times Union are being '
            'asked to tell these essential stories.'
        ),
    )
    assert seattle.matched is False
    assert seattle.reason == 'hard_negative'
    assert match_post('Times Union coverage of downtown #AlbanyNY redevelopment.').matched is True


def test_helderberg_southafrica_hashtag_not_escarpment() -> None:
    sa = match_post(
        "Montego's Bags o Wags partners with Sweet Paws to support Helderberg "
        'community caregivers #southafrica',
        alt_text='Sweet Paws Rescue and Care in the Helderberg community.',
    )
    assert sa.matched is False
    assert sa.reason == 'hard_negative'
    assert match_post('Hike the Helderberg Escarpment this weekend near #AlbanyNY.').matched is True


def test_schaghticoke_rd_kent_ct_not_town() -> None:
    ct = match_post(
        'South Cascades along Schaghticoke Rd., Kent, CT. Roadside…60 feet. '
        '#Connecticut #NewEngland #waterfalls'
    )
    assert ct.matched is False
    assert ct.reason == 'hard_negative:schaghticoke_ct'
    assert match_post('Town of Schaghticoke, NY board meeting tonight.').matched is True


def test_watervliet_mi_bridgman_hs_ratings_not_ny() -> None:
    mi = match_post(
        '2026 Ratings: Bridgman. BOYS TEAM RATINGS Buchanan Red Arrow Watervliet '
        'Hartford Bloomingdale.'
    )
    assert mi.matched is False
    assert mi.reason == 'entity_other:watervliet_mi'
    assert match_post('Watervliet, NY water main break on 19th Street.').matched is True


def test_troy_pa_bradford_nws_not_troy_ny() -> None:
    pa = match_post(
        'Tornado Warning issued by NWS Binghamton NY',
        alt_text=(
            'Western Bradford County in northeastern Pennsylvania. At 631 PM EDT, '
            'a severe thunderstorm was located over Springfield, or over Troy, '
            'moving southeast at 25 mph.'
        ),
    )
    assert pa.matched is False
    assert pa.reason == 'hard_negative:troy_pa'
    assert (
        match_post(
            'Democrats and Republicans in Troy, New York, took on a national '
            'Catholic health system.'
        ).matched
        is True
    )


def test_rotterdam_world_city_architecture_not_ny() -> None:
    arch = match_post(
        'Ten architecture and design events this month in Detroit, NYC, LA, '
        'San Francisco, Houston, London, Hong Kong, Paris, and Rotterdam.',
        alt_text='architecture & design events available in Paris and Rotterdam.',
    )
    assert arch.matched is False
    assert arch.reason == 'hard_negative:malta_europe'
    assert match_post('Meal Train for Rotterdam Community Center Free Food Fridge').matched is True


def test_troy_johnson_founder_not_troy_ny() -> None:
    person = match_post(
        'The Future of Book Publishing from the WSJ Future of Everything Festival',
        alt_text=(
            'The festival was held in New York City on May 18, 2022. In this clip, '
            "AALBC.com's Founder, Troy Johnson discusses publishing."
        ),
    )
    assert person.matched is False
    assert person.reason == 'hard_negative:troy_person_name'
    assert match_post('Dinner in Troy, New York tonight.').matched is True


def test_albany_riverfront_powers_park_saratoga_derby_liberty_recall() -> None:
    assert (
        match_post(
            'Critically Acclaimed Jazz Artists set to Perform at Albany Riverfront '
            'Jazz Festival returning to Jennings Landing.'
        ).matched
        is True
    )
    assert match_post("The second jam, held at Troy's Powers Park on August 15.").matched is True
    assert (
        match_post('Saratoga Derby winner Glacius returns in Saturday Nashville Derby.').matched
        is True
    )
    assert (
        match_post("Ben Weaver is working in George Weaver's Saratoga barn this meet.").matched
        is True
    )
    assert (
        match_post(
            "The Liberty Park redevelopment in Albany will team two of the region's "
            'biggest development companies.'
        ).matched
        is True
    )
    assert match_post('Campaign to keep the Burdett Birth Center open.').matched is True


def test_new_york_times_union_not_times_union() -> None:
    nyt = match_post(
        'Unionized New York Times staffers are speaking out against Kalshi.',
        alt_text='New York Times Union Demands Company Abandon Kalshi Talks',
    )
    assert nyt.matched is False
    assert nyt.reason == 'hard_negative'
    assert match_post('Times Union coverage of downtown #AlbanyNY redevelopment.').matched is True


def test_troy_road_ithaca_not_troy_ny() -> None:
    ithaca = match_post(
        'The Ithaca Planning Board approved the community solar project on Troy Road. '
        '#EastIthacaTompkinsCounty #NY'
    )
    assert ithaca.matched is False
    assert ithaca.reason == 'hard_negative:troy_road_ithaca'
    assert match_post('Road work on Troy Road near #AlbanyNY starts Monday.').matched is True


def test_troy_pa_near_troy_nws_binghamton() -> None:
    pa = match_post(
        'Severe Thunderstorm Warning issued by NWS Binghamton NY. '
        'Storm over Springfield, or near Troy, moving southeast at 30 mph.'
    )
    assert pa.matched is False
    assert pa.reason == 'hard_negative:troy_pa'
    assert (
        match_post(
            'Democrats and Republicans in Troy, New York, took on a national '
            'Catholic health system.'
        ).matched
        is True
    )


def test_troy_achilles_film_not_troy_ny() -> None:
    film = match_post(
        'Troy Achilles Speech to Myrmidons [HD]',
        alt_text='ALL Democrats: NY Rep Jeffries, NY Sen Schumer follow the Mag 7.',
    )
    assert film.matched is False
    assert film.reason == 'hard_negative:troy_person_name'
    pitt = match_post('Brad Pitt in Troy (2004). Watching in New York tonight.')
    assert pitt.matched is False
    assert pitt.reason == 'hard_negative:troy_person_name'
    assert match_post('Dinner in Troy, New York tonight.').matched is True


def test_yaddo_saratoga_and_rotterdamcc_allowlist() -> None:
    assert match_post('The Yaddo Mansion in Saratoga, 2025.').matched is True
    from server.allowlists import load_allowlist_dids, load_allowlist_handles

    dids = load_allowlist_dids()
    handles = load_allowlist_handles()
    kept = match_post(
        "Don't forget, Rotterdam! Sister Harmony Group this Friday.",
        author_did='did:plc:4larojxjaliyaswc47za27zi',
        author_handle='rotterdamcc.bsky.social',
        allowlist_dids=dids,
        allowlist_handles=handles,
    )
    assert kept.matched is True
    assert kept.reason == 'allowlist_did'
