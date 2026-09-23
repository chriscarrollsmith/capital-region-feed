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

    # Indivisible BR GOTV pages say Capital Region; handle often omits parish names.
    indivis = match_post(
        'Join Team Jamie Davis Every Saturday for our GOTV Activation Day!! '
        'https://indivisiblebr.org/team-jamie-davis-super-saturday-gotv-activation-capital-region/',
        alt_text='Team Jamie Davis-SUPER SATURDAY-GOTV Activation-CAPITAL REGION',
        author_handle='indivisbatonrouge.bsky.social',
    )
    assert indivis.matched is False
    assert indivis.reason == 'hard_negative:louisiana_capital_region'

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

    # Harrisburg Day of Caring — Allison Hill is not Albany's United Way chapter.
    allison = match_post(
        'Brown Plus team members proudly participated in the United Way of the Capital '
        "Region's 2026 Day of Caring last Friday, helping Wildheart Ministries with litter "
        'removal along the sidewalks of Allison Hill!'
    )
    assert allison.matched is False
    assert allison.reason == 'hard_negative:pennsylvania_capital_region'

    keep = match_post('Capital Region students visited Harrisburg before returning to #AlbanyNY.')
    assert keep.matched is True
    assert (
        match_post(
            'United Way of the Capital Region Day of Caring volunteers downtown (#AlbanyNY).'
        ).matched
        is True
    )


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

    # Bristol UK bus bots: Citylines / inbound route opposite Clifton Park.
    bristol = match_post(
        'The inbound 5 was right on time opposite Clifton Park. Wearing a '
        'Citylines East livery this far west is a mild geographical identity crisis.',
        author_handle='bristolbusbot.live',
    )
    assert bristol.matched is False
    assert bristol.reason == 'hard_negative:clifton_park_uk'

    assert match_post('Water main break on Carlton Road in Clifton Park, NY.').matched is True


def test_clifton_park_baltimore_md_not_ny() -> None:
    md = match_post(
        'Volunteers clean up part of Clifton Park on 9/11 in tribute to a young woman '
        'from Catonsville who perished on Flight 93.',
        alt_text='Tribute to 9/11 victim from Catonsville — www.wmar2news.com',
    )
    assert md.matched is False
    assert md.reason == 'hard_negative:clifton_park_md'
    assert (
        match_post('Farmers market opens Saturday in Clifton Park near #AlbanyNY.').matched is True
    )


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

    # Academic exam proctors + time-of-day cue must not unlock the theatre.
    academic = match_post(
        "Here's what my must-do to-do list for today looks like at 3:30 pm.\n\n"
        '1. Answer urgent emails\n2. Record announcement video\n3. Email proctors\n'
        '4. Fix grade syncing\n5. Letter of rec\n10. Proctor exam'
    )
    assert academic.matched is False
    assert match_post('Sold out night at Proctors — doors at 7pm').matched is True


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

    nj = match_post(
        'Opening Reception 10/10/2026 at the Dr. Martin Luther King Jr. Center Newtonville NJ.'
    )
    assert nj.matched is False
    assert nj.reason == 'hard_negative:newtonville_ma'


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

    greater = match_post(
        "Greater Albany's school board is weighing pension-obligation bonds.\n\n"
        '#OR #MarketReadiness #FinancialTransparency #PublicEducation',
        alt_text=(
            'Board hears pension-bond briefing; bond adviser urges early authorization. '
            'Piper Jaffray briefed the Greater Albany Public School District.'
        ),
        author_handle='citizenptnewsor.bsky.social',
    )
    assert greater.matched is False
    assert greater.reason == 'hard_negative'
    assert (
        match_post(
            'Greater Albany trail network expansions were discussed at City Hall (#AlbanyNY).'
        ).matched
        is True
    )


def test_palace_ichabod_howes_518_and_section2_bethlehem_recall() -> None:
    assert (
        match_post(
            "Trey Anastasio Band will perform at Albany's Palace Theatre on Nov. 12 "
            'before joining Billy Strings for benefit concerts later in November.'
        ).matched
        is True
    )
    assert (
        match_post(
            'Ichabod Crane Central School district board members say they will give public '
            'financial updates, with Valatie residents wanting more answers.'
        ).matched
        is True
    )
    assert (
        match_post(
            'The Iroquois Museum in Howes Cave houses a comprehensive collection of '
            'modern Iroquois art.'
        ).matched
        is True
    )
    assert match_post('Burn ban goes into effect this week. #518outdoors').matched is True
    assert (
        match_post(
            "With confidence skyrocketed, Bethlehem girls' basketball is surging into the "
            'state final four (Section 2). #518hoops'
        ).matched
        is True
    )
    assert match_post('Shenendehowa senior Liam Anderson started powerlifting.').matched is True


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


def test_albany_wire_remote_albuquerque_not_local() -> None:
    remote = match_post(
        'ALBANY, N.Y., Sept. 03, 2026 (GLOBE NEWSWIRE) — Curia Global, Inc. '
        'today announced the grand opening of its expanded campus in '
        'Albuquerque, New Mexico.'
    )
    assert remote.matched is False
    assert remote.reason == 'hard_negative:albany_wire_remote'
    assert (
        match_post(
            'ALBANY, N.Y. (WRGB) — Lawmakers met in the state Capitol about housing.'
        ).matched
        is True
    )
    assert match_post('ALBANY, N.Y. — A new brewery opens on Lark Street downtown.').matched is True


def test_albany_ithaca_contrast_not_cap_region() -> None:
    poem = match_post(
        'I decided to become The Poet Laureate Of Tompkins County, New York. '
        "I don't want to stray again - Not up to Albany, Where the bureaucracy "
        'Survives. Ithaca is where The hearts and souls Of New York Naturally reside.'
    )
    assert poem.matched is False
    assert poem.reason == 'hard_negative:albany_ithaca_contrast'
    assert (
        match_post(
            'Jules Netherland traveled from the Bronx to the New York state Capitol '
            'in Albany several times to lobby for medical aid in dying.'
        ).matched
        is True
    )


def test_glens_falls_and_empire_underground_keep() -> None:
    assert match_post('GLENS FALLS NY Sep 3 Climate Report: High: 82 Low: 59').matched is True
    assert match_post('Glens Falls boys soccer beat Ichabod Crane 3-0').matched is True
    assert match_post('Show tonight at Empire Underground').matched is True
    assert (
        match_post('Moon Tooth announce East Coast dates: Nov. 21 @ Empire Underground').matched
        is True
    )
    assert match_post('Tickets on sale for Saturday at Empire Underground').matched is True


def test_brussels_municipality_capital_region_is_belgium() -> None:
    brussels = match_post(
        'The City of Brussels is now the seventh municipality in the Capital Region '
        'to throw its weight behind the Region’s legal proceedings.',
        alt_text=(
            'City of Brussels joins other municipalities in the fight against RNP-07L overflights'
        ),
    )
    assert brussels.matched is False
    assert brussels.reason == 'hard_negative:belgium_capital_region'
    assert (
        match_post(
            'Capital Region mayors meet partners in Brussels for a twinning trip '
            'with #AlbanyNY officials.'
        ).matched
        is True
    )


def test_delhi_imd_capital_region_is_india() -> None:
    delhi = match_post(
        'IMD Red Alert Delhi Issued: Heavy Rains and Severe Waterlogging!',
        alt_text=(
            'IMD red alert Delhi brings heavy rain and severe waterlogging to the '
            'capital region, disrupting flights and transit. #India #DelhiRain'
        ),
    )
    assert delhi.matched is False
    assert delhi.reason == 'hard_negative:india_capital_region'
    assert (
        match_post('Flood advisory for the Capital Region tonight near #AlbanyNY.').matched is True
    )


def test_schenectady_multi_state_hashtag_spam_drops() -> None:
    spam = match_post(
        'Fire up @\nSnap: funkykush.85\n\n'
        '#Creek #Frederick #Maryland #Oshkosh #Wisconsin #Pittsburg '
        '#Palo-Alto #Bossier-City #Louisiana #Portland #Maine '
        '#Schenectady #New-York'
    )
    assert spam.matched is False
    assert spam.reason == 'hard_negative:schenectady_hashtag_spam'
    assert match_post('Schenectady City Council meets Tuesday at City Hall.').matched is True


def test_albany_fm_bandscan_nyc_not_ny_context() -> None:
    dx = match_post(
        'Drove to Mt Greylock Summit with family. Did FM Bandscan on the way down: '
        'VT, NH, Boston, Albany, Springfield, CT. No NYC or HV. #fmdx #dx'
    )
    assert dx.matched is False
    assert dx.reason == 'hard_negative:albany_bandscan'
    assert match_post('FM bandscan from downtown #AlbanyNY caught WRPI tonight.').matched is True


def test_empac_and_pine_bush_and_proctors_sold_out_keep() -> None:
    assert match_post('EMPAC presents a new media installation this week.').matched is True
    assert match_post('Albany Pine Bush Preserve trail conditions after rain.').matched is True
    assert match_post('Sold out night at Proctors — what a show.').matched is True
    assert match_post('Proctors is a beautiful historic building downtown.').matched is False


def test_dutch_rotterdam_bethlehem_person_not_multi_local() -> None:
    dutch = match_post(
        'Gifmoorden lijken maar zelden voor te komen. Toch werden zowel Rotterdam '
        'als Den Haag deze zomer opgeschrikt door opzienbarende zaken. '
        'Toxicoloog Corine Bethlehem legt uit waarom gif zo’n angstaanjagend '
        'effectief moordmiddel kan zijn.'
    )
    assert dutch.matched is False
    assert dutch.reason in {
        'hard_negative:malta_europe',
        'hard_negative:bethlehem_person_name',
    }
    short = match_post('Rotterdam. Toxicoloog Corine Bethlehem legt uit.')
    assert short.matched is False
    assert short.reason == 'hard_negative:bethlehem_person_name'
    assert match_post('Town of Bethlehem NY board meeting tonight.').matched is True
    assert match_post('Tonight at Rotterdam Square Mall in Rotterdam, NY').matched is True


def test_galway_dublin_city_walk_list_not_galway_ny() -> None:
    walk = match_post(
        'Walked London, Birmingham, Glasgow, Edinburgh, NYC, DC, Berlin, '
        'Galway, Dublin, Paris, Wellfleet etc'
    )
    assert walk.matched is False
    assert walk.reason == 'hard_negative:galway_ireland'
    assert match_post('Galway NY high school soccer tonight').matched is True


def test_schenectady_avenue_not_city() -> None:
    ave = match_post('Schenectady Ave.')
    assert ave.matched is False
    assert ave.reason == 'hard_negative'
    assert match_post('Schenectady Avenue in Brooklyn').matched is False
    assert match_post('Schenectady was built by General Electric.').matched is True


def test_opera_saratoga_thespa_and_with_anticipation_recall() -> None:
    opera = match_post(
        'Mary Birnbaum, currently the General and Artistic Director of '
        "Saratoga Springs' Opera Saratoga, will transition to an advisory role."
    )
    assert opera.matched is True
    spa = match_post(
        'Last 3 Days of the Saratoga summer meet! Prints available. #Saratoga #TheSpa #horseracing'
    )
    assert spa.matched is True
    stakes = match_post(
        "Saratoga: Liam's Law holds off heavily favored stablemate to win the With Anticipation."
    )
    assert stakes.matched is True


def test_massry_harriman_quackenbush_peebles_corning_nysm_recall() -> None:
    assert (
        match_post('HVCC Foundation is acquiring the Massry Center for the Arts.').matched is True
    )
    assert match_post('Renovations continue at the Harriman Campus.').matched is True
    assert match_post('Brunch at Quackenbush Square this Saturday.').matched is True
    assert match_post('Peebles Island State Park hike tomorrow.').matched is True
    assert match_post('Walk along the Corning Preserve trail.').matched is True
    assert match_post('New exhibit opens at the New York State Museum.').matched is True


def test_nested_saratoga_springs_prefers_longer_token() -> None:
    """Nested saratoga ⊂ saratoga springs must not pick bare saratoga alphabetically."""
    both = match_post("Visiting Saratoga Springs' downtown shops near Broadway.")
    assert both.matched is False
    assert both.reason == 'ambiguous_no_context:saratoga springs'
    assert match_post('Saratoga Springs, NY city council meets Tuesday.').matched is True


def test_cardiff_wales_capital_region_not_ny() -> None:
    card = match_post(
        'Major infrastructure and regeneration projects across south-east Wales '
        'could gain access to investment expertise and capital under a new '
        "partnership with the UK's National Wealth Fund.",
        alt_text=(
            'Cardiff Capital Region to gain access to UK National Wealth Fund '
            'Nation.Cymru staff report.'
        ),
        author_handle='nation.cymru',
    )
    assert card.matched is False
    assert card.reason in {
        'hard_negative',
        'hard_negative:uk_wales_capital_region',
    }
    # Body "Capital Region" + Wales without the Cardiff phrase.
    wales = match_post(
        'Investment across south-east Wales for the Capital Region.',
    )
    assert wales.matched is False
    assert wales.reason == 'hard_negative:uk_wales_capital_region'
    handle_only = match_post(
        'A Capital Region partnership with the National Wealth Fund.',
        author_handle='nation.cymru',
    )
    assert handle_only.matched is False
    assert handle_only.reason == 'hard_negative:uk_wales_capital_region'
    assert (
        match_post('Anyone in the Capital District/Saratoga area, Mohawk Valley?').matched is True
    )


def test_fai_cup_waterford_galway_not_multi_local() -> None:
    fai = match_post(
        'Gonna be a Waterford/Galway final, calling it now. With the Blues to '
        'prevail and get our hands on only our 3rd FAI Cup.'
    )
    assert fai.matched is False
    assert fai.reason == 'hard_negative:galway_ireland'
    assert match_post('Drive from Galway to Waterford for the farmers market.').matched is True


def test_siege_of_troy_film_not_troy_ny() -> None:
    siege = match_post(
        'Did you ever feel that King of New York could have done with a bit more '
        'Wing Chun? That the siege of Troy might have been handled a bit more '
        'successfully if the Greeks had a bazooka?',
        alt_text='Father Joe Review – Venice Film Festival',
    )
    assert siege.matched is False
    assert siege.reason == 'hard_negative:troy_person_name'
    assert match_post('Dinner in Troy, New York tonight.').matched is True


def test_hopeful_stakes_and_worktab_saratoga_recall() -> None:
    hopeful = match_post(
        'Saratoga: The Good Life is 2-for-2 after his 4-length triumph in the '
        'Grade 1 Hopeful Stakes.'
    )
    assert hopeful.matched is True
    worktab = match_post(
        'Magnitude returned to the worktab at Saratoga on Sept. 6, breezing a '
        'half-mile in his first move since the Whitney.'
    )
    assert worktab.matched is True


def test_castor_troy_wrestling_not_multi_local() -> None:
    nxt = match_post(
        'More NXT Releases according to Pro Wrestling Illustrious:\n'
        '* Bryceton Cleese\n'
        '* Axel Rainstorm\n'
        '* Castor Troy\n'
        '* Bethlehem Jesusface\n'
        '* Oaklynnsleigh Zee'
    )
    assert nxt.matched is False
    assert nxt.reason == 'hard_negative:troy_person_name'
    assert match_post('Castor Street reopen in Troy, New York next week.').matched is True


def test_downtown_albany_and_troy_iron_nail_recall() -> None:
    downtown = match_post(
        'Investors are lining up apartment conversions of two downtown Albany buildings.'
    )
    assert downtown.matched is True
    iron = match_post('1800s Factory RUINS and FORGOTTEN Tunnel (Troy Iron and Nail Factory)')
    assert iron.matched is True


def test_america250_and_funny_cide_gio_ponti_saratoga_recall() -> None:
    america = match_post('I explore that question through Saratoga and America250.')
    assert america.matched is True
    funny = match_post('Scramjet won the Funny Cide Stakes on closing day.')
    assert funny.matched is True
    gio = match_post('Siyouincanada took the Gio Ponti on the last card of the meet.')
    assert gio.matched is True


def test_gio_ponti_designer_not_stakes() -> None:
    designer = match_post("L'objet design : la théière Aéro, le chef-d'oeuvre fuselé de Gio Ponti")
    assert designer.matched is False
    stakes = match_post('Gio Ponti Stakes drew a full field on closing day.')
    assert stakes.matched is True


def test_saratoga_national_historical_park_and_battlefield_landmarks() -> None:
    park = match_post(
        "On Sept. 19, 1777, Burgoyne's army fought at Freeman's Farm. "
        'Walk the ground today at Saratoga National Historical Park.'
    )
    assert park.matched is True
    assert match_post('Tour Bemus Heights before the evening lecture.').matched is True
    assert match_post('Bemis Heights overlooks the Hudson battlefield trail.').matched is True


def test_funny_bones_albany_comedy_recall() -> None:
    show = match_post('Omg we saw comedian Trae Crowder at Funny Bones in Albany- we howled.')
    assert show.matched is True
    # Other-city Funny Bone franchises without Cap Region towns stay out.
    other = match_post('Caught a set at Funny Bones in Dayton last night.')
    assert other.matched is False


def test_philippines_metro_manila_capital_region_not_ny() -> None:
    manila = match_post(
        'Jobless rate climbs as growth slows.',
        alt_text=(
            'Philippine jobless rate surges to 4-year high as growth slows '
            'The capital region of Metro Manila logged the highest jobless rate '
            'of 8.2 per cent.'
        ),
    )
    assert manila.matched is False
    assert manila.reason in {
        'hard_negative',
        'hard_negative:philippines_capital_region',
    }
    assert (
        match_post(
            'Capital Region students visited Metro Manila on exchange before '
            'returning to #AlbanyNY.'
        ).matched
        is True
    )


def test_florida_crtpa_capital_region_not_ny() -> None:
    crtpa = match_post(
        'Today we visited the Orchard Pond Greenway with the team from Capital '
        'Region Transportation Planning Agency to look at SunTrail upgrades. '
        'CRTPA will hold another public meeting soon.'
    )
    assert crtpa.matched is False
    assert crtpa.reason in {
        'hard_negative',
        'hard_negative:florida_crtpa_capital_region',
    }
    orchard = match_post(
        'Capital Region trail planners toured Orchard Pond Greenway near Tallahassee.'
    )
    assert orchard.matched is False
    assert orchard.reason == 'hard_negative:florida_crtpa_capital_region'
    assert match_post('Capital Region trail planners met in Saratoga County.').matched is True


def test_clark_hall_not_lark_hall() -> None:
    guelph = match_post(
        '33 years ago today Fugazi played Peter Clark Hall University of Guelph, '
        'Guelph, ON, Canada with Burn 51 and Shudder to Think.',
        alt_text='location map',
    )
    assert guelph.matched is False
    assert match_post('Albany: Helmet @ Lark Hall this Friday.').matched is True


def test_travers_brothers_not_travers_stakes() -> None:
    creed = match_post(
        'Edward recruits Lucy to the Jackdaw and helps Assassins Rhona Dinsmore '
        'and the Travers brothers.',
        alt_text="Assassin's Creed: Black Flag Resynced | Part 5",
    )
    assert creed.matched is False
    assert (
        match_post("It's Travers Day at Saratoga. The main event is at 6:35 p.m.").matched is True
    )


def test_stillwater_ok_acars_not_stillwater_ny() -> None:
    acars = match_post(
        'Air to Ground Message: NYC CREW who got taken off. '
        'Area: Stillwater, OK, USA Type: Airbus A319'
    )
    assert acars.matched is False
    assert acars.reason == 'hard_negative:stillwater_ok'
    assert match_post('Stillwater, NY town board meets Tuesday.').matched is True


def test_erasmusbrug_rotterdam_not_rotterdam_ny() -> None:
    dutch = match_post(
        'Brooklyn Bridge en Ponte Vecchio naast bruggetje over de Dommel',
        alt_text=(
            'De Ponte Vecchio in Florence, de Brooklyn Bridge in New York, '
            'het viaduct in Millau en de Erasmusbrug in Rotterdam: wat heeft '
            'dat te maken met het simpele nieuwe bruggetje over de Dommel in '
            'Den Bosch?'
        ),
    )
    assert dutch.matched is False
    assert dutch.reason == 'hard_negative:malta_europe'
    assert match_post('New bakery opens in Rotterdam, NY this weekend.').matched is True


def test_tipsy_taco_latham_recall() -> None:
    tipsy = match_post(
        'Tipsy Taco Cantina in Latham is getting a different name and a new menu.',
        alt_text='Latham restaurant to get different name, new menu',
    )
    assert tipsy.matched is True


def test_monica_latham_routledge_not_latham_ny() -> None:
    academic = match_post(
        'Publication: Monica Latham, "Virginia Woolf in the French Imagination". '
        'London; New York: Routledge, 2026. ISBN 9781032878904'
    )
    assert academic.matched is False
    assert academic.reason == 'hard_negative:latham_person_name'
    assert match_post('New cafe opens in Latham, NY next to the circle.').matched is True


def test_loudonville_nws_cleveland_ashland_not_ny() -> None:
    nws = match_post(
        'Special Weather Statement issued September 9 at 6:03PM EDT by NWS Cleveland OH '
        'At 603 PM EDT, Doppler radar was tracking a strong thunderstorm over '
        'Loudonville, or 16 miles south of Ashland, moving east at 40 mph.'
    )
    assert nws.matched is False
    assert nws.reason == 'hard_negative:loudonville_oh'
    northeast = match_post(
        'Loudonville Dog Breeder Hit With 10 New Charges, Including Animal Cruelty '
        '#NortheastOhio #News'
    )
    assert northeast.matched is False
    assert northeast.reason == 'hard_negative:loudonville_oh'
    assert match_post('Road work begins in Loudonville near Albany this week.').matched is True


def test_rotterdam_filmmaker_venice_not_rotterdam_ny() -> None:
    film = match_post(
        'Educated in Moscow and then New York, Georgian filmmaker Levan Kogashvili '
        'was discovered in Rotterdam in 2010 with his debut film. After winning awards '
        'at Tribeca and Jeddah, he is presenting Guria at #Venice2026.'
    )
    assert film.matched is False
    assert film.reason == 'hard_negative:malta_europe'
    assert match_post('New bakery opens in Rotterdam, NY this weekend.').matched is True


def test_capitaldistrict_hashtag_and_local_festival_recall() -> None:
    assert (
        match_post(
            'Bus trip from #GlensFalls to #Albany for the show. #capitaldistrict #northcountry'
        ).matched
        is True
    )
    assert match_post('Autumn Glow Festival opens in Rotterdam').matched is True
    assert match_post('Giant Saratoga Pumpkinfest set to stage for 11th year').matched is True
    assert (
        match_post(
            "Recovery Sports Grill on New Scotland Ave has been Albany's go-to sports bar"
        ).matched
        is True
    )
    assert (
        match_post('Second venture for Albany Ale & Oyster owners offers neighborhood feel').matched
        is True
    )
    assert (
        match_post('Man rescued on Thruway after dump truck overturns in Rotterdam').matched is True
    )
    assert match_post("From Scotia-Glenville's signing ceremony this week").matched is True
    assert (
        match_post("It's Whitney Day at Saratoga, and first post is about 40 minutes away.").matched
        is True
    )
    assert (
        match_post(
            'Saratoga Springs traveling to Christian Brothers Academy in a rematch '
            'of the Class AA sectional championship game.'
        ).matched
        is True
    )
    assert (
        match_post(
            "Amid NYSDOH's investigation into the Albany Center for Independent Living "
            'and Delmar Center for Rehabilitation and Nursing'
        ).matched
        is True
    )


def test_route_787_takoma_park_md_not_albany_i787() -> None:
    md = match_post(
        'Twice-yearly #TakomaPark check: new (9/6/2026) Ride On Route 18 schedule '
        'still shows Flower Avenue as Route 787. It is no longer a Maryland state route.\n'
        'assets.montgomerycountymd.gov/files/2026-0...'
    )
    assert md.matched is False
    assert md.reason == 'hard_negative:route_787_md'
    assert match_post('Crash on I-787 northbound near downtown Albany.').matched is True
    assert match_post('Construction on Route 787 in Albany NY this weekend.').matched is True


def test_albany_sheriff_library_semiconductor_and_saratoga_dining_recall() -> None:
    assert (
        match_post(
            "ALBANY SHERIFF'S INVESTIGATOR Busted for Allegedly Using Flock Cameras "
            'to Track Ex-girlfriend 3,000 TIMES'
        ).matched
        is True
    )
    assert (
        match_post(
            'Tattoo artist talk about Japanese Yokai drawing at the Albany public library'
        ).matched
        is True
    )
    assert (
        match_post(
            "An auction of a defunct semiconductor startup's equipment is on hold after "
            "the firm's Albany landlord sought protections from potential damage."
        ).matched
        is True
    )
    assert (
        match_post(
            "It's always exciting when a new restaurant comes to Saratoga. But when "
            "Noah's Italian and Bear's Cup Bakehouse opened within about a week…"
        ).matched
        is True
    )


def test_venezuela_capital_region_not_ny() -> None:
    venezuela = match_post(
        'How Organized Communities Are Rebuilding in the Wake of Venezuela’s Twin Earthquakes',
        alt_text=(
            'At 3:55 a.m. on June 25, a flicker of cellular service allowed Carlos '
            'to send word from La Guaira that he and his family had survived the '
            'twin earthquakes that shook Venezuela’s capital region hours earlier.'
        ),
    )
    assert venezuela.matched is False
    assert venezuela.reason in {
        'hard_negative',
        'hard_negative:venezuela_capital_region',
    }
    assert (
        match_post(
            'Capital Region students visited Caracas after studying Venezuela’s '
            'capital region, then returned to #AlbanyNY.'
        ).matched
        is True
    )


def test_troy_conner_spartanburg_not_troy_ny() -> None:
    sc = match_post(
        'Seventh Circuit Solicitor Barry Barnette explained Christopher Kastner '
        'tracked his wife to the tattoo shop, attacked her, and then shot Troy Conner.',
        alt_text=(
            'Suspect accused of Upstate tattoo shop murder denied bond An arrest '
            'has been made in a Spartanburg County shooting late Thursday night '
            'that has left a man dead.'
        ),
    )
    assert sc.matched is False
    assert sc.reason in {
        'hard_negative:troy_sc',
        'hard_negative:troy_person_name',
    }
    assert match_post('Dinner in Troy, New York tonight.').matched is True


def test_bethlehem_area_lehigh_valley_not_town_of_bethlehem() -> None:
    area = match_post(
        'Bethlehem Area students remember 9/11 first responders by climbing 110 flights of steps',
        alt_text=(
            'Over 2,000 Liberty and Freedom high school students got together '
            'Friday to climb 110 flights of steps, in honor of the floors New York '
            'firefighters climbed on Sept. 11, 2001.'
        ),
    )
    assert area.matched is False
    assert area.reason == 'hard_negative:bethlehem_pa'
    lehigh = match_post(
        'From my longtime news home... #LehighValley #Allentown #Bethlehem #Easton #Pa07 mcall.com'
    )
    assert lehigh.matched is False
    assert lehigh.reason == 'hard_negative:bethlehem_pa'
    assert match_post('Town of Bethlehem, NY board meeting tonight.').matched is True


def test_brook_tavern_stillwater_burgoyne_and_section2_recall() -> None:
    assert match_post('Brook Tavern in Saratoga is sold, minimal changes planned.').matched is True
    assert (
        match_post(
            'On Sept. 11, 1777, the American army waited at Stillwater for Burgoyne.'
        ).matched
        is True
    )
    assert (
        match_post(
            'Stand at Stillwater today and the landscape looks quiet. In Sept. 1777, '
            'thousands of soldiers were preparing for battle.'
        ).matched
        is True
    )
    assert match_post('Section 2 football: Shaker vs Colonie at 7 tonight.').matched is True
    assert match_post('Pine Bush Observatory open house this weekend.').matched is True
    assert match_post('Town of Halfmoon board meeting tonight.').matched is True


def test_rotterdam_netherlands_oil_exchange_not_ny() -> None:
    nl = match_post(
        'Ab Montag wird mit massivem Anstieg des Ölpreises gerechnet an den Börsen in '
        'New York in den USA u. Rotterdam in den Niederlanden u. Frankfurt am Main '
        'in der BRD!'
    )
    assert nl.matched is False
    assert nl.reason == 'hard_negative:malta_europe'
    assert match_post('Town board meeting in Rotterdam, NY tonight.').matched is True
    assert match_post("BJ's Warehouse Club in Rotterdam is hiring.").matched is True


def test_town_of_malta_cdta_and_local_venue_recall() -> None:
    assert match_post('Town of Malta community day this Saturday.').matched is True
    assert match_post('GlobalFoundries Malta campus hiring fair next week.').matched is True
    assert match_post('CDTA route 12 schedule changes Monday.').matched is True
    assert match_post('Howe Caverns school field trip Friday.').matched is True
    assert match_post('USS Slater destroyer escort tours this Sunday.').matched is True
    assert match_post('LarkFest street festival this weekend.').matched is True
    assert match_post('Pearlpalooza downtown this weekend.').matched is True
    assert match_post('Five Rivers Environmental Education Center programs.').matched is True
    assert match_post('Indian Ladder Trail open for hiking.').matched is True
    assert match_post('Thacher Park overlook hike Sunday.').matched is True
    assert match_post('Collar City Tweed Ride Sunday.').matched is True
    assert match_post('Canfield Casino ghost tour tickets.').matched is True
    assert match_post('Vischer Ferry firefighters respond to a garage fire.').matched is True
    assert match_post('Hart Cluett Museum exhibit opens.').matched is True
    assert match_post('Burden Iron Works museum Saturday.').matched is True
    assert match_post('Bombers Burrito Bar downtown special.').matched is True
    assert match_post('WAMC Roundtable tomorrow morning.').matched is True
    assert match_post('Siena Saints tip off tonight.').matched is True
    assert match_post('Secret Caverns after dark tour.').matched is True


def test_bethlehem_universal_hall_houston_institute_stillwater_recall() -> None:
    assert match_post('Town of Bethlehem board meeting tonight on Delaware Ave.').matched is True
    assert match_post('Bethlehem Public Library hosts author night.').matched is True
    assert (
        match_post(
            'Wish Benefit Tour at Universal Preservation Hall in Saratoga Springs on October 6.'
        ).matched
        is True
    )
    assert match_post("Show at RPI's Houston Field House tonight.").matched is True
    assert match_post('New show at the Albany Institute of History & Art.').matched is True
    assert (
        match_post('America250 events in Stillwater along the Hudson this weekend.').matched is True
    )
    assert match_post('See you at Washington Park farmers market Saturday.').matched is True


def test_loudonville_mid_ohio_county_flood_not_ny() -> None:
    flood = match_post(
        'Flood Advisory in effect until 11:45 PM for Ashland, Holmes, Knox, Morrow, and '
        'Richland Counties. Avoid flooded roads near Mansfield, Loudonville, Mount Gilead.'
    )
    assert flood.matched is False
    assert flood.reason == 'hard_negative:loudonville_oh'
    severe = match_post(
        'Severe Thunderstorm Warning for Ashland, Holmes, Knox, and Richland Counties '
        'with rotation near Loudonville affecting Perrysville and Bellville.'
    )
    assert severe.matched is False
    assert severe.reason == 'hard_negative:loudonville_oh'
    assert match_post('Road work begins in Loudonville near Albany this week.').matched is True


def test_ireland_metrolink_capital_region_not_ny() -> None:
    metro = match_post(
        'Irish Government has approved MetroLink for detailed tendering. The line will '
        'link Swords and Dublin city centre, improving links across the capital region.'
    )
    assert metro.matched is False
    assert metro.reason == 'hard_negative:ireland_capital_region'
    assert (
        match_post('Join us for Capital Region Voter Registration Day in Albany.').matched is True
    )


def test_galway_girl_song_not_galway_ny() -> None:
    song = match_post(
        'R.E.M. - Leaving New York / Ed Sheeran - Galway Girl / The Police - Reggatta De Blanc'
    )
    assert song.matched is False
    assert song.reason == 'hard_negative:galway_ireland'
    assert match_post('Town of Galway NY hosts a farmers market this Saturday.').matched is True


def test_troy_nader_and_troy_davis_person_names_not_city() -> None:
    attorney = match_post(
        'Immigration Attorney Troy Nader Moslemi at Chinese Community event in NYC.'
    )
    assert attorney.matched is False
    assert attorney.reason == 'hard_negative:troy_person_name'
    heisman = match_post(
        'Rand-O thoughts on Iowa State, Eddie George, Troy Davis — and Paul McCartney. '
        'A 1995 weekend in downtown New York City.'
    )
    assert heisman.matched is False
    assert heisman.reason == 'hard_negative:troy_person_name'
    assert match_post('New cafe opens in Troy, NY near the waterfront.').matched is True


def test_stillwater_township_nj_not_stillwater_ny() -> None:
    nj = match_post(
        'CBS News New York: road rage on Millbrook Road in Stillwater Township. '
        'Sussex County, N.J. chair disputes allegations.'
    )
    assert nj.matched is False
    assert nj.reason == 'hard_negative:stillwater_nj'
    assert (
        match_post('Town of Stillwater hosts America250 celebration on Burgoyne Avenue.').matched
        is True
    )


def test_times_union_photo_credit_not_local_paper() -> None:
    credit = match_post(
        'Flock might be a tipping point in a broader anti-tech movement. '
        'Is Flock spying on you? (Jim Franco/Times Union/Getty)'
    )
    assert credit.matched is False
    assert credit.reason == 'hard_negative:times_union_photo_credit'
    article = match_post(
        'Times Union coverage: Common Council debates housing on Lark Street. '
        'Photo (Jim Franco/Times Union/Getty)'
    )
    assert article.matched is True


def test_albany_corporation_counsel_and_newyork_hashtag_recall() -> None:
    counsel = match_post(
        "John Reilly Jr. has been named as Albany's corporation counsel, setting up a "
        'Common Council confirmation process that will likely focus on his experience.'
    )
    assert counsel.matched is True
    sand = match_post(
        "Outdoor #News: #NewYork's Saratoga Sand Plains WMA is #Northern Zone hunting "
        'with a #Southern Zone feel'
    )
    assert sand.matched is True


def test_albany_avenue_crown_heights_not_city() -> None:
    nyc = match_post(
        'NYC yellow school bus slams apartment building during morning rush. '
        'Eastern Parkway near Albany Avenue in Crown Heights around 7:20 a.m.'
    )
    assert nyc.matched is False
    assert nyc.reason == 'hard_negative'
    assert match_post('Crash near Hackett Blvd in #AlbanyNY').matched is True


def test_curtain_call_theatre_and_colonie_local_cues() -> None:
    curtain = match_post(
        "Curtain Call Theater in Latham has a brilliant production of Delia Ephron's "
        'play Left on 10th showing through October 4th.'
    )
    assert curtain.matched is True
    assert match_post('Left on 10th at Curtain Call Theatre through October 4th').matched is True
    towers = match_post(
        'Six people have been sent to the hospital after a fire at the Towers of '
        'Colonie on Thursday'
    )
    assert towers.matched is True
    assert towers.reason == 'colonie_local'
    south = match_post('South Colonie CSD is looking for businesses to host Toys for Tots drives.')
    assert south.matched is True
    assert south.reason == 'colonie_local'


def test_bethlehem_town_board_not_person_or_star_of() -> None:
    board = match_post(
        'Tempers run hot at Town Board meeting as land-use battle drags on in '
        'Bethlehem with residents and Town Board divided over solutions.'
    )
    assert board.matched is True
    star = match_post(
        'A little waterfall of grass lily, star of Bethlehem & a little waterfall, '
        'both in the New York Botanical Garden'
    )
    assert star.matched is False
    assert star.reason == 'hard_negative:bethlehem_star_of'


def test_thespa_gunma_not_saratoga_spa() -> None:
    jp = match_post('【新生日本代表】#fcgifu #fctokyo #giravanz #hollyhock #thespa #trinita #verdy')
    assert jp.matched is False
    spa = match_post(
        'Last 3 Days of the Saratoga summer meet! Prints available. #Saratoga #TheSpa #horseracing'
    )
    assert spa.matched is True


def test_ichabod_crane_literary_not_school() -> None:
    lyric = match_post("Someday I'll win I'm not Ichabod Crane And though dark my days")
    assert lyric.matched is False
    merch = match_post(
        'Headless Horseman Ichabod Crane greeting card #Halloween #SleepyHollow #classic'
    )
    assert merch.matched is False
    school = match_post(
        'Ichabod Crane Central School district board members say they will give public '
        'financial updates, with Valatie residents wanting more answers.'
    )
    assert school.matched is True


def test_globalfoundries_wordcloud_not_malta_campus() -> None:
    cloud = match_post("Bluesky's Top 10 Trending Words: ukraine, globalfoundries, monaco, malta")
    assert cloud.matched is False
    campus = match_post('GlobalFoundries Malta campus hiring fair next week.')
    assert campus.matched is True
    ny = match_post('GlobalFoundries secures a federal award to expand manufacturing in Malta, NY.')
    assert ny.matched is True


def test_troy_kingston_person_not_city() -> None:
    person = match_post('Troy Kingston: Fighting for real New Yorkers with his viral videos')
    assert person.matched is False
    assert person.reason == 'hard_negative:troy_person_name'
    assert match_post('Downtown Troy hosts a First Friday art walk tonight.').matched is True


def test_frear_park_downtown_troy_crossings_colonie_recall() -> None:
    frear = match_post(
        'Two developers are seeking tax exemptions to build a 72-unit apartment complex '
        'off Oakwood Avenue near the entrance to Frear Park in Troy.'
    )
    assert frear.matched is True
    downtown = match_post(
        "One of downtown Troy's most prominent buildings will be in the hands of a "
        'lender following a foreclosure auction Friday.'
    )
    assert downtown.matched is True
    crossings = match_post(
        'Colonie will be welcoming the change of seasons with the return of their '
        'annual Harvest Fest at the Crossings of Colonie on September 26.'
    )
    assert crossings.matched is True
    officials = match_post(
        'Troy city officials confirmed that a teen was injured in an e-bike-involved '
        'crash on Thursday night near Fourth Avenue and Monroe Avenue.'
    )
    assert officials.matched is True
    # Comma before "New York Botanical Garden" must not look like "Bethlehem, NY".
    star_comma = match_post('grass lily, star of Bethlehem, New York Botanical Garden')
    assert star_comma.matched is False
    assert star_comma.reason == 'hard_negative:bethlehem_star_of'
    assert match_post('Town of Bethlehem Public Library book sale this weekend.').matched is True


def test_bethlehem_holy_land_carol_not_town() -> None:
    carol = match_post(
        "I'm trying to figure out why this morning's earworm is O Little Town Of Bethlehem"
    )
    assert carol.matched is False
    assert carol.reason == 'hard_negative:bethlehem_holy_land'
    palestine = match_post(
        "Park Slope farmers market selling olive oil and za'atar from Bethlehem, Palestine. "
        '#nyc #palestine'
    )
    assert palestine.matched is False
    assert palestine.reason == 'hard_negative:bethlehem_holy_land'
    assert match_post('Bethlehem Town Board voted on the budget at Town Hall.').matched is True


def test_newtonville_village_day_not_colonie() -> None:
    village = match_post(
        '[PHOTOS] Newtonville Village Day and the dedication of Setti D. Warren Plaza. '
        'Newton turned out Sunday. Mayor Marc Laredo and John Kerry were on hand beside '
        'the Austin Street development.'
    )
    assert village.matched is False
    assert village.reason == 'hard_negative:newtonville_ma'
    assert (
        match_post('New shops opening on Newtonville Avenue in Colonie near #AlbanyNY.').matched
        is True
    )


def test_cdta_algeria_not_capital_district_transit() -> None:
    algeria = match_post(
        'Le CDTA produit désormais des puces électroniques entièrement conçues en Algérie.'
    )
    assert algeria.matched is False
    assert algeria.reason == 'hard_negative:cdta_algeria'
    assert match_post('CDTA route 12 schedule changes Monday.').matched is True


def test_blue_collar_city_not_troy_nickname() -> None:
    browns = match_post(
        "Congratulations Cleveland Browns! I'll root for any blue collar city, "
        'especially an underdog.'
    )
    assert browns.matched is False
    assert match_post('Collar City Tweed Ride Sunday.').matched is True


def test_rotterdam_denhaag_hashtags_not_ny() -> None:
    dutch = match_post('Eindelijk naar bed Een #Rotterdam #DenHaag #Curaçao #NewYork Big Apple dag')
    assert dutch.matched is False
    assert dutch.reason == 'hard_negative:malta_europe'
    assert match_post('Mabee Farm Autumn Glow Festival in Rotterdam Junction.').matched is True


def test_albany_democrats_hochul_and_war_room_recall() -> None:
    dems = match_post('HOCHUL to join Albany Democrats at Saturday campaign rally.')
    assert dems.matched is True
    assert dems.reason == 'strong_positive'
    tavern = match_post(
        "Mets legend resurrects iconic Albany tavern: 'A real save'. "
        'The War Room Tavern, a bar hot spot for lawmakers in the capital, reopens.'
    )
    assert tavern.matched is True


def test_sooke_firesmoke_capital_region_not_ny() -> None:
    sooke = match_post(
        'South Island friends should batten down — the wildfire north of Sooke grew overnight.',
        alt_text='firesmoke.ca predicting the smoke to drift over capital region tonight.',
    )
    assert sooke.matched is False
    assert sooke.reason == 'hard_negative:canadian_capital_region'
    local = match_post('Capital Region air quality advisory for #AlbanyNY this evening.')
    assert local.matched is True


def test_times_union_busting_not_local_paper() -> None:
    bingo = match_post(
        'This seems like a good format for the current times',
        alt_text='Union Busting BINGO YouTube video by Sam Maxis',
    )
    assert bingo.matched is False
    assert match_post('Times Union coverage of downtown #AlbanyNY redevelopment.').matched is True


def test_brunswick_boat_manufacturer_not_town_ny() -> None:
    boats = match_post(
        'Dubliner named chief executive of New York-listed boat manufacturer Brunswick'
    )
    assert boats.matched is False
    assert boats.reason == 'hard_negative:brunswick_corp'
    assert match_post('Town of Brunswick, NY board meeting tonight.').matched is True


def test_malta_independence_day_not_town_ny() -> None:
    indep = match_post(
        'Malta Independence Day\nNew York City Appreciation day\nArmenia Independence day'
    )
    assert indep.matched is False
    assert indep.reason == 'hard_negative:malta_europe'
    assert match_post('Concert tonight at the Malta Amphitheater, Malta, NY.').matched is True


def test_schuylerville_town_hall_and_central_warehouse_recall() -> None:
    hall = match_post(
        "What can Saratoga's landscape tell us about the Revolution? "
        'Thursday at 7 p.m. at Saratoga Town Hall in Schuylerville.'
    )
    assert hall.matched is True
    assert hall.reason == 'strong_positive'
    warehouse = match_post(
        "Asbestos remediation has been completed at Albany's Central Warehouse. "
        "Now the city's longstanding eyesore is entering its final stages of demolition."
    )
    assert warehouse.matched is True
    assert warehouse.reason == 'strong_positive'


def test_french_colonie_tech_praxis_not_town_ny() -> None:
    praxis = match_post(
        'La startup Praxis annonce vouloir installer sa colonie tech en Uruguay. '
        'Le retour des néocolons tech.',
        alt_text=(
            'Praxis, a New York-based crypto-backed digital nation, has signed a deal '
            'to build a physical city in Uruguay.'
        ),
    )
    assert praxis.matched is False
    assert match_post('Town of Colonie police responded on Central Avenue.').matched is True


def test_ancient_troy_turkey_not_city_of_troy_ny() -> None:
    ancient = match_post(
        'Troy: 2,800-year-old market discovered. The ancient city of Troy, located in '
        'northwestern Turkey, continues to hold surprises for researchers.'
    )
    assert ancient.matched is False
    assert ancient.reason in {
        'hard_negative:troy_person_name',
        'hard_negative:troy_ancient',
        'ambiguous_no_context:troy',
    }
    assert match_post('City of Troy announces downtown paving for October.').matched is True


def test_drupalcon_rotterdam_not_town_ny() -> None:
    conf = match_post(
        'With DrupalCon Rotterdam approaching, plan your week. '
        '#Drupal #DrupalConRotterdam\nPhoto credits: Drupal AI Summit NYC 2026'
    )
    assert conf.matched is False
    assert conf.reason == 'hard_negative:malta_europe'
    assert match_post('On Pangburn Rd Rotterdam New York').matched is True


def test_idiomatic_five_rivers_not_nature_center() -> None:
    idiom = match_post('I just want to give him this box so I can go home and pee five rivers.')
    assert idiom.matched is False
    nature = match_post('Hike the trails at Five Rivers Environmental Education Center in Delmar.')
    assert nature.matched is True


def test_freeman_farm_curly_apostrophe_and_albany_edu_recall() -> None:
    curly = match_post(
        'On Sept. 19, 1777, Burgoyne won Freeman’s Farm after seven hours of fighting. '
        'He did not open the road to Albany.'
    )
    assert curly.matched is True
    assert curly.reason == 'strong_positive'
    campus = match_post('Pathogen morphology research continues at www.albany.edu/cihs/faculty...')
    assert campus.matched is True
    assert campus.reason == 'strong_positive'
