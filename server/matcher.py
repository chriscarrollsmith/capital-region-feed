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
        albany\s*,?\s*(?:ny|n\.y\.|new\s+york)
      | \#albanyny\b
      | \#albany_ny\b
      # Word-boundary after region/district — "capital regional" (ES/LatAm) is not NY.
      | (?:new\s+york(?:'s)?\s+)?capital\s+(?:region|district)\b
      | greater\s+albany
      | albany\s+county
      | rensselaer\s+county
      | schenectady\s+county
      | saratoga\s+county
      # Word-boundary: reject hashtag stuffing like #schenectadyparkcleanup.
      # Negative lookahead: cuisine "Schenectady-style" is not the city.
      | \bschenectady\b(?![\s-]+style\b)
      | \bguilderland\b
      | \bniskayuna\b
      | \bwatervliet\b
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
      # NWS / scanner copy often abbreviates East/North Greenbush.
      # Leading word boundary: "white Greenbush Bakery" must not match "e Greenbush".
      | \b(?:e\.?|east)\s+greenbush
      | \b(?:n\.?|north)\s+greenbush
      | mechanicville
      # Word boundary after "hills" so "Burnt hillside" (wildfire terrain) does not match.
      | burnt\s+hills\b
      | ballston\s+spa
      # Cap Region Clifton Park — UK cricket ground gated in conflict helper.
      | clifton\s+park
      | averill\s+park
      # Wire datelines often use "N.Y." with periods.
      | saratoga\s+springs\s*,?\s*(?:ny|n\.y\.|new\s+york)
      # Require a word boundary after "ny" so "Troy Nyhammer" does not match.
      | (?<!\d\s)(?<![\w.])troy(?![\w@.-])\s*,?\s*(?:ny\b|n\.y\.|new\s+york)
      | boght\s+corners
      | newtonville
      | \bcoeymans\b
      # Word boundary: Belgian sculptor "van Helderbergh" must not match.
      | \bhelderbergs?\b
      # Cap Region Altamont cues only — bare "Altamont" stays ambiguous (CA festival / band names).
      | \baltamont-based\b
      | altamont\s+fair
      | altamont(?:\s*,)?\s*ny\b
      | empire\s+state\s+plaza
      | albany\s+capital\s+center
      | albany\s+mayor\b
      | university\s+at\s+albany
      | \bualbany\b
      | suny\s+albany
      # Albany music venue — often listed as "Albany: … @ Lark Hall" without ", NY".
      | lark\s+hall\b
      # Capital Repertory Theatre (Albany) — hashtag #CapitalRep often omits venue cues.
      | \#?capitalrep\b
      | capital\s+rep(?:ertory)?\b
      | proctors\s+collaborative\b
      # Interstate 787 — require I-/interstate/route prefix. Bare "on 787" matches
      # medical PR ("Study on 787 Brain Tumor Patients").
      | \b(?:i-?|interstate\s+|route\s+|ny\s+)787\b
      | \bon\s+i-?787\b
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
      # Travers / Midsummer Derby / "Saratoga feature" wires often omit ", NY".
      | travers\s+stakes\b
      | travers\s+(?:day|weekend)\b
      | \bthe\s+travers\b
      | \bahead\s+of\s+(?:the\s+)?travers\b
      | \bwins?\s+(?:the\s+)?travers\b
      | travers\s+at\s+saratoga\b
      | midsummer\s+derby\b
      | saratoga\s+feature\b
      # Revolutionary War campaign / battlefield copy often omits ", NY".
      | saratoga\s+campaign\b
      | battles?\s+of\s+saratoga\b
      # Named Grade 1 / meet stakes at the Race Course often omit ", NY".
      | (?:h\.?\s*allen\s+)?jerkens(?:\s+memorial)?\b
      | \bgrade\s+[123i]+\s+forego\b
      | \b(?:the\s+)?forego\s+stakes\b
      | \bin\s+(?:the\s+)?forego\b
      | \bballerina\s+stakes\b
      | personal\s+ensign(?:\s+stakes)?\b
      | \bskidmore(?:\s+stakes)?\b[\s\S]{0,80}\bsaratoga\b
      | \bsaratoga\b[\s\S]{0,80}\bskidmore(?:\s+stakes)?\b
      # "Grade 1 runners at Saratoga" wires omit the word "stakes".
      | grade\s+[123i]+\b[\s\S]{0,80}\bat\s+saratoga\b
      | at\s+saratoga\b[\s\S]{0,80}grade\s+[123i]+\b
      # Win / victory wires: "6,000 wins … at Saratoga" without Race Course.
      | \b(?:wins?|won|victory)\b[\s\S]{0,80}\bat\s+saratoga\b
      | \bat\s+saratoga\b[\s\S]{0,80}\b(?:wins?|won|victory)\b
      # Legal Aid / county-exec / South Albany Airport copy often omits ", NY".
      | albany\s*/\s*amsterdam\b
      | albany\s+exec(?:utive)?\b
      | albany\s+county\s+exec(?:utive)?\b
      | south\s+albany\b
      # NWS / storm copy: Corinth NY cluster with Saratoga Springs (no ", NY").
      | \bcorinth\b[\s\S]{0,100}saratoga\s+springs\b
      | saratoga\s+springs\b[\s\S]{0,100}\bcorinth\b
      # SPAC / #SPAC next to Saratoga (avoid bare SPAC webinars without venue).
      | \#?spac\b[\s\S]{0,200}\bsaratoga\b
      | \bsaratoga\b[\s\S]{0,200}\#?spac\b
      # Amtrak station / jazz festival often omit ", NY".
      | amtrak[\s\S]{0,80}saratoga\s+springs\b
      | saratoga\s+springs[\s\S]{0,80}amtrak
      | saratoga\s+jazz\s+festival
      # Tourism / race-week phrasing often omits ", NY".
      | play\s+the\s+ponies[\s\S]{0,80}\bsaratoga\b
      | \bsaratoga\b[\s\S]{0,80}play\s+the\s+ponies
      # Thoroughbred / OTTB / Fasig-Tipton sales often omit ", NY".
      # Require word boundaries so #delmarthoroughbredclub cannot pair with #saratoga.
      | (?:\bthoroughbreds?\b|\#?ottb\b|\baftercare\b)[\s\S]{0,100}saratoga(?:\s+springs)?\b
      | saratoga(?:\s+springs)?\b[\s\S]{0,100}(?:\bthoroughbreds?\b|\#?ottb\b|\baftercare\b)
      | fasig[-\s]?tipton[\s\S]{0,40}\bsaratoga\b
      | \bsaratoga\b[\s\S]{0,40}fasig[-\s]?tipton
      # Graded stakes copy often says "at Saratoga" without Springs/NY.
      | \bstakes\b[\s\S]{0,80}\bat\s+saratoga\b
      | at\s+saratoga\b[\s\S]{0,80}\bstakes\b
      # Battlefield NPS / tourism often omits ", NY".
      | saratoga\s+battlefield\b
      # Graded race titles / starts often omit "stakes" beside Saratoga.
      | christophe\s+clement[\s\S]{0,80}\bsaratoga\b
      | \bsaratoga\b[\s\S]{0,80}christophe\s+clement
      | glens\s+falls[\s\S]{0,80}\bsaratoga\b
      | \bsaratoga\b[\s\S]{0,80}glens\s+falls
      | \#saratogaracing\b
      | \#saratoga\b[\s\S]{0,120}\b(?:race|racing|stakes|turf|odds|handicap)\b
      | \b(?:race|racing|stakes|turf|odds|handicap)\b[\s\S]{0,120}\#saratoga\b
      # Race cards often say "6th race at Saratoga" or "Stakes Preview @ Saratoga".
      | \brace\b[\s\S]{0,40}\bat\s+saratoga\b
      | at\s+saratoga\b[\s\S]{0,40}\brace\b
      | \bstakes\b[\s\S]{0,80}@\s*saratoga\b
      | @\s*saratoga\b[\s\S]{0,80}\bstakes\b
      # Cross-country pick cards: "Saratoga – Race 5" (en/em dash or colon).
      | \bsaratoga\b[\s]*[\u2013\u2014\-:]\s*race\s*\d
      # DRF / stakes wires omit "the" ("Grade 2 Saratoga Special").
      | (?:the\s+|grade\s+[12]\s+)?saratoga\s+special\b
      | battles?\s+of\s+saratoga\b
      # America 250 / Revolutionary War copy often omits ", NY".
      | \b1777\b[\s\S]{0,100}\bsaratoga\b
      | \bsaratoga\b[\s\S]{0,100}\b1777\b
      # Race-meet wires often omit ", NY" ("remainder of Saratoga meet").
      | \bsaratoga\s+meet\b
      # Harness track / barn-fire wires often omit ", NY".
      | saratoga\s+springs\s+harness\s+track
      | harness\s+track[\s\S]{0,40}saratoga\s+springs
      # Thacher State Park / WildPlay adventure copy often omits ", NY".
      | thacher\s+state\s+park
      | wildplay\s+thacher
      # Revolutionary War anniversary / tourism often omits ", NY".
      | \#saratoga250\b
      | \bburgoyne\b[\s\S]{0,120}\bsaratoga\b
      | \bsaratoga\b[\s\S]{0,120}\bburgoyne\b
      # Saratoga campaign wires often name Fort Schuyler / Oriskany without ", NY".
      | fort\s+schuyler[\s\S]{0,160}\bsaratoga\b
      | \bsaratoga\b[\s\S]{0,160}fort\s+schuyler
      | \boriskany\b[\s\S]{0,160}\bsaratoga\b
      | \bsaratoga\b[\s\S]{0,160}\boriskany\b
      # Maiden / race cards often say "races at Saratoga" without ", NY".
      | (?:maiden|races?|racing)\s+at\s+saratoga\b
      | \bat\s+saratoga\b[\s\S]{0,60}(?:maiden|stakes|races?\b)
      | maiden\s+watch[\s\S]{0,100}\bsaratoga\b
      | \bsaratoga\b[\s\S]{0,100}maiden\s+watch
      # CCC battlefield preservation wires often omit ", NY".
      | \bccc\b[\s\S]{0,80}\bsaratoga\b
      | \bsaratoga\b[\s\S]{0,80}\bccc\b
      # City of Rensselaer / RPI (county is already covered above).
      | rensselaer\s+polytechnic
      # Distinctive Saratoga Springs venues (often omit ", NY").
      | caffe\s+lena\b
      | high\s+rock\s+park\s+pavilions?\b
      # The Egg (Empire State Plaza) — co-occur with Albany; bare "egg" is too noisy.
      | \bthe\s+egg\b[\s\S]{0,80}\balbany\b
      | \balbany\b[\s\S]{0,80}\bthe\s+egg\b
      | \bthe\s+egg\s+presents\b
      | high\s+rock\s+park[\s\S]{0,80}\bsaratoga\b
      | \bsaratoga\b[\s\S]{0,80}high\s+rock\s+park\b
      # Local paper — reject hyphenated "Sun-Times union" (Chicago),
      # "Seattle Times union", and "New York Times Union" guild phrasing.
      | (?<!york\s)(?<!seattle\s)(?<![\w-])times\s+union\b
      | albany\s+business\s+review\b
      # Albany Riverfront Jazz Festival / Jennings Landing often omit ", NY".
      | albany\s+riverfront\s+jazz
      | jennings\s+landing\b
      # Town of Rotterdam community orgs often omit ", NY".
      | rotterdam\s+community\s+center\b
      # Troy Powers Park concert copy often omits ", NY".
      | \btroy['\u2019]?s\s+powers\s+park\b
      | powers\s+park[\s\S]{0,80}\btroy\b
      | \btroy\b[\s\S]{0,80}powers\s+park\b
      # Yaddo artist colony (Saratoga Springs) often pairs with bare Saratoga.
      | \byaddo\b
      # Glens Falls (Warren County / Cap Region media market) — climate bots and
      # local sports often omit ", NY" beyond the city+state token order.
      | \bglens\s+falls\b
      # Albany music venue — tour posters often say "@ Empire Underground" alone.
      | empire\s+underground\b
      # RPI Experimental Media and Performing Arts Center — listings often omit Troy.
      | \bempac\b
      | experimental\s+media\s+and\s+performing\s+arts\s+center\b
      # Albany Pine Bush Preserve — trail/habitat copy often drops ", NY".
      | albany\s+pine\s+bush(?:\s+preserve)?\b
      | pine\s+bush\s+preserve\b
      # Named Saratoga Race Course stakes / barn copy often omits ", NY".
      | saratoga\s+derby\b
      | saratoga\s+barn\b
      # Albany Liberty Park redevelopment wires often omit ", NY".
      | liberty\s+park[\s\S]{0,80}\balbany\b
      | \balbany\b[\s\S]{0,80}liberty\s+park\b
      # Burdett Birth Center (Troy) NPR wires sometimes drop the city qualifier.
      | burdett\s+birth(?:ing)?\s+center\b
      # County sheriff ICE wires often omit "County" / ", NY".
      | rensselaer\s+sheriff\b
      # Albany Med / AMC trauma wires often omit ", NY".
      | albany\s+medical\s+center\b
      | albany\s+med(?:ical)?\s+health\s+system\b
      | albany\s+med\b
      # Amtrak station code ALB (Albany–Rensselaer); not Cascades Albany (ALY).
      # Status bots put the station code far below the Amtrak header (~160–220 chars).
      | amtrak[\s\S]{0,240}albany\s*\(\s*alb\s*\)
      | albany\s*\(\s*alb\s*\)[\s\S]{0,240}amtrak
      | stopped\s+in\s+albany\s*\(\s*alb\s*\)
      | empire\s+service[\s\S]{0,160}albany\s*\(\s*alb\s*\)
      | albany\s*\(\s*alb\s*\)[\s\S]{0,160}empire\s+service
      | amtrak[\s\S]{0,80}(?:nyp|nyc)\s*->\s*alb\b
      | amtrak[\s\S]{0,80}alb\s*->\s*(?:nyp|nyc)\b
      # Workout / immortality race-meet copy often omits "Race Course" / ", NY".
      | \bbreeze\b[\s\S]{0,80}\bat\s+saratoga\b
      | \bat\s+saratoga\b[\s\S]{0,80}\bbreeze\b
      | saratoga\s+immortality\b
      # BJ's Wholesale Club warehouse wires for Town of Rotterdam NY.
      | bj['\u2019]?s\b[\s\S]{0,120}\brotterdam\b
      | \brotterdam\b[\s\S]{0,120}bj['\u2019]?s\b
      # Legal Aid Society of Northeastern NY — org handle/acronym without ", NY".
      | \blasnny\b
      # Biz-journal HQ copy often omits ", NY" for Cap Region Latham / Halfmoon.
      # Require "… in Latham" (or Latham regional HQ) — not "Latham & Watkins' … Office".
      | (?:regional\s+)?(?:office|hq)\s+in\s+latham\b
      | \blatham\b(?![\s\S]{0,40}watkins)[\s\S]{0,40}regional\s+(?:office|hq)\b
      | \blatham\s+circle\b
      | \blatham\s+farms\b
      | halfmoon\s+hq\b
      | (?:office|hq)\s+in\s+halfmoon\b
      | new\s+halfmoon\s+(?:location|hq)\b
      # Local PropTech — require Cap Region place (bare #rentredi is SEO spam).
      | \brentredi\b[\s\S]{0,80}(?:\blatham\b|\balbany\b|capital\s+region\b|\#albanyny\b)
      | (?:\blatham\b|\balbany\b|capital\s+region\b|\#albanyny\b)[\s\S]{0,80}\brentredi\b
      # Local business / tourism copy often says "Albany region" without ", NY".
      | albany\s+region\b
      # Crossgates Mall / Commons (Guilderland) — often omit ", NY".
      | crossgates\s+(?:commons|mall)\b
      # Tri-City ValleyCats / Joseph L. Bruno Stadium (Troy).
      | tri-?city\s+valleycats\b
      | \bvalleycats\b
      | joseph\s+l\.?\s+bruno\s+stadium
      | \bbruno\s+stadium\b
      # Park Playhouse (Washington Park, Albany).
      | park\s+playhouse\b
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
# ``troy`` ignores email local-parts (``troy@…``), hyphenated names/domains
# (``troy-caperton``), troy weight (oz / "10.8 troy" / Portuguese ``onça-troy``),
# "Donna Troy", broadcaster "Troy Aikman", NFL "Troy Fautanu", and footballer
# "Troy Deeney".
_TROY_PLACE = (
    r'(?<!\d\s)(?<![\w.])(?<!donna\s)'
    r'(?<!on[cç]a[- ])(?<!onza[- ])(?<!ounce[- ])'
    r'troy(?![\w@.-])'
    r'(?!\s*(?:oz|ounces?|ozt|weight|aikman|fautanu|deeney)\b)'
)

_AMBIGUOUS_PLACE = re.compile(
    rf"""
    (?:
        \balbany\b
      | {_TROY_PLACE}
      | \blatham\b
      | \bmalta\b
      # Town of Scotia — not the province inside "Nova Scotia".
      | (?<!nova\s)\bscotia\b
      | \bbethlehem\b
      # Town of Brunswick — not the province inside "New Brunswick" / URL
      # slug "new-brunswick-…".
      | (?<!new[\s-])\bbrunswick\b
      | \bgalway\b
      | \bstillwater\b
      | \bwaterford\b
      | \brotterdam\b
      | \bhalfmoon\b
      | \bcolonie\b
      # City of Rensselaer (not only "Rensselaer County" strong/entity).
      | \brensselaer\b
      | saratoga\s+springs
      | \bsaratoga\b
      # Leading boundary: do not match inside "around Lake George".
      | \bround\s+lake\b
      | \bdelmar\b
      | \bravena\b
      | \baltamont\b
      | \bsand\s+lake\b
      | \bgreen\s+island\b
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
# Wire datelines use ``N.Y.``; locals often write ``NYS`` for New York State.
# The two-letter form is case-sensitive (``NY`` / ``ny`` only): Norwegian/Danish
# sentence-initial ``Ny`` ("New") must not unlock Troy / Waterford / etc.
_NY_CONTEXT = re.compile(
    r"""
    (?:
      # Reject letters AND digits before NY so aircraft type suffixes like
      # A321-271NY (Wizz Air Malta) do not unlock Cap Region context.
      # Use Unicode ``\w`` so non-ASCII letters (e.g. Hungarian tartomány)
      # cannot leave a bare ``ny`` token.
        (?-i:(?<!\w)(?:NY|ny)(?!\w))(?!\s*times\b)
      # Reject handle TLDs like @socialists.nyc (dot before nyc) and aircraft
      # registrations such as 9H-NYC (Malta → Palma flight trackers).
      | (?<![.-])\bnyc\b
      | n\.y\.
      | \bnys\b
      | new\s+york(?!\s+(?:
            times|post|daily\s+news|magazine|observer|herald|metro|sun
          )\b)
      # Bare "upstate" usually means NY; do not unlock "Upstate South Carolina".
      | upstate(?!\s+(?:south|north)\s+carolina\b)
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
# intact for the troy ambiguous-place guard (``troy`` must not match ``troy@``).
_HANDLE_MENTION = re.compile(r'(?<![\w.])@[\w.-]+', flags=re.UNICODE)

# Hard negatives that always win over an otherwise-strong local phrase
# (e.g. "capital region" inside "capital region of Madrid").
# "New Albany Bus Station" is Albany NY's proposed terminal, not New Albany IN/MS.
_HARD_NEGATIVE_BLOCKS_STRONG = re.compile(
    r"""
    (?:
        albany\s+park
      | new\s+albany(?!\s+bus\s+(?:station|terminal|depot))
      | albany\s+state\s+university
      | national\s+capital\s+region
      # Slash form appears in Brussels Times cards ("Brussels/Capital Region").
      | brussels[- /]capital\s+region
      | canadian\s+capital\s+region
      # Other-state / non-NY newsroom jargon (e.g. Jackson MS bureau).
      | capital\s+region\s+bureau\b
      # Putnam / Hudson Valley street — not City of Albany.
      | albany\s+post\s+road
      # Cape Town suburb / wine region / municipal alerts — not Helderberg Escarpment.
      # Prefer #capetown / #southafrica / Helderberg College; window covers text→hashtag.
      | helderberg\s+college\b
      | helderberg[\s\S]{0,280}(?:
            cape\s+town|\#capetown\b|western\s+cape|south\s+africa|
            \#southafrica\b|\#africa\b|\bstrand\b|sweet\s+paws|pressportal\.co\.za
          )
      | (?:
            cape\s+town|\#capetown\b|western\s+cape|south\s+africa|
            \#southafrica\b|\#africa\b|sweet\s+paws|pressportal\.co\.za
          )[\s\S]{0,280}helderberg
      # Seattle / New York Times guild phrasing — not Albany Times Union.
      | seattle\s+times(?:\s+union)?\b
      | seattle[\s\S]{0,100}times\s+union\b
      | times\s+union[\s\S]{0,100}seattle
      | new\s+york\s+times(?:\s+union)?\b
      | times\s+union[\s\S]{0,100}new\s+york\s+times
      # SoCal / multi-track handicap hashtag stuffing — not Race Course NY.
      | (?:\#socal\b|\#losangeles\b|southern\s+california)[\s\S]{0,200}\#saratoga\b
      | \#saratoga\b[\s\S]{0,200}(?:\#socal\b|\#losangeles\b|southern\s+california)
      | (?:\#parxracing\b|\#thistledown\b|\#horseshoeindy\b|
         \#assiniboia(?:downs)?\b|\#mountaineerpark\b|
         \#woodbine\b|\#charlestownraces\b|\#remingtonpark\b)[\s\S]{0,200}\#saratoga\b
      | \#saratoga\b[\s\S]{0,200}(?:\#parxracing\b|\#thistledown\b|
         \#horseshoeindy\b|\#assiniboia(?:downs)?\b|\#mountaineerpark\b|
         \#woodbine\b|\#charlestownraces\b|\#remingtonpark\b)
      # Long Island SEO hashtag stuffing with #albany / #albanyny — not Cap Region.
      | (?:\#longisland(?:ny|newyork)?\b|\#longislandrealestate\b)[\s\S]{0,200}\#albany(?:ny|_ny)?\b
      | \#albany(?:ny|_ny)?\b[\s\S]{0,200}(?:\#longisland(?:ny|newyork)?\b|\#longislandrealestate\b)
      # Binghamton housing development — not Saratoga Race Course / Springs.
      | saratoga\s+terrace\b
      # Louisiana MPO — not NY Capital District Regional Planning Commission (CDRPC).
      | capital\s+region\s+planning\s+commission
      # Berlin-Brandenburg / Germany "capital region" (Medienboard wires).
      | berlin[- /]brandenburg\s+capital\s+region
      | berlin[- ]brandenburg[\s\S]{0,80}capital\s+region\b
      | capital\s+region\b[\s\S]{0,80}berlin[- ]brandenburg
      # NJ street — not Town of Brunswick NY.
      | brunswick\s+pike\b
      | capital\s+region\s+of\s+(?:
            madrid|spain|belgium|brussels|paris|france|berlin|germany|
            tokyo|seoul|beijing|delhi|ottawa|canberra|rome|italy|
            amsterdam|vienna|warsaw|prague|lisbon|athens|dublin|
            canada|denmark|copenhagen|mississippi|louisiana|pennsylvania|
            california|sacramento|korea|south\s+korea|ukraine|kyiv|kiev|
            sudan|khartoum|virginia|richmond|colombia|bogot[aá]|
            iceland|reykjav[ií]k|finland|helsinki|australia|
            georgia|atlanta|russia|moscow|bulgaria|sofia|japan|tokyo|
            michigan|lansing
          )\b
      | ukrainian\s+capital\s+region
      | (?:russian|moscow)\s+capital\s+region
      | sudan(?:ese)?\s+capital\s+region
      | icelandic\s+capital\s+region
      | french\s+capital\s+region
      | finnish\s+capital\s+region
      | australian\s+capital\s+region
      | (?:virginia|richmond)\s+capital\s+region
      | (?:bulgarian?|sofia)\s+capital\s+region
      | (?:japanese?|tokyo)\s+capital\s+region
      | michigan\s+capital\s+region
      | bogot[aá]\s+capital\s+district
      | capital\s+district\s*,?\s*colombia\b
      | icelandic\s+capital\s+district
      | capital\s+district\s+fire\s+and\s+rescue
      | (?:the\s+)?egg\s+and\s+art\s+garden
      | art\s+garden\s+kc\b
      # Papua New Guinea — "National Capital District" contains Cap District.
      | national\s+capital\s+district
      # Harrisburg PA utility — not NY Capital Region.
      | capital\s+region\s+water\b
      | pennsylvania\s+capital\s+region
      # Canberra / ACT charity branding (Rise Above).
      | capital\s+region\s+cancer\s+relief
      | rise\s+above\s*[-–—]?\s*capital\s+region
      | hauptstadtregion
      # Law firm — not Town of Latham NY HQ wires.
      | latham\s*(?:&|and)\s*watkins
      # Oregon 5A Mid-Willamette / South Albany & West Albany high schools.
      | mid[- ]?willamette
      | west\s+albany\s+high(?:\s+school)?
      | south\s+albany\s+high(?:\s+school)?
      | (?:south|west)\s+albany\b[\s\S]{0,200}(?:
            \boregon\b|\#oregon\b|corvallis|crescent\s+valley|
            \blebanon\b|silverton|mid[- ]?willamette|\.oregonlive\.
          )
      | (?:
            \boregon\b|\#oregon\b|corvallis|crescent\s+valley|
            \blebanon\b|silverton|mid[- ]?willamette|\.oregonlive\.
          )[\s\S]{0,200}(?:south|west)\s+albany\b
      | disney(?:['\u2019]?s)?\s+saratoga\s+springs
      | watervliet\s*,?\s*(?:mi|michigan)\b
      | troy\s*,?\s*(?:mi|michigan)\b
      | detroit\s*/\s*troy\b
      | loudonville\s*,?\s*(?:oh|ohio)\b
      | reinvent\s*albany
      # PubMed journal abbreviation — not Albany NY local news.
      | aging\s*\(\s*albany\s*ny\s*\)
      # Sports-print spam lists Albany among many cities.
      | rowonebrand
      # Belgian sculptor surname — not Helderberg Escarpment.
      | van\s+helderbergh
      # Irish crystal brand — not Town of Waterford NY.
      | waterford\s+crystal
      | waterford\s+wedgwood
      # Rensselaer, Indiana — not City of Rensselaer NY.
      | rensselaer\s*,?\s*(?:in|indiana)\b
      # Myth / poetry title — not the City of Troy.
      | helen\s+of\s+troy
      # Albany County WY NWS — do not let "Albany County" strong override.
      | albany\s*,\s*(?:wy|wyoming)\b
      | albany\s+county[\s\S]{0,80}(?:\b(?:wy|wyoming)\b|\#wy\b|\#wywx\b|\blaramie\b|cheyenne)
      | (?:\b(?:wy|wyoming)\b|\#wy\b|\#wywx\b|\blaramie\b|nws\s+cheyenne)[\s\S]{0,80}albany\s+county
      # Lansing MI airport — not NY Capital Region.
      | capital\s+region\s+international\s+airport
      | capital\s+region\b[\s\S]{0,80}(?:\blansing\b|grand\s+rapids)
      | (?:\blansing\b|grand\s+rapids|nws\s+grand\s+rapids)[\s\S]{0,80}capital\s+region\b
      # Sofia / Tokyo "capital region" wires.
      | (?:bulgarian?|sofia)\s+capital\s+region
      | (?:japanese?|tokyo)\s+capital\s+region
      | capital\s+region\s+of\s+(?:bulgaria|sofia|japan|tokyo)\b
      # GTA Liberty City car named Albany — not Albany NY.
      | liberty\s+city
      | \bgta\s*iv?\b
      # Montclair CA park — not Saratoga Springs NY.
      | saratoga\s+park[\s\S]{0,120}montclair
      | montclair[\s\S]{0,120}saratoga\s+park
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Canadian "capital region" cues (Ottawa national, Victoria BC / #yyj / CRD).
# Snowbirds = RCAF demo team (Victoria-area flyovers); not birdwatching copy.
# Window is 240 chars: CFAX call-in intros often put #yyj / #BCpoli after a long clause.
# Goldstream / Vancouver Island rail copy often says bare "Victoria" (no "BC").
_CANADIAN_GEO_CUE = (
    r'\bcanada\b|\bcanadian\b|\bottawa\b|\#canadian\w*|'
    r'\#yyj\b|\#bcpoli\b|british\s+columbia|\blangford\b|'
    r'victoria(?:\s*,?\s*bc\b)|greater\s+victoria|'
    r'\bgoldstream\b|vancouver\s*island|vancouverisland|peers\s+victoria|'
    r'restoreislandrail|'
    r'capital\s+regional\s+district|\blivable\s+crd\b|'
    r'timescolonist\.com|ottawacitizen\.com|\bsnowbirds?\b|parkland\s+secondary|'
    r'\bcfax\b|cfax\.com'
)

# Ottawa / Canada / BC "capital region" co-occurring with Canadian cues (not NY).
_CANADIAN_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        canadian\s+capital\s+region
      | capital\s+region\s+of\s+(?:canada|ottawa)\b
      | capital\s+region\b[\s\S]{{0,240}}(?:{_CANADIAN_GEO_CUE})
      | (?:{_CANADIAN_GEO_CUE})[\s\S]{{0,240}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Maryland / DC-metro "capital region" (Prince George's, MoCo MD tourism, etc.).
# Avoid bare "montgomery county" — NYS Montgomery County appears in ALY weather.
_MD_DC_GEO_CUE = (
    # AppView cards often use a curly apostrophe in "George's".
    r"prince\s+george['\u2019]?s|\bmaryland\b|\#md(?:wx|politics|gov|news)?\b|"
    r'\#maryland\w*|'
    r'washington(?:\s*,?\s*d\.?c\.?\b)|(?<![\w.])dc\s+metro\b|'
    r'(?<![\w.])dc\s+snipers?\b|national\s+law\s+enforcement\s+museum|'
    r'\bdmv\b|silver\s+spring|\bbethesda\b|\brockville\b|'
    r'olney\s+theatre|national\s+harbor|college\s+park|\bannapolis\b|'
    # NPS funding wires contrast "national parks outside the capital region".
    r'national\s+parks?\s+outside|'
    # DC go-go music listings often omit Maryland / DC place tokens.
    r'\bgo-go\b|go\s+go\s+(?:music|concert|band)|'
    # NBC4 / Telemundo 44 are DC-metro stations; AppView cards say "capital region".
    r'\bnbc\s*4\b|\btelemundo\s*44\b'
)

# MD/DC "capital region/district" co-occurring with Maryland / DC cues (not NY).
# AppView cards say "Capital district, Washington, D.C." as well as "capital region".
_MD_DC_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        capital\s+(?:region|district)\b[\s\S]{{0,280}}(?:{_MD_DC_GEO_CUE})
      | (?:{_MD_DC_GEO_CUE})[\s\S]{{0,280}}capital\s+(?:region|district)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Louisiana "capital region" (Baton Rouge / WBRZ). Handles like wbrz-mirror /
# wbrznews2 are checked via author_handle because short copy often omits geo.
_LA_GEO_CUE = (
    r'\bbaton\s+rouge\b|\blouisiana\b|\#la(?:wx|politics|gov)?\b|'
    r'\bwbrz\b|wbrz\.com|east\s+baton\s+rouge|west\s+feliciana|'
    r'\bfeliciana\b|st\.?\s+mary\s+parish|'
    r'\bascension(?:\s+parish)?\b|\bprairieville\b|\bsorrento\b|'
    r'\bst\.?\s+amant\b|\bgonzales\b|'
    # LA MPO name (NY uses CDRPC — Capital District Regional Planning Commission).
    r'capital\s+region\s+planning\s+commission|\bcrpc\b'
)

_LA_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        capital\s+region\s+of\s+louisiana\b
      | capital\s+region\s+planning\s+commission
      | capital\s+region\b[\s\S]{{0,480}}(?:{_LA_GEO_CUE})
      | (?:{_LA_GEO_CUE})[\s\S]{{0,480}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Pennsylvania "capital region" (Harrisburg / Capital Region Water).
_PA_GEO_CUE = (
    r'\bharrisburg\b|\bpennsylvania\b|\#pa(?:wx|politics|gov)\b|'
    r'capital\s+region\s+water\b|pennlive|susquehanna\b|'
    r'pennsylvania\s+capital\s+region'
)

_PA_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        pennsylvania\s+capital\s+region
      | capital\s+region\s+of\s+pennsylvania\b
      | capital\s+region\s+water\b
      | capital\s+region\b[\s\S]{{0,160}}(?:{_PA_GEO_CUE})
      | (?:{_PA_GEO_CUE})[\s\S]{{0,160}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# California "capital region" (Sacramento / SacBee).
_CA_GEO_CUE = (
    r'\bsacramento\b|\bcalifornia\b|\#ca(?:wx|politics|gov)\b|'
    r'\bsacbee\b|sacbee\.com|folsom\b|elk\s+grove\b|'
    r'west\s+sacramento|roseville\b'
)

_CA_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        capital\s+region\s+of\s+(?:california|sacramento)\b
      | capital\s+region\b[\s\S]{{0,160}}(?:{_CA_GEO_CUE})
      | (?:{_CA_GEO_CUE})[\s\S]{{0,160}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# South Korea "capital region" (Seoul / Gyeonggi / KMA heat alerts).
_KR_GEO_CUE = (
    r'\bseoul\b|\bgyeonggi\b|\bkorea\b|south\s+korea|'
    r'\bincheon\b|\bkoreaherald\b|korea\s+herald|'
    r'\bgangnam\b|han\s+river|\bkma\b'
)

_KR_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        capital\s+region\s+of\s+(?:korea|south\s+korea|seoul)\b
      | greater\s+seoul
      | capital\s+region\b[\s\S]{{0,160}}(?:{_KR_GEO_CUE})
      | (?:{_KR_GEO_CUE})[\s\S]{{0,160}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Ukraine "capital region" (Kyiv / AP wire mirrors).
_UA_GEO_CUE = (
    r'\bukraine\b|\bukrainian\b|\bkyiv\b|\bkiev\b|'
    r'\#ukraine\b|\#kyiv\b|kyivunderattack'
)

_UA_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        ukrainian\s+capital\s+region
      | capital\s+region\s+of\s+(?:ukraine|kyiv|kiev)\b
      | capital\s+region\b[\s\S]{{0,160}}(?:{_UA_GEO_CUE})
      | (?:{_UA_GEO_CUE})[\s\S]{{0,160}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Russia "capital region" (Moscow drone / sabotage wire mirrors).
_RU_GEO_CUE = (
    r'\brussia\b|\brussian\b|\bmoscow\b|\bkremlin\b|'
    r'\#russia\b|\#moscow\b|\#putin\b'
)

_RU_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        (?:russian|moscow)\s+capital\s+region
      | capital\s+region\s+of\s+(?:russia|moscow)\b
      | capital\s+region\b[\s\S]{{0,160}}(?:{_RU_GEO_CUE})
      | (?:{_RU_GEO_CUE})[\s\S]{{0,160}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Sudan "capital region" (Khartoum UNEP / wire cards).
_SD_GEO_CUE = r'\bsudan\b|\bsudanese\b|\bkhartoum\b|\#sudan\b|\#khartoum\b'

_SD_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        sudan(?:ese)?\s+capital\s+region
      | capital\s+region\s+of\s+(?:sudan|khartoum)\b
      | capital\s+region\b[\s\S]{{0,160}}(?:{_SD_GEO_CUE})
      | (?:{_SD_GEO_CUE})[\s\S]{{0,160}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Virginia "capital region" (Richmond / Spanberger / Dominion Energy).
_VA_GEO_CUE = (
    r'\bvirginia\b|\brichmond\b|\#va(?:wx|politics|gov|news)?\b|'
    r'\#virginia\w*|'
    r'\bspanberger\b|dominion(?:\s+energy)?\b|\bnextera\b|'
    r'hampton\s+roads|northern\s+virginia'
)

_VA_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        (?:virginia|richmond)\s+capital\s+region
      | capital\s+region\s+of\s+(?:virginia|richmond)\b
      | capital\s+region\b[\s\S]{{0,280}}(?:{_VA_GEO_CUE})
      | (?:{_VA_GEO_CUE})[\s\S]{{0,280}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# VHSL / Richmond-area "Capital District" high-school athletics — not NY.
_VA_CAPITAL_DISTRICT_CUE = (
    r'\bhenrico\b|\bhanover\b|mechanicsville|\bashland\b|\batlee\b|'
    r'\bvarina\b|highland\s+springs|\brichmond\b|\bvirginia\b|'
    r'\#va(?:wx|politics|gov|news)?\b|\#virginia\w*'
)

_VA_CAPITAL_DISTRICT = re.compile(
    rf"""
    (?:
        capital\s+district\b[\s\S]{{0,200}}(?:{_VA_CAPITAL_DISTRICT_CUE})
      | (?:{_VA_CAPITAL_DISTRICT_CUE})[\s\S]{{0,200}}capital\s+district\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Berlin-Brandenburg / Germany "capital region" (Medienboard / games funding).
# Prefer compound / funding cues over bare "Berlin" (orchestra repertoire posts).
_DE_GEO_CUE = (
    r'berlin[- ]brandenburg|\bbrandenburg\b|\bmedienboard\b|hauptstadtregion|'
    r'\bgermany\b|\bgerman\b|\#brandenburg\b|\#germany\b|'
    r'berlin[- /]brandenburg'
)

_DE_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        berlin[- /]brandenburg\s+capital\s+region
      | (?:german|brandenburg)\s+capital\s+region
      | capital\s+region\s+of\s+(?:germany|berlin|brandenburg)\b
      | capital\s+region\b[\s\S]{{0,200}}(?:{_DE_GEO_CUE})
      | (?:{_DE_GEO_CUE})[\s\S]{{0,200}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bogotá "Capital District" / Colombia — not NY Capital District.
_CO_CAPITAL_DISTRICT = re.compile(
    r"""
    (?:
        bogot[aá]\s*,?\s*bogot[aá]\s+capital\s+district
      | bogot[aá]\s+capital\s+district
      | capital\s+district\s*,?\s*colombia
      | capital\s+district\b[\s\S]{0,80}\bcolombia\b
      | \bcolombia\b[\s\S]{0,80}capital\s+district\b
      | \bbogot[aá]\b[\s\S]{0,80}capital\s+district\b
      | capital\s+district\b[\s\S]{0,80}\bbogot[aá]\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Papua New Guinea "National Capital District" (Port Moresby) — not NY.
_PNG_GEO_CUE = (
    r'national\s+capital\s+district|\bport\s+moresby\b|\bmotu\s+koitabu\b|'
    r'\bncdpha\b|papua\s+new\s+guinea|\bpost\s+courier\b'
)

_PNG_CAPITAL_DISTRICT = re.compile(
    rf"""
    (?:
        national\s+capital\s+district
      | capital\s+district\b[\s\S]{{0,160}}(?:{_PNG_GEO_CUE})
      | (?:{_PNG_GEO_CUE})[\s\S]{{0,160}}capital\s+district\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bay Area Albany / Saratoga (CA) city lists — not Albany NY / Saratoga Springs NY.
_BAY_AREA_PLACE_CUE = (
    r'\bbay\s+area\b|\bpiedmont\b|\batherton\b|\boakland\b|'
    r'\bberkeley\b|\bemeryville\b|\bcupertino\b|\blos\s+gatos\b|'
    r'\bmenlo\s+park\b|\bsan\s+mateo\b|\bsanta\s+clara\b|'
    r'\bfremont\b|\bhayward\b|\blivermore\b|\bmillbrae\b|'
    r'\bpalo\s+alto\b|\bsunnyvale\b|\balameda\s+co\b|'
    r'sfchronicle|san\s+francisco'
)

_BAY_AREA_ALBANY = re.compile(
    rf"""
    (?:
        \balbany\b[\s\S]{{0,200}}(?:{_BAY_AREA_PLACE_CUE})
      | (?:{_BAY_AREA_PLACE_CUE})[\s\S]{{0,200}}\balbany\b
      | \bsaratoga\b[\s\S]{{0,200}}(?:{_BAY_AREA_PLACE_CUE})
      | (?:{_BAY_AREA_PLACE_CUE})[\s\S]{{0,200}}\bsaratoga\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Matt Damon / Tom McCarthy film "Stillwater" — not the Town of Stillwater NY.
_STILLWATER_FILM = re.compile(
    r"""
    (?:
        ["'“”]?stillwater["'“”]?\s*,?\s*starring
      | (?:film|movie|picture)\s+["'“”]stillwater["'“”]
      | since\s+["'“”]stillwater["'“”]
      | \bstillwater\b[\s\S]{0,100}\b(?:matt\s+damon|tom\s+mccarthy)\b
      | \b(?:matt\s+damon|tom\s+mccarthy)\b[\s\S]{0,100}\bstillwater\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# North Country "Stillwater Road" (Lewis Co / Croghan) — not Town of Stillwater NY.
_STILLWATER_ROAD_NORTH = re.compile(
    r"""
    (?:
        \bstillwater\s+road\b
      | \bcroghan\b
      | lewis\s+co(?:unty)?\b
      | \bbuf\.nws
      | \bbuf\.weather
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Troy, Michigan (Detroit suburb) — not Troy NY, even when NYC/Ithaca appear.
# Hashtag metro dumps (#Michigan … #Troy) need a wider window than "Troy, MI".
_TROY_MICHIGAN = re.compile(
    r"""
    (?:
        troy\s*,?\s*(?:mi|michigan)\b
      | detroit\s*/\s*troy\b
      | troy\s*/\s*detroit\b
      | (?<![\w.])troy(?![\w@.-])[\s\S]{0,40}\b(?:mi|michigan)\b
      | \b(?:mi|michigan)\b[\s\S]{0,40}(?<![\w.])troy(?![\w@.-])
      | \#troy\b[\s\S]{0,280}(?:\#michigan\b|\#detroit\b|\#grandrapids\b)
      | (?:\#michigan\b|\#detroit\b|\#grandrapids\b)[\s\S]{0,280}\#troy\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Indiana Albany + Saratoga weather towns (Muncie / Ball State / #inwx).
# Keep cues distinctive — avoid Bay Area "Union City" / generic "farmland".
_IN_ALBANY_SARATOGA = re.compile(
    r"""
    (?:
        \#inwx\b
      | \bmuncie\b
      | ball\s+state
      | \bindiana\b
      | \bmodoc\b
      | \bridgeville\b
      | parker\s+city
      | \bwinchester\b[\s\S]{0,80}\b(?:eaton|lynn|selma)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Iceland "capital region" / "capital district" (Reykjavik Police / mbl.is mirrors).
# Window 280 covers mbl.is → "District Commissioner of the Capital Region" card distance.
_IS_GEO_CUE = (
    r'\biceland\b|\bicelandic\b|\breykjav[ií]k\b|'
    r'\#iceland\b|\#reykjav[ií]k\b|mbl\.is|'
    r'miklabraut|grens[aá]svegur|\bkringlan\b'
)

_IS_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        icelandic\s+capital\s+(?:region|district)
      | capital\s+(?:region|district)\s+of\s+(?:iceland|reykjav[ií]k)\b
      | capital\s+(?:region|district)\b[\s\S]{{0,280}}(?:{_IS_GEO_CUE})
      | (?:{_IS_GEO_CUE})[\s\S]{{0,280}}capital\s+(?:region|district)\b
      # Reykjavik emergency services branding.
      | capital\s+district\s+fire\s+and\s+rescue
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Paris / France "capital region" (YouTube storm cards: "French capital region").
_FR_GEO_CUE = (
    r'\bfrance\b|\bfrench\b|\bparis\b|'
    r'\#france\b|\#paris\b|\#french\b'
)

_FR_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        french\s+capital\s+region
      | capital\s+region\s+of\s+(?:france|paris)\b
      | capital\s+region\b[\s\S]{{0,200}}(?:{_FR_GEO_CUE})
      | (?:{_FR_GEO_CUE})[\s\S]{{0,200}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Brussels / Belgium "Capital Region" (Brussels Times municipality / RNP-07L cards).
# Hyphen and slash forms are already hard-negatives; this catches co-occurrence
# ("City of Brussels … municipality in the Capital Region").
_BE_GEO_CUE = (
    r'\bbrussels\b|\bbelgium\b|\bbelgian\b|'
    r'\#brussels\b|\#belgium\b|\#belgian\b|'
    r'brussels\s+times|rnp-?07l\b|brusselsairport|'
    r'flanders\b|wallonia\b|anderlecht\b|schaerbeek\b'
)

_BE_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        brussels[- /]capital\s+region
      | belgian\s+capital\s+region
      | capital\s+region\s+of\s+(?:belgium|brussels)\b
      | capital\s+region\b[\s\S]{{0,280}}(?:{_BE_GEO_CUE})
      | (?:{_BE_GEO_CUE})[\s\S]{{0,280}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Delhi / India "capital region" (IMD flood cards; not NY Capital Region).
_IN_GEO_CUE = (
    r'\bdelhi\b|\bindia\b|\bindian\b|new\s+delhi\b|'
    r'\#delhi\b|\#india\b|\#indian\b|\#delhirain\b|'
    r'\#innews\b|imd\s+red\s+alert|\bimd\b[\s\S]{0,40}\bdelhi\b|'
    r'noida\b|gurugram\b|gurgaon\b|ncr\s+delhi\b'
)

_IN_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        (?:indian|delhi)\s+capital\s+region
      | capital\s+region\s+of\s+(?:india|delhi)\b
      | capital\s+region\b[\s\S]{{0,280}}(?:{_IN_GEO_CUE})
      | (?:{_IN_GEO_CUE})[\s\S]{{0,280}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Boston Newton village / MBTA Worcester Line — not Colonie Newtonville.
_NEWTONVILLE_MA = re.compile(
    r"""
    (?:
        newtonville[\s\S]{0,220}(?:
            \bboston\b|\bmbta\b|\#mbta\b|newton\s+ma\b|newton\s+highlands|
            west\s+newton|worcester\s+line|garden\s+city
        )
      | (?:
            \bboston\b|\bmbta\b|\#mbta\b|newton\s+ma\b|newton\s+highlands|
            west\s+newton|worcester\s+line
        )[\s\S]{0,220}newtonville
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# CT bank-job "Capital District" (East Hartford) — not NY Capital District.
_CT_CAPITAL_DISTRICT = re.compile(
    r"""
    (?:
        capital\s+district[\s\S]{0,140}(?:
            east\s+hartford|\bhartford\b|\bconnecticut\b|(?<![a-z0-9])ct-
        )
      | (?:
            east\s+hartford|\bhartford\b|\bconnecticut\b|(?<![a-z0-9])ct-
        )[\s\S]{0,140}capital\s+district
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Long Island / Adelphi "Green Island Sangha" — not Village of Green Island NY.
_GREEN_ISLAND_OTHER = re.compile(
    r"""
    (?:
        green\s+island\s+sangha
      | green\s+island[\s\S]{0,160}(?:adelphi|long\s+island|plum\s+village)
      | (?:adelphi|long\s+island|plum\s+village)[\s\S]{0,160}green\s+island
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Brunswick Schools (OH) + NYC tournament cards — not Town of Brunswick NY.
_BRUNSWICK_SCHOOLS_OTHER = re.compile(
    r"""
    (?:
        brunswick\s+schools?\b[\s\S]{0,220}(?:
            new\s+york\s+city|nba\s+math\s+hoops|\bcleveland\b
        )
      | (?:
            new\s+york\s+city|nba\s+math\s+hoops|\bcleveland\b
        )[\s\S]{0,220}brunswick\s+schools?\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Kansas City "The Egg and Art Garden" — not Albany's The Egg.
_EGG_KANSAS_CITY = re.compile(
    r"""
    (?:
        (?:the\s+)?egg\s+and\s+art\s+garden
      | art\s+garden\s+kc\b
      | \bat\s+the\s+egg\b[\s\S]{0,120}(?:kansas\s+city|\bkc\b)
      | (?:kansas\s+city|\bkc\b)[\s\S]{0,120}\bat\s+the\s+egg\b
      | bottoms?\s+up\s+festival[\s\S]{0,80}\bthe\s+egg\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Finland "capital region" (Helsinki / HSL winter timetables).
_FI_GEO_CUE = (
    r'\bfinland\b|\bfinnish\b|\bhelsinki\b|'
    r'\#finland\b|\#helsinki\b|helsinki\s+region\s+transport|'
    r'(?<![a-z0-9])hsl(?![a-z0-9])'
)

_FI_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        finnish\s+capital\s+region
      | capital\s+region\s+of\s+(?:finland|helsinki)\b
      | capital\s+region\b[\s\S]{{0,160}}(?:{_FI_GEO_CUE})
      | (?:{_FI_GEO_CUE})[\s\S]{{0,160}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Denmark "capital region" (Copenhagen / Hovedstaden / light-rail wires).
_DK_GEO_CUE = (
    r'\bdenmark\b|\bdanish\b|\bcopenhagen\b|\bhovedstaden\b|'
    r'hovedstadens\s+letbane|\bletbane\b|'
    r'\bish[øo]j\b|\blundtofte\b|\bgladsaxe\b|'
    r'\bdr\s+reports\b|\#dkpol\b'
)

_DK_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        danish\s+capital\s+region
      | capital\s+region\s*,?\s*denmark\b
      | capital\s+region\s+of\s+(?:denmark|copenhagen)\b
      | capital\s+region\b[\s\S]{{0,200}}(?:{_DK_GEO_CUE})
      | (?:{_DK_GEO_CUE})[\s\S]{{0,200}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Alberta "capital region" (Edmonton metro / #AbPoli data-centre wires).
_AB_GEO_CUE = (
    r'\balberta\b|\bedmonton\b|\bcalgary\b|'
    r'\#ab(?:poli|leg)\b|\#cdnpoli\b|\bglubish\b|'
    r'\bcdnpoli\b|alberta\s+faces\b'
)

_AB_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        alberta\s+capital\s+region
      | capital\s+region\s+of\s+(?:alberta|edmonton)\b
      | capital\s+region\b[\s\S]{{0,200}}(?:{_AB_GEO_CUE})
      | (?:{_AB_GEO_CUE})[\s\S]{{0,200}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bulgaria "capital region" (Sofia GPS / aviation wires).
_BG_GEO_CUE = (
    r'\bbulgaria\b|\bbulgarian\b|\bsofia\b|'
    r'informat\.ro|\#bgpoli\b'
)

_BG_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        bulgarian\s+capital\s+region
      | capital\s+region\s+of\s+(?:bulgaria|sofia)\b
      | capital\s+region\b[\s\S]{{0,200}}(?:{_BG_GEO_CUE})
      | (?:{_BG_GEO_CUE})[\s\S]{{0,200}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Japan "capital region" (Tokyo metro earthquake / rail wires).
_JP_GEO_CUE = (
    r'\bjapan\b|\bjapanese\b|\btokyo\b|'
    r'\bibaraki\b|\bchiba\b|\bsaitama\b|\bkanagawa\b|'
    r'\byokohama\b|\bosaka\b|\#japan\b'
)

_JP_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        japanese\s+capital\s+region
      | capital\s+region\s+of\s+(?:japan|tokyo)\b
      | capital\s+region\b[\s\S]{{0,200}}(?:{_JP_GEO_CUE})
      | (?:{_JP_GEO_CUE})[\s\S]{{0,200}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Michigan "capital region" (Lansing / Capital Region International Airport).
_MI_GEO_CUE = (
    r'\blansing\b|\bmichigan\b|\#miwx\b|'
    r'grand\s+rapids|nws\s+grand\s+rapids|'
    r'capital\s+region\s+international\s+airport'
)

_MI_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        michigan\s+capital\s+region
      | capital\s+region\s+international\s+airport
      | capital\s+region\s+of\s+(?:michigan|lansing)\b
      | capital\s+region\b[\s\S]{{0,200}}(?:{_MI_GEO_CUE})
      | (?:{_MI_GEO_CUE})[\s\S]{{0,200}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Australia "capital region" (Canberra / ACT / Queanbeyan charities).
_AU_GEO_CUE = (
    r'\bcanberra\b|\bqueanbeyan\b|\baustralia\b|\baustralian\b|'
    r'\#auspol\b|riseabovecbr|\.org\.au\b|'
    r'australian\s+capital\s+territory|\bact\s+eden'
)

_AU_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        australian\s+capital\s+region
      | capital\s+region\s+cancer\s+relief
      | rise\s+above\s*[-–—]?\s*capital\s+region
      | capital\s+region\s+of\s+(?:australia|canberra|act)\b
      | capital\s+region\b[\s\S]{{0,160}}(?:{_AU_GEO_CUE})
      | (?:{_AU_GEO_CUE})[\s\S]{{0,160}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Georgia (US) "capital region" (Atlanta metro ICE / wire mirrors).
_GA_ATLANTA_GEO_CUE = (
    r'\batlanta\b|\bgeorgia\b|\#ga(?:pol|wx|politics)\b|'
    r'state\s+of\s+georgia|operation\s+safe\s+community'
)

_GA_ATLANTA_CAPITAL_REGION = re.compile(
    rf"""
    (?:
        (?:atlanta|georgia)\s+capital\s+region
      | capital\s+region\s+of\s+(?:georgia|atlanta)\b
      | capital\s+region\b[\s\S]{{0,160}}(?:{_GA_ATLANTA_GEO_CUE})
      | (?:{_GA_ATLANTA_GEO_CUE})[\s\S]{{0,160}}capital\s+region\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Japanese poetry / translation using "capital district" (not NY).
_JP_CAPITAL_DISTRICT = re.compile(
    r"""
    (?:
        \#senryu\b
      | \bsenryu\b
      | \bbanzai\b
      | capital\s+district\b[\s\S]{0,120}[\u3040-\u30ff\u3400-\u9fff]
      | [\u3040-\u30ff\u3400-\u9fff][\s\S]{0,120}capital\s+district\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Image alt-text "drought/burnt hills" / "Burnt hillside" — not the town of Burnt Hills.
_BURNT_HILLS_DESCRIPTIVE = re.compile(
    r"""
    (?:
        (?:drought|wildfire|scorched|charred|brown|fire|gorse)\b[\s\S]{0,60}burnt\s+hills
      | burnt\s+hills(?:ide)?\b[\s\S]{0,60}(?:drought|wildfire|scorched|
         charred|fire|gorse|huddersfield)
      | drought\s*/\s*burnt\s+hills
      | burnt\s+hillside\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Roblox game maps named after Rensselaer County — not local news.
_RENSSELAER_ROBLOX = re.compile(
    r"""
    (?:
        \#roblox\b
      | \broblox\b
      | rensselaer\s+county[\s\S]{0,80}\broblox\b
      | \broblox\b[\s\S]{0,80}rensselaer\s+county
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Disney's Saratoga Springs Resort (WDW) — not Saratoga Springs NY.
_DISNEY_SARATOGA = re.compile(
    r"""
    (?:
        disney(?:['\u2019]?s)?\s+saratoga\s+springs
      | saratoga\s+springs\s+resort
      | treehouse\s+villas
      | \bwdwmagic\b
      | wdwmagic\.com
      | disney\s+world
      | \bwdw\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# European Malta / AIS shipping / Gozo film / Netherlands Rotterdam — not
# Town of Malta NY or Town of Rotterdam NY.
_MALTA_EUROPE = re.compile(
    r"""
    (?:
        \bmmsi\b
      | \bvesselalert\b
      | \bais\b
      | flag:\s*malta\b
      | bandera:\s*malta\b
      | dest\.?\s*:\s*rotterdam\b
      | \blmml\b
      | isle\s+of\s+mtv
      | mediterranean
      | \bmalti\b
      | callsign:\s*9ha
      | \b9ha\d+\b
      | \b9h-[a-z0-9]+\b
      | wizz\s+air\s+malta
      | malta\s+international
      | malta\s+sends\b
      | wildfires?\s+in\s+portugal
      | \bportugal\b[\s\S]{0,80}\bmalta\b
      | \bmalta\b[\s\S]{0,80}\bportugal\b
      # Gozo / Lovin Malta film tourism (Troy as film title, not Troy NY).
      | \bgozo\b
      | lovin\s+malta
      | film(?:ed|ing)?\s+on\s+gozo
      | malta\s+as\s+a\s+film
      | \bmgarr\b
      # Malta Tourism / JFK nonstop (Delta Endless Summer cards).
      | malta\s+tourism
      | endless\s+summer[\s\S]{0,60}\bmalta\b
      | \bmalta\b[\s\S]{0,60}endless\s+summer
      | jfk[\s\-–—]*malta
      | malta[\s\-–—]*jfk
      | delta(?:['\u2019]?s)?\s+nonstop[\s\S]{0,60}\bmalta\b
      | \beturbonews\b
      | eturbonews\.com
      # Rotterdam, The Netherlands (architecture / football wire mirrors).
      | \bthe\s+netherlands\b
      | \bnetherlands\b
      | \bfeyenoord\b
      | sparta\s+rotterdam
      | \barchdaily\b
      | archdaily\.com
      # Dutch "New York Pizza" chain bankruptcies (vestigingen / failliet / Rijnmond).
      | new\s+york\s+pizza
      | \brotterdamse\b
      | \bfailliet\b
      | \bvestigingen?\b
      | \brijnmond\b
      | rijnmond\.nl
      | \bdagblad010\b
      | dagblad010\.nl
      | \bomroepdelft\b
      | omroepdelft\.nl
      # International Film Festival Rotterdam (IFFR) — not Town of Rotterdam NY.
      | rotterdam\s+film(?:s)?\b
      | film\s+festival[\s\S]{0,40}\brotterdam\b
      | \biffr\b
      # Book / tourism photos of Hotel New York in Rotterdam.
      | new\s+york\s+hotel[\s\S]{0,60}\brotterdam\b
      | \brotterdam\b[\s\S]{0,60}new\s+york\s+hotel
      | hotel\s+new\s+york[\s\S]{0,60}\brotterdam\b
      # World-city architecture/design calendars list NL Rotterdam next to NYC.
      | archinect\.com
      | architecture\s+(?:&|and)\s+design[\s\S]{0,160}\brotterdam\b
      | \brotterdam\b[\s\S]{0,160}architecture\s+(?:&|and)\s+design
      | \b(?:paris|london|hong\s+kong|detroit)\b[\s\S]{0,220}\brotterdam\b
      | \brotterdam\b[\s\S]{0,220}\b(?:paris|london|hong\s+kong|detroit)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Brunswick Records (jazz/shellac discographies) — not Town of Brunswick NY.
_BRUNSWICK_RECORDS = re.compile(
    r"""
    (?:
        \(\s*brunswick\s*,\s*\d{4}\s*\)
      | brunswick\s+records?\b
      | on\s+brunswick\s+(?:records?\b|78s?\b|label\b)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Brunswick Corporation / Tulsa OK job listings — not Town of Brunswick NY.
_BRUNSWICK_OK = re.compile(
    r"""
    (?:
        brunswick[\s\S]{0,80}\b(?:tulsa|oklahoma|\bok\b)\b
      | \b(?:tulsa|oklahoma)\b[\s\S]{0,80}brunswick
      | brunswick\s*,?\s*ok\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Montclair / San Bernardino "Saratoga Park" — not Saratoga Springs NY.
_SARATOGA_PARK_CA = re.compile(
    r"""
    (?:
        saratoga\s+park[\s\S]{0,160}(?:montclair|san\s+bernardino|\#montclair|\bca\b)
      | (?:montclair|san\s+bernardino|\#montclair)[\s\S]{0,160}saratoga\s+park
      | \#montclairsanbernardinocounty
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# GTA / Liberty City vehicle named Albany — not Albany NY.
_GTA_ALBANY = re.compile(
    r"""
    (?:
        liberty\s+city
      | \bgta\s*iv?\b
      | \brockstar\b[\s\S]{0,120}\balbany\b
      | \balbany\b[\s\S]{0,120}(?:liberty\s+city|\bgta\b|cousin\s+roman)
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

# Troy, South Carolina (NWS Greenville-Spartanburg / Greenwood County).
_TROY_SC = re.compile(
    r"""
    (?:
        greenville[- ]spartanburg
      | \bnws\s+greenville
      | upstate\s+south\s+carolina
      | greenwood\s+county
      | troy\s*,?\s*(?:sc|south\s+carolina)\b
      | (?<![\w.])gsp\.(?:nws|weather)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Galway Ireland (university / county / League of Ireland / tourism itineraries).
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
      | \bireland\b
      | \birish\b
      | day\s+tripping\s+from\s+galway
      | \#visitireland\b
      | \#wildatlanticway\b
      | wild\s+atlantic\s+way
      | \bdingle\b
      | \bkinsale\b
      # LOI fixture lists often omit "Ireland" / "United".
      | \#derrycityfc\b
      | derry\s+city
      | \bdundalk\b
      | \bsligo\b
      | \bshelbourne\b
      | \bshels\b
      | \bbohemians\b
      | \bbohs\b
      | \bdrogheda\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Scotia (Village of Scotia NY) vs Montreal Banque Scotia / Osheaga stage.
_SCOTIA_MONTREAL = re.compile(
    r"""
    (?:
        banque\s+scotia
      | cin[eé]ma\s+banque\s+scotia
      | scotia\s+forest\s+stage
      | parc\s+jean[-\s]?drapeau
      | \bosheaga\b
      | \bmontr[eé]al\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Watervliet, Michigan (jobs / hospital / HS ratings) — not Watervliet NY.
_WATERVLIET_MI = re.compile(
    r"""
    (?:
        watervliet\s*,?\s*(?:mi|michigan)\b
      | watervliet[\s\S]{0,80}\b(?:mi|michigan)\b
      | \b(?:mi|michigan)\b[\s\S]{0,80}watervliet
      # SW Michigan HS ratings cards often omit ", MI".
      | watervliet[\s\S]{0,160}\b(?:bridgman|buchanan|red\s+arrow|bloomingdale)\b
      | \b(?:bridgman|buchanan|red\s+arrow|bloomingdale)\b[\s\S]{0,160}watervliet
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Loudonville, Ohio (hashtag civic lists / high-school soccer) — not Loudonville NY.
_LOUDONVILLE_OH = re.compile(
    r"""
    (?:
        loudonville\s*,?\s*(?:oh|ohio)\b
      | loudonville[\s\S]{0,160}(?:\#ohio\b|\bohio\b|\#oh\d+\b)
      | (?:\#ohio\b|\bohio\b|\#oh\d+\b)[\s\S]{0,160}loudonville
      | \bwaynedale\b
      | golden\s+bears[\s\S]{0,80}loudonville
      | loudonville[\s\S]{0,80}golden\s+bears
      | \bohsaa\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Clifton Park (York / Rotherham, England) — not Clifton Park, NY.
# Watersplash is the Rotherham council paddling pool; handles like
# rotherhamcouncil often omit "Rotherham" in the body.
_CLIFTON_PARK_UK = re.compile(
    r"""
    (?:
        \byorkshire\b
      | \bdurham\b
      | \bcricket(?:er)?s?\b
      | county\s+championship
      | \brotherham\b
      | rotherham\s+show
      | \.gov\.uk\b
      | south\s+yorkshire
      | \bwatersplash\b
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
      # Travel/heritage cards often omit ", PA" (CNN steel town / UNESCO / SteelStacks).
      | bethlehem[\s\S]{0,220}(?:
            steel(?:stacks|town|\s+town)?|unesco|lehigh\s+valley|
            christmas\s+spirit|industrial\s+heritage|snow\s+globe
          )
      | (?:
            steel(?:stacks|town|\s+town)?|unesco|lehigh\s+valley|
            christmas\s+spirit|industrial\s+heritage
          )[\s\S]{0,220}bethlehem
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Cuisine adjective "Schenectady-style …" (LA food trucks) — not the city.
_SCHENECTADY_STYLE = re.compile(
    r'\bschenectady[\s-]+style\b',
    re.IGNORECASE,
)

# Conversational "Firstname and Troy," person address — not the city.
# Keep Cap Region place lists ("Albany and Troy, NY") via the NY/place rescue below.
_TROY_PERSON_NAME = re.compile(
    r"""
    (?:
        \band\s+troy,
      | \bjana\s+and\s+troy\b
      | \btroy\s+deeney\b
      | \btroy\s+johnson\b
      # Bylines / founder intros: "Founder, Troy Johnson".
      | founder[,\s]+troy\b
      # Vocative "Welcome to Mt Kisco, NY, Troy." — not City of Troy.
      | welcome\s+to\b[^.]{0,100},\s*troy\.
      | mt\.?\s*kisco[\s\S]{0,100}\btroy\b
      | \btroy\b[\s\S]{0,100}mt\.?\s*kisco
      # Film / mythic Troy (Achilles, Brad Pitt 2004) — not City of Troy.
      | \btroy\s+achilles\b
      | \bachilles\b[\s\S]{0,40}\btroy\b
      | \btroy\b[\s\S]{0,40}\bachilles\b
      | \bmyrmidons?\b
      | brad\s+pitt[\s\S]{0,80}\btroy\b
      | \btroy\b[\s\S]{0,80}brad\s+pitt
      | \btroy\s*\(\s*2004\s*\)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Troy, Pennsylvania (Bradford County / NWS Binghamton) — not City of Troy NY.
_TROY_PA = re.compile(
    r"""
    (?:
        troy\s*,?\s*(?:pa|pennsylvania)\b
      | \bbradford\s+county\b
      | western\s+bradford
      # NWS wording varies: "over Troy" or "near Troy" after Springfield.
      | over\s+springfield,\s+or\s+(?:over|near)\s+troy\b
      | \btroy\b[\s\S]{0,120}\bbradford\b
      | \bbradford\b[\s\S]{0,120}\btroy\b
      | \btroy\b[\s\S]{0,100}\bnortheastern\s+pennsylvania\b
      | \bnortheastern\s+pennsylvania\b[\s\S]{0,100}\btroy\b
      # NWS Binghamton CWA covers Troy PA; Troy NY is Albany CWA.
      | nws\s+binghamton[\s\S]{0,360}\b(?:over|near)\s+troy\b
      | \b(?:over|near)\s+troy\b[\s\S]{0,360}nws\s+binghamton
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Ithaca / Tompkins "Troy Road" solar/planning wires — not City of Troy NY.
# Hashtag alternatives must not sit behind ``\b`` (``#`` is non-word).
_TROY_ROAD_ITHACA = re.compile(
    r"""
    (?:
        troy\s+road\b[\s\S]{0,400}(?:
            \b(?:ithaca|tompkins)\b|\#eastithaca(?:tompkinscounty)?\b
          )
      | (?:
            \b(?:ithaca|tompkins)\b|\#eastithaca(?:tompkinscounty)?\b
          )[\s\S]{0,400}troy\s+road\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Schaghticoke Road in Kent, CT — not the Town of Schaghticoke NY.
_SCHAGHTICOKE_CT = re.compile(
    r"""
    (?:
        schaghticoke\s+(?:rd\.?|road|ave(?:nue)?|st(?:reet)?)\b[\s\S]{0,120}\b(?:
            ct\b|connecticut|\#connecticut\b|kent
          )
      | \b(?:ct\b|connecticut|\#connecticut\b|kent)\b[\s\S]{0,120}schaghticoke\s+(?:
            rd\.?|road|ave(?:nue)?|st(?:reet)?
          )\b
      | schaghticoke[\s\S]{0,120}\#(?:connecticut|newengland)\b
      | \#(?:connecticut|newengland)\b[\s\S]{0,120}schaghticoke
      | schaghticoke[\s\S]{0,80}\bkent\b[\s\S]{0,40}\bct\b
      | \bkent\b[\s\S]{0,40}\bct\b[\s\S]{0,80}schaghticoke
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Waterford, Connecticut (New London ferry / Hartford Tpke) — not Waterford NY.
_WATERFORD_CT = re.compile(
    r"""
    (?:
        waterford\s*,?\s*(?:ct|connecticut)\b
      | waterford[\s\S]{0,160}\b(?:new\s+london|hartford\s+tpke|connecticut)\b
      | \b(?:new\s+london|hartford\s+tpke|connecticut)\b[\s\S]{0,160}waterford
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
      # Georgia HBCU — not University at Albany / SUNY Albany.
      | albany\s+state\s+university
      # California city / racetrack (not Delmar, NY).
      | \bdel\s+mar\b
      # Street name elsewhere (e.g. Rochester / Salem IL) — not the Town of Delmar.
      | delmar\s+(?:st(?:reet)?|ave(?:nue)?)\b
      # Brooklyn / NYC street — not the City of Troy.
      | troy\s+(?:ave(?:nue)?|st(?:reet)?)\b
      # Brooklyn subway (Saratoga Av on the 3) — not Saratoga Springs.
      | saratoga\s+av(?:e(?:nue)?)?\b
      # PubMed journal abbreviation — not Albany NY local news.
      | aging\s*\(\s*albany\s*ny\s*\)
      # National politician, not Troy NY.
      | \btroy\s+jackson\b
      | national\s+capital\s+region
      | national\s+capital\s+district
      # Slash form appears in Brussels Times cards ("Brussels/Capital Region").
      | brussels[- /]capital\s+region
      | canadian\s+capital\s+region
      | capital\s+region\s+bureau\b
      # Putnam / Hudson Valley street — not City of Albany.
      | albany\s+post\s+road
      # Cape Town suburb / wine region / municipal alerts — not Helderberg Escarpment.
      # Prefer #capetown / #southafrica / Helderberg College; window covers text→hashtag.
      | helderberg\s+college\b
      | helderberg[\s\S]{0,280}(?:
            cape\s+town|\#capetown\b|western\s+cape|south\s+africa|
            \#southafrica\b|\#africa\b|\bstrand\b|sweet\s+paws|pressportal\.co\.za
          )
      | (?:
            cape\s+town|\#capetown\b|western\s+cape|south\s+africa|
            \#southafrica\b|\#africa\b|sweet\s+paws|pressportal\.co\.za
          )[\s\S]{0,280}helderberg
      # Seattle / New York Times guild phrasing — not Albany Times Union.
      | seattle\s+times(?:\s+union)?\b
      | seattle[\s\S]{0,100}times\s+union\b
      | times\s+union[\s\S]{0,100}seattle
      | new\s+york\s+times(?:\s+union)?\b
      | times\s+union[\s\S]{0,100}new\s+york\s+times
      # SoCal / multi-track handicap hashtag stuffing — not Race Course NY.
      | (?:\#socal\b|\#losangeles\b|southern\s+california)[\s\S]{0,200}\#saratoga\b
      | \#saratoga\b[\s\S]{0,200}(?:\#socal\b|\#losangeles\b|southern\s+california)
      | (?:\#parxracing\b|\#thistledown\b|\#horseshoeindy\b|
         \#assiniboia(?:downs)?\b|\#mountaineerpark\b|
         \#woodbine\b|\#charlestownraces\b|\#remingtonpark\b)[\s\S]{0,200}\#saratoga\b
      | \#saratoga\b[\s\S]{0,200}(?:\#parxracing\b|\#thistledown\b|
         \#horseshoeindy\b|\#assiniboia(?:downs)?\b|\#mountaineerpark\b|
         \#woodbine\b|\#charlestownraces\b|\#remingtonpark\b)
      # Long Island SEO hashtag stuffing with #albany / #albanyny — not Cap Region.
      | (?:\#longisland(?:ny|newyork)?\b|\#longislandrealestate\b)[\s\S]{0,200}\#albany(?:ny|_ny)?\b
      | \#albany(?:ny|_ny)?\b[\s\S]{0,200}(?:\#longisland(?:ny|newyork)?\b|\#longislandrealestate\b)
      # Binghamton housing development — not Saratoga Race Course / Springs.
      | saratoga\s+terrace\b
      # Louisiana MPO — not NY Capital District Regional Planning Commission (CDRPC).
      | capital\s+region\s+planning\s+commission
      # Berlin-Brandenburg / Germany "capital region" (Medienboard wires).
      | berlin[- /]brandenburg\s+capital\s+region
      | berlin[- ]brandenburg[\s\S]{0,80}capital\s+region\b
      | capital\s+region\b[\s\S]{0,80}berlin[- ]brandenburg
      # NJ street — not Town of Brunswick NY.
      | brunswick\s+pike\b
      | capital\s+region\s+of\s+(?:
            madrid|spain|belgium|brussels|paris|france|berlin|germany|
            tokyo|seoul|beijing|delhi|ottawa|canberra|rome|italy|
            amsterdam|vienna|warsaw|prague|lisbon|athens|dublin|
            canada|denmark|copenhagen|mississippi|louisiana|pennsylvania|
            california|sacramento|korea|south\s+korea|ukraine|kyiv|kiev|
            sudan|khartoum|virginia|richmond|colombia|bogot[aá]|
            iceland|reykjav[ií]k|bulgaria|sofia|japan|michigan|lansing
          )\b
      | ukrainian\s+capital\s+region
      | sudan(?:ese)?\s+capital\s+region
      | icelandic\s+capital\s+(?:region|district)
      | (?:bulgarian?|sofia)\s+capital\s+region
      | (?:japanese?|tokyo)\s+capital\s+region
      | michigan\s+capital\s+region
      | capital\s+region\s+international\s+airport
      | liberty\s+city
      | \bgta\s*iv?\b
      | capital\s+district\s+fire\s+and\s+rescue
      | (?:the\s+)?egg\s+and\s+art\s+garden
      | art\s+garden\s+kc\b
      | (?:virginia|richmond)\s+capital\s+region
      | bogot[aá]\s+capital\s+district
      | capital\s+district\s*,?\s*colombia\b
      | capital\s+region\s+water\b
      | pennsylvania\s+capital\s+region
      | hauptstadtregion
      # Law firm — not Town of Latham NY HQ wires.
      | latham\s*(?:&|and)\s*watkins
      # Oregon 5A Mid-Willamette / South Albany & West Albany high schools.
      | mid[- ]?willamette
      | west\s+albany\s+high(?:\s+school)?
      | south\s+albany\s+high(?:\s+school)?
      | (?:south|west)\s+albany\b[\s\S]{0,200}(?:
            \boregon\b|\#oregon\b|corvallis|crescent\s+valley|
            \blebanon\b|silverton|mid[- ]?willamette|\.oregonlive\.
          )
      | (?:
            \boregon\b|\#oregon\b|corvallis|crescent\s+valley|
            \blebanon\b|silverton|mid[- ]?willamette|\.oregonlive\.
          )[\s\S]{0,200}(?:south|west)\s+albany\b
      | jc\s+latham
      | saratoga\s+springs\s*,\s*ut\b
      | saratoga\s+springs\s+ut\b
      | disney(?:['\u2019]?s)?\s+saratoga\s+springs
      | watervliet\s*,?\s*(?:mi|michigan)\b
      | troy\s*,?\s*(?:mi|michigan)\b
      | detroit\s*/\s*troy\b
      | loudonville\s*,?\s*(?:oh|ohio)\b
      | reinvent\s*albany
      | rowonebrand
      # Radio-market lists often omit the comma ("Albany GA").
      | albany\s+ga\b
      | waterford\s+crystal
      | waterford\s+wedgwood
      | rensselaer\s*,?\s*(?:in|indiana)\b
      | helen\s+of\s+troy
      | colonie\s+de\s+vacances
      | colonie\s+num[eé]rique
      | une\s+colonie
      | m[eê]me\s+colonie
      | university\s+of\s+galway
      | galway\s*,?\s*ireland\b
      | galway\s+united\b
      | galway\s+fc\b
      | league\s+of\s+ireland
      | van\s+helderbergh
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
        # "Sold out night at Proctors" is a venue event without "tickets".
        sold\s+out|
        open\s+mic|festival|concert|orchestra|philharmonic|comedy\s+night|show\s+starts|
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
      | \bthe\s+egg\s+presents\b
      | albany\s+palace\s+(?:theatre|theater)
      | palace\s+(?:theatre|theater)\s+(?:albany|in\s+albany)
      | capital\s+repertory
      | \bcap\s+rep\b
      | albany\s+civic\s+(?:theater|theatre)
      | park\s+playhouse\b
      | empire\s+underground\b
      | \bempac\b
      | experimental\s+media\s+and\s+performing\s+arts\s+center\b
      | joseph\s+l\.?\s+bruno\s+stadium
      | \bbruno\s+stadium\b
      | tri-?city\s+valleycats\b
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
      | albany\s*,\s*(?:wy|wyoming)\b
      # Laramie public library branding — Cap Region uses Albany Public Library.
      | albany\s+county\s+library\b
      # NWS bots tag state as [WY] / #WYwx; Laramie Valley is WY-only context.
      | albany\s+county[\s\S]{0,240}(?:
            \bwyoming\b|\#wy\b|\#wywx\b|\[wy\]|\blaramie\b|cheyenne|nws\s+cheyenne
        )
      | (?:
            \#wy\b|\#wywx\b|\bwyoming\b|\[wy\]|\blaramie\b|cheyenne|nws\s+cheyenne
        )[\s\S]{0,240}albany\s+county
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _albany_county_wy_conflict(haystack: str, author_handle: str | None = None) -> bool:
    """True when Albany County cues point at Wyoming, not NY."""
    handle = (author_handle or '').strip().lower()
    wy_handle = bool(re.search(r'wyomingpublicmedia|\bwpm\b', handle))
    if not _ALBANY_COUNTY_WY.search(haystack) and not (
        wy_handle and re.search(r'albany\s+county', haystack, flags=re.IGNORECASE)
    ):
        return False
    # Explicit NY state context wins (comparison posts, etc.).
    if re.search(r'\b(?:ny|new\s+york)\b', haystack, flags=re.IGNORECASE):
        return False
    return True


def _ny_capital_region_context(haystack: str) -> bool:
    """True when haystack has explicit NY Cap Region anchors."""
    return (
        re.search(
            rf"""
            (?:
              # Mirror _NY_CONTEXT: reject digit-prefixed aircraft suffixes (271NY)
              # and Unicode letters (e.g. Hungarian tartomány …ny).
                (?-i:(?<!\w)(?:NY|ny)(?!\w))
              | \b(?:nys|new\s+york)\b
              | n\.y\.
              | hudson\s+valley
              | \#albanyny\b
              | albany\s*,?\s*(?:ny|n\.y\.|new\s+york)
              | schenectady
              | {_TROY_PLACE}
            )
            """,
            haystack,
            flags=re.IGNORECASE | re.VERBOSE,
        )
        is not None
    )


def _canadian_capital_region_conflict(haystack: str, author_handle: str | None = None) -> bool:
    """True when 'capital region' refers to Ottawa/Canada/BC, not NY."""
    if _ny_capital_region_context(haystack):
        return False
    if _CANADIAN_CAPITAL_REGION.search(haystack):
        return True
    # Times Colonist / CFAX / Ottawa Citizen / Island rail cards often omit domain cues.
    handle = (author_handle or '').strip().lower()
    if re.search(r'timescolonist|\bcfax|ottawacitizen|restoreislandrail', handle) and re.search(
        r'capital\s+region\b', haystack, flags=re.IGNORECASE
    ):
        return True
    return False


def _md_dc_capital_region_conflict(haystack: str, author_handle: str | None = None) -> bool:
    """True when 'capital region/district' refers to MD/DC metro, not NY."""
    if _ny_capital_region_context(haystack):
        return False
    if _MD_DC_CAPITAL_REGION.search(haystack):
        return True
    # Maryland Banner / MoCo community media often omit "Maryland" in the body.
    handle = (author_handle or '').strip().lower()
    if re.search(r'banner(?:moco|pgcounty)|marylandbanner|\bmymcmedia\b', handle) and re.search(
        r'capital\s+(?:region|district)\b', haystack, flags=re.IGNORECASE
    ):
        return True
    return False


def _png_capital_district_conflict(haystack: str) -> bool:
    """True when 'capital district' refers to Port Moresby / PNG, not NY."""
    if not _PNG_CAPITAL_DISTRICT.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _albany_bay_area_conflict(haystack: str) -> bool:
    """True when bare Albany/Saratoga are Bay Area CA cities, not NY."""
    if re.search(
        r'(?:albany|saratoga(?:\s+springs)?)\s*,?\s*(?:ny|n\.y\.|new\s+york)|'
        r'\#albanyny\b|\#saratogasprings\b',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return _BAY_AREA_ALBANY.search(haystack) is not None


def _stillwater_film_conflict(haystack: str) -> bool:
    """True when Stillwater refers to the Matt Damon film, not Stillwater NY."""
    if not re.search(r'\bstillwater\b', haystack, flags=re.IGNORECASE):
        return False
    if not _STILLWATER_FILM.search(haystack):
        return False
    if re.search(
        r'stillwater\s*,?\s*(?:ny|n\.y\.|new\s+york)\b|town\s+of\s+stillwater',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _stillwater_road_conflict(haystack: str, author_handle: str | None = None) -> bool:
    """True when Stillwater is a North Country street / Lewis Co spotter, not Town NY."""
    if not re.search(r'\bstillwater\b', haystack, flags=re.IGNORECASE):
        return False
    handle = (author_handle or '').strip().lower()
    handle_hit = bool(re.search(r'(?<![a-z0-9])buf\.(?:nws|weather)', handle))
    if not (_STILLWATER_ROAD_NORTH.search(haystack) or handle_hit):
        return False
    if re.search(
        r'stillwater\s*,?\s*(?:ny|n\.y\.|new\s+york)\b|town\s+of\s+stillwater|'
        r'saratoga\s+county|\#albanyny\b|nws\s+albany',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _troy_michigan_conflict(haystack: str) -> bool:
    """True when Troy refers to the Detroit suburb, not Troy NY."""
    if not re.search(r'\btroy\b', haystack, flags=re.IGNORECASE):
        return False
    if not _TROY_MICHIGAN.search(haystack):
        return False
    if re.search(
        r'troy\s*,?\s*(?:ny|n\.y\.|new\s+york)\b|city\s+of\s+troy|'
        r'troy\s+(?:music\s+hall|savings\s+bank)',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _troy_sc_conflict(haystack: str, author_handle: str | None = None) -> bool:
    """True when Troy refers to Troy SC (NWS GSP), not Troy NY."""
    if not re.search(r'\btroy\b', haystack, flags=re.IGNORECASE):
        return False
    handle = (author_handle or '').strip().lower()
    handle_hit = bool(re.search(r'(?<![a-z0-9])gsp\.(?:nws|weather)', handle))
    if not handle_hit and not _TROY_SC.search(haystack):
        return False
    if re.search(
        r'troy\s*,?\s*(?:ny|n\.y\.|new\s+york)\b|city\s+of\s+troy|'
        r'troy\s+(?:music\s+hall|savings\s+bank)',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _indiana_albany_saratoga_conflict(haystack: str, distinct: set[str]) -> bool:
    """True when Albany+Saratoga are Indiana weather towns, not NY Cap Region."""
    if not ({'albany', 'saratoga', 'saratoga springs'} & distinct):
        return False
    if len({'albany', 'saratoga', 'saratoga springs'} & distinct) < 2 and 'albany' not in distinct:
        return False
    # Require Albany paired with a Saratoga token (multi-local Indiana towns).
    if 'albany' not in distinct or not ({'saratoga', 'saratoga springs'} & distinct):
        return False
    if not _IN_ALBANY_SARATOGA.search(haystack):
        return False
    if re.search(
        r'(?:albany|saratoga(?:\s+springs)?)\s*,?\s*(?:ny|n\.y\.|new\s+york)\b|'
        r'\#albanyny\b|\#nywx\b|nws\s+albany',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _california_capital_region_conflict(haystack: str, author_handle: str | None = None) -> bool:
    """True when 'capital region' refers to Sacramento / CA, not NY."""
    if _ny_capital_region_context(haystack):
        return False
    if _CA_CAPITAL_REGION.search(haystack):
        return True
    handle = (author_handle or '').strip().lower()
    if re.search(r'\bsacbee\b', handle) and re.search(
        r'capital\s+region\b', haystack, flags=re.IGNORECASE
    ):
        return True
    return False


def _korea_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to Seoul / Gyeonggi, not NY."""
    if not _KR_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _ukraine_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to Kyiv / Ukraine, not NY."""
    if not _UA_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _russia_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to Moscow / Russia, not NY."""
    if not _RU_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _sudan_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to Khartoum / Sudan, not NY."""
    if not _SD_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _virginia_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to Richmond / Virginia, not NY."""
    if not _VA_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _virginia_capital_district_conflict(haystack: str) -> bool:
    """True when 'capital district' refers to VHSL Richmond-area athletics, not NY."""
    if not _VA_CAPITAL_DISTRICT.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _germany_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to Berlin-Brandenburg / Germany, not NY."""
    if not _DE_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _colombia_capital_district_conflict(haystack: str) -> bool:
    """True when 'capital district' refers to Bogotá / Colombia, not NY."""
    if not _CO_CAPITAL_DISTRICT.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _iceland_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region/district' refers to Reykjavik / Iceland, not NY."""
    if not _IS_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _france_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to Paris / France, not NY."""
    if not _FR_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _belgium_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to Brussels / Belgium, not NY."""
    if not _BE_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _india_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to Delhi / India, not NY."""
    if not _IN_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _newtonville_ma_conflict(haystack: str) -> bool:
    """True when Newtonville refers to Boston Newton / MBTA, not Colonie NY."""
    if not re.search(r'\bnewtonville\b', haystack, flags=re.IGNORECASE):
        return False
    if not _NEWTONVILLE_MA.search(haystack):
        return False
    if re.search(
        r'newtonville\s*,?\s*(?:ny|n\.y\.|new\s+york)\b|\#albanyny\b|colonie',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return not _ny_capital_region_context(haystack)


def _ct_capital_district_conflict(haystack: str) -> bool:
    """True when 'capital district' refers to CT bank territory, not NY."""
    if not _CT_CAPITAL_DISTRICT.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _green_island_other_conflict(haystack: str) -> bool:
    """True when Green Island refers to LI sangha / Adelphi, not the village."""
    if not re.search(r'green\s+island', haystack, flags=re.IGNORECASE):
        return False
    if not _GREEN_ISLAND_OTHER.search(haystack):
        return False
    if re.search(
        r'green\s+island\s*,?\s*(?:ny|n\.y\.|new\s+york)\b|village\s+of\s+green\s+island',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _brunswick_schools_other_conflict(haystack: str) -> bool:
    """True when Brunswick Schools refers to OH + NYC events, not Brunswick NY."""
    if not re.search(r'\bbrunswick\b', haystack, flags=re.IGNORECASE):
        return False
    if not _BRUNSWICK_SCHOOLS_OTHER.search(haystack):
        return False
    if re.search(
        r'brunswick\s*,?\s*(?:ny|n\.y\.|new\s+york)\b|town\s+of\s+brunswick',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _egg_kansas_city_conflict(haystack: str) -> bool:
    """True when 'The Egg' refers to Kansas City's Art Garden venue, not Albany."""
    if not re.search(r'\bthe\s+egg\b|\begg\s+and\s+art\s+garden\b', haystack, flags=re.IGNORECASE):
        return False
    if re.search(
        r'(?:the\s+egg|egg)\s*,?\s*(?:albany|ny|n\.y\.)|'
        r'\balbany\b[\s\S]{0,80}\bthe\s+egg\b|'
        r'\bthe\s+egg\b[\s\S]{0,80}\balbany\b|'
        r'\#albanyny\b|empire\s+state\s+plaza',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return _EGG_KANSAS_CITY.search(haystack) is not None


def _finland_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to Helsinki / Finland, not NY."""
    if not _FI_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _denmark_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to Copenhagen / Denmark, not NY."""
    if not _DK_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _alberta_capital_region_conflict(haystack: str, author_handle: str | None = None) -> bool:
    """True when 'capital region' refers to Alberta / Edmonton metro, not NY."""
    if _ny_capital_region_context(haystack):
        return False
    if _AB_CAPITAL_REGION.search(haystack):
        return True
    # Handles like barbh-ab.bsky.social often omit "Alberta" in the body.
    handle = (author_handle or '').strip().lower()
    if re.search(r'(?:^|[.-])ab(?:[.-]|$)', handle) and re.search(
        r'capital\s+region\b', haystack, flags=re.IGNORECASE
    ):
        return True
    return False


def _bulgaria_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to Sofia / Bulgaria, not NY."""
    if not _BG_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _japan_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to Tokyo metro / Japan, not NY."""
    if not _JP_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _michigan_capital_region_conflict(haystack: str, author_handle: str | None = None) -> bool:
    """True when 'capital region' refers to Lansing MI airport metro, not NY."""
    if _ny_capital_region_context(haystack):
        return False
    if _MI_CAPITAL_REGION.search(haystack):
        return True
    handle = (author_handle or '').strip().lower()
    # NWS Grand Rapids bots (grr.nws-bot.us) often omit Lansing in short text.
    if re.search(r'(?<![a-z0-9])grr(?:\.|nws|-)', handle) and re.search(
        r'capital\s+region\b', haystack, flags=re.IGNORECASE
    ):
        return True
    return False


def _australia_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to Canberra / ACT, not NY."""
    if not _AU_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _georgia_atlanta_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to Atlanta / Georgia, not NY."""
    if not _GA_ATLANTA_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _japan_capital_district_conflict(haystack: str) -> bool:
    """True when 'capital district' is Japanese poetry/translation, not NY."""
    if not re.search(r'capital\s+district\b', haystack, flags=re.IGNORECASE):
        return False
    if not _JP_CAPITAL_DISTRICT.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _burnt_hills_descriptive_conflict(haystack: str) -> bool:
    """True when 'burnt hills' describes drought/wildfire terrain, not the town."""
    if not re.search(r'burnt\s+hills', haystack, flags=re.IGNORECASE):
        return False
    if not _BURNT_HILLS_DESCRIPTIVE.search(haystack):
        return False
    if re.search(
        r'burnt\s+hills\s*,?\s*(?:ny|n\.y\.|new\s+york)\b|'
        r'town\s+of\s+burnt\s+hills|burnt\s+hills-ballston',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return not _ny_capital_region_context(haystack)


def _rensselaer_roblox_conflict(haystack: str) -> bool:
    """True when Rensselaer County refers to a Roblox map, not NY locality."""
    if not re.search(r'rensselaer\s+county', haystack, flags=re.IGNORECASE):
        return False
    if not _RENSSELAER_ROBLOX.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _disney_saratoga_conflict(haystack: str, author_handle: str | None = None) -> bool:
    """True when Saratoga Springs refers to Disney World, not NY."""
    if not re.search(r'\bsaratoga\b', haystack, flags=re.IGNORECASE):
        return False
    if _ny_capital_region_context(haystack):
        return False
    if _DISNEY_SARATOGA.search(haystack):
        return True
    handle = (author_handle or '').strip().lower()
    return bool(re.search(r'wdwmagic|\bdisney', handle))


def _louisiana_capital_region_conflict(haystack: str, author_handle: str | None = None) -> bool:
    """True when 'capital region' refers to Baton Rouge / LA, not NY."""
    if _ny_capital_region_context(haystack):
        return False
    if _LA_CAPITAL_REGION.search(haystack):
        return True
    # WBRZ mirrors (wbrz, wbrznews2, …) often omit "Baton Rouge" in short copy.
    handle = (author_handle or '').strip().lower()
    if re.search(r'(?<![a-z0-9])wbrz', handle) and re.search(
        r'capital\s+region\b', haystack, flags=re.IGNORECASE
    ):
        return True
    return False


def _pennsylvania_capital_region_conflict(haystack: str) -> bool:
    """True when 'capital region' refers to Harrisburg / PA, not NY."""
    if not _PA_CAPITAL_REGION.search(haystack):
        return False
    return not _ny_capital_region_context(haystack)


def _malta_europe_conflict(haystack: str) -> bool:
    """True when Malta/Rotterdam refer to the EU island / AIS shipping, not NY.

    Do not use bare Troy / New York as NY anchors here: film titles ("Troy") and
    firm bylines ("ODA New York") collide with Cap Region tokens while describing
    Gozo shoots or Rotterdam architecture.
    """
    if not re.search(r'\bmalta\b|\brotterdam\b', haystack, flags=re.IGNORECASE):
        return False
    if not _MALTA_EUROPE.search(haystack):
        return False
    if re.search(
        r'(?:malta|rotterdam)\s*,?\s*(?:ny|n\.y\.|new\s+york)\b|'
        r'town\s+of\s+(?:malta|rotterdam)',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


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
    if re.search(
        r'galway\s*,?\s*(?:ny|n\.y\.)\b|town\s+of\s+galway',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _scotia_montreal_conflict(haystack: str) -> bool:
    """True when Scotia refers to Montreal Banque Scotia / Osheaga, not NY."""
    if not re.search(r'\bscotia\b', haystack, flags=re.IGNORECASE):
        return False
    if not _SCOTIA_MONTREAL.search(haystack):
        return False
    if re.search(
        r'scotia\s*,?\s*(?:ny|n\.y\.|new\s+york)\b|town\s+of\s+scotia|'
        r'village\s+of\s+scotia',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _watervliet_mi_conflict(haystack: str) -> bool:
    """True when Watervliet refers to Michigan, not the city in NY."""
    if not re.search(r'\bwatervliet\b', haystack, flags=re.IGNORECASE):
        return False
    if not _WATERVLIET_MI.search(haystack):
        return False
    if re.search(
        r'watervliet\s*,?\s*(?:ny|n\.y\.|new\s+york)\b',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _loudonville_oh_conflict(haystack: str, author_handle: str | None = None) -> bool:
    """True when Loudonville refers to Ohio, not the hamlet near Albany."""
    if not re.search(r'\bloudonville\b', haystack, flags=re.IGNORECASE):
        return False
    handle = (author_handle or '').strip().lower()
    # Waynedale HS accounts often omit Ohio / Waynedale from the body.
    handle_hit = bool(re.search(r'waynedale', handle))
    if not (_LOUDONVILLE_OH.search(haystack) or handle_hit):
        return False
    if re.search(
        r'loudonville\s*,?\s*(?:ny|n\.y\.|new\s+york)\b',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return not _ny_capital_region_context(haystack)


def _brunswick_records_conflict(haystack: str) -> bool:
    """True when Brunswick refers to the record label, not Town of Brunswick NY."""
    if not re.search(r'\bbrunswick\b', haystack, flags=re.IGNORECASE):
        return False
    if not _BRUNSWICK_RECORDS.search(haystack):
        return False
    if re.search(
        r'brunswick\s*,?\s*(?:ny|n\.y\.|new\s+york)\b|town\s+of\s+brunswick',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _brunswick_ok_conflict(haystack: str) -> bool:
    """True when Brunswick refers to Tulsa OK / corp jobs, not Town of Brunswick NY."""
    if not re.search(r'\bbrunswick\b', haystack, flags=re.IGNORECASE):
        return False
    if not _BRUNSWICK_OK.search(haystack):
        return False
    if re.search(
        r'brunswick\s*,?\s*(?:ny|n\.y\.|new\s+york)\b|town\s+of\s+brunswick',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _saratoga_park_ca_conflict(haystack: str) -> bool:
    """True when Saratoga Park refers to Montclair CA, not Saratoga Springs NY."""
    if not re.search(r'\bsaratoga\b', haystack, flags=re.IGNORECASE):
        return False
    if not _SARATOGA_PARK_CA.search(haystack):
        return False
    if re.search(
        r'saratoga(?:\s+springs)?\s*,?\s*(?:ny|n\.y\.|new\s+york)\b',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return not _ny_capital_region_context(haystack)


def _gta_albany_conflict(haystack: str) -> bool:
    """True when Albany refers to a GTA / Liberty City vehicle, not Albany NY."""
    if not re.search(r'\balbany\b', haystack, flags=re.IGNORECASE):
        return False
    if not _GTA_ALBANY.search(haystack):
        return False
    if re.search(
        r'albany\s*,?\s*(?:ny|n\.y\.|new\s+york)\b|#albanyny\b',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _bethlehem_pa_conflict(haystack: str) -> bool:
    """True when Bethlehem refers to Pennsylvania, not Town of Bethlehem NY."""
    if not re.search(r'\bbethlehem\b', haystack, flags=re.IGNORECASE):
        return False
    if not _BETHLEHEM_PA.search(haystack):
        return False
    if re.search(
        r'bethlehem\s*,?\s*(?:ny|n\.y\.)\b|town\s+of\s+bethlehem',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _schenectady_style_conflict(haystack: str) -> bool:
    """True when Schenectady is a cuisine adjective (…-style), not the city."""
    return bool(_SCHENECTADY_STYLE.search(haystack))


_SCHENECTADY_REMOTE_HASHTAGS = re.compile(
    r"""
    \#(?:
        maryland|wisconsin|louisiana|arkansas|oshkosh|frederick|
        portland|maine|palo-?alto|bossier(?:-city)?|camarilla|
        pittsburg|creek|south-sanfrancisco|north-littlerock|davis
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _schenectady_hashtag_spam_conflict(haystack: str) -> bool:
    """True when Schenectady appears in multi-state hashtag stuffing, not local copy."""
    if not re.search(r'\#schenectady\b|\bschenectady\b', haystack, flags=re.IGNORECASE):
        return False
    remote = {m.group(0).lower() for m in _SCHENECTADY_REMOTE_HASHTAGS.finditer(haystack)}
    if len(remote) < 2:
        return False
    if re.search(
        r"""
        (?:
            schenectady\s*,?\s*(?:ny|n\.y\.|new\s+york)\b
          | schenectady\s+county\b
          | \bproctors\b
          | \#albanyny\b
          | capital\s+(?:region|district)\b
        )
        """,
        haystack,
        flags=re.IGNORECASE | re.VERBOSE,
    ):
        return False
    return True


def _albany_bandscan_conflict(haystack: str) -> bool:
    """True when bare Albany is an FM DX / bandscan market list unlocked by NYC."""
    if not re.search(
        r'\#fmdx\b|bandscan|fm\s+dx\b|fm\s+band\s*scan',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    if not re.search(r'\balbany\b', haystack, flags=re.IGNORECASE):
        return False
    if re.search(
        r"""
        (?:
            albany\s*,?\s*(?:ny|n\.y\.|new\s+york)\b
          | \#albanyny\b
          | capital\s+(?:region|district)\b
          | \bschenectady\b
          | \btroy\s*,?\s*(?:ny|n\.y\.)\b
        )
        """,
        haystack,
        flags=re.IGNORECASE | re.VERBOSE,
    ):
        return False
    return True


def _troy_person_name_conflict(haystack: str) -> bool:
    """True when Troy is a person name / Westchester vocative, not the city."""
    if not _TROY_PERSON_NAME.search(haystack):
        return False
    # Real place mentions still keep. Bare "in Troy" is too loose for film titles
    # like "Brad Pitt in Troy (2004)".
    if re.search(
        r"""
        (?:
            \btroy\s*,?\s*(?:ny|n\.y\.|new\s+york)\b
          | city\s+of\s+troy
          | troy\s+(?:street|avenue|ave)\b
          | \balbany\s+and\s+troy\b
        )
        """,
        haystack,
        flags=re.IGNORECASE | re.VERBOSE,
    ):
        return False
    return True


def _troy_pa_conflict(haystack: str) -> bool:
    """True when Troy refers to Bradford County PA, not City of Troy NY."""
    if not re.search(r'\btroy\b', haystack, flags=re.IGNORECASE):
        return False
    if not _TROY_PA.search(haystack):
        return False
    if re.search(
        r"""
        (?:
            \btroy\s*,?\s*(?:ny|n\.y\.|new\s+york)\b
          | city\s+of\s+troy
          | \bin\s+troy,\s*n
        )
        """,
        haystack,
        flags=re.IGNORECASE | re.VERBOSE,
    ):
        return False
    return True


def _troy_road_ithaca_conflict(haystack: str) -> bool:
    """True when Troy Road is an Ithaca/Tompkins street, not City of Troy NY."""
    if not re.search(r'troy\s+road\b', haystack, flags=re.IGNORECASE):
        return False
    if not _TROY_ROAD_ITHACA.search(haystack):
        return False
    if re.search(
        r"""
        (?:
            \btroy\s*,?\s*(?:ny|n\.y\.|new\s+york)\b
          | city\s+of\s+troy
          | \bin\s+troy\b
        )
        """,
        haystack,
        flags=re.IGNORECASE | re.VERBOSE,
    ):
        return False
    return True


_ALBANY_WIRE_REMOTE = re.compile(
    r"""
    (?:
        \balbuquerque\b
      | \bnew\s+mexico\b
      | \bphoenix\b
      | \barizona\b
      | \bdenver\b
      | \bcolorado\b
      | \bhouston\b
      | \bdallas\b
      | \bsan\s+francisco\b
      | \blos\s+angeles\b
      | \bseattle\b
      | \bmiami\b
      | \batlanta\b
      | \bminneapolis\b
      | \bdetroit\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ALBANY_WIRE_LOCAL_RESCUE = re.compile(
    r"""
    (?:
        empire\s+state\s+plaza
      | lark\s+(?:street|hall)\b
      | \bualbany\b
      | university\s+at\s+albany
      | capital\s+(?:region|district)\b
      | \bschenectady\b
      | \btroy\s*,?\s*(?:ny|n\.y\.)\b
      | city\s+of\s+(?:albany|troy)\b
      | albany\s+medical
      | mvp\s+arena
      | \bproctors\b
      | empire\s+underground\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _albany_wire_remote_conflict(haystack: str) -> bool:
    """True when an Albany, N.Y. wire dateline fronts a remote-city story.

    Press wires often open with the HQ dateline (ALBANY, N.Y.) while the body
    is about an Albuquerque / Phoenix / etc. facility. Keep Cap Region body
    cues as rescue so local Albany copy is not dropped.
    """
    if not re.search(
        r'\balbany,?\s*(?:n\.?\s*y\.?|new\s+york)\b',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    if not _ALBANY_WIRE_REMOTE.search(haystack):
        return False
    if _ALBANY_WIRE_LOCAL_RESCUE.search(haystack):
        return False
    return True


def _albany_ithaca_contrast_conflict(haystack: str) -> bool:
    """True when Albany is only a contrast to Ithaca/Tompkins, not Cap Region news."""
    if not re.search(r'\balbany\b', haystack, flags=re.IGNORECASE):
        return False
    if not re.search(r'\b(?:ithaca|tompkins(?:\s+county)?)\b', haystack, flags=re.IGNORECASE):
        return False
    # Explicit Cap Region anchors (beyond bare Albany + New York) keep.
    if re.search(
        r"""
        (?:
            albany\s*,?\s*(?:ny|n\.y\.)\b
          | \bualbany\b
          | university\s+at\s+albany
          | capital\s+(?:region|district)\b
          | \bschenectady\b
          | empire\s+state\s+plaza
          | new\s+york\s+state\s+capitol
          | state\s+capitol\s+in\s+albany
        )
        """,
        haystack,
        flags=re.IGNORECASE | re.VERBOSE,
    ):
        return False
    return True


def _schaghticoke_ct_conflict(haystack: str) -> bool:
    """True when Schaghticoke is a Kent CT road, not the Town of Schaghticoke NY."""
    if not re.search(r'\bschaghticoke\b', haystack, flags=re.IGNORECASE):
        return False
    if not _SCHAGHTICOKE_CT.search(haystack):
        return False
    if re.search(
        r'schaghticoke\s*,?\s*(?:ny|n\.y\.|new\s+york)\b|town\s+of\s+schaghticoke',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _waterford_ct_conflict(haystack: str) -> bool:
    """True when Waterford refers to Connecticut, not Waterford NY."""
    if not re.search(r'\bwaterford\b', haystack, flags=re.IGNORECASE):
        return False
    if not _WATERFORD_CT.search(haystack):
        return False
    if re.search(
        r'waterford\s*,?\s*(?:ny|n\.y\.)\b|town\s+of\s+waterford',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _waterford_crystal_conflict(haystack: str) -> bool:
    """True when Waterford refers to the Irish crystal brand, not Waterford NY."""
    if not re.search(r'\bwaterford\b', haystack, flags=re.IGNORECASE):
        return False
    if not re.search(r'waterford\s+(?:crystal|wedgwood)\b', haystack, flags=re.IGNORECASE):
        return False
    if re.search(
        r'waterford\s*,?\s*(?:ny|n\.y\.)\b|town\s+of\s+waterford',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _rensselaer_indiana_conflict(haystack: str) -> bool:
    """True when Rensselaer refers to Indiana, not the city in NY."""
    if not re.search(r'\brensselaer\b', haystack, flags=re.IGNORECASE):
        return False
    if not re.search(
        r'rensselaer\s*,?\s*(?:in|indiana)\b|'
        r'rensselaer[\s\S]{0,80}\bindiana\b|'
        r'\bindiana\b[\s\S]{0,80}rensselaer',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(
        r'rensselaer\s*,?\s*(?:ny|n\.y\.|new\s+york)\b|'
        r'city\s+of\s+rensselaer|rensselaer\s+polytechnic|'
        r'rensselaer\s+county',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    return True


def _clifton_park_uk_conflict(haystack: str, author_handle: str | None = None) -> bool:
    """True when Clifton Park refers to York/Rotherham (England), not NY."""
    if not re.search(r'clifton\s+park', haystack, flags=re.IGNORECASE):
        return False
    if re.search(
        r'clifton\s+park\s*,?\s*(?:ny|n\.y\.|new\s+york)\b',
        haystack,
        flags=re.IGNORECASE,
    ):
        return False
    if _NY_CONTEXT.search(haystack):
        return False
    if _CLIFTON_PARK_UK.search(haystack):
        return True
    # Council accounts often omit "Rotherham" in Watersplash / park promo copy.
    handle = (author_handle or '').strip().lower()
    if re.search(r'rotherham', handle):
        return True
    return False


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
    # Kansas City's Egg and Art Garden must not unlock Albany's The Egg.
    if 'egg' in venue and _egg_kansas_city_conflict(haystack):
        return None
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
        # Albany County WY must not be rescued by the bare "Albany County" token.
        if _albany_county_wy_conflict(haystack, author_handle):
            return MatchResult(False, 'entity_other:albany_county_wy')
        if _STRONG_POSITIVE.search(haystack) and not _HARD_NEGATIVE_BLOCKS_STRONG.search(haystack):
            return MatchResult(True, 'strong_positive_over_negative')
        return MatchResult(False, 'hard_negative')

    if entity is not None and entity.region == 'capital_ny':
        if entity.entity_id == 'albany_county_ny' and _albany_county_wy_conflict(
            haystack, author_handle
        ):
            return MatchResult(False, 'entity_other:albany_county_wy')
        if entity.entity_id == 'watervliet_ny' and _watervliet_mi_conflict(haystack):
            return MatchResult(False, 'entity_other:watervliet_mi')
        if entity.entity_id == 'rensselaer_county_ny' and _rensselaer_roblox_conflict(haystack):
            return MatchResult(False, 'hard_negative:rensselaer_roblox')
        if entity.entity_id == 'schenectady_ny' and _schenectady_style_conflict(haystack):
            return MatchResult(False, 'hard_negative:schenectady_style')
        if entity.entity_id == 'schenectady_ny' and _schenectady_hashtag_spam_conflict(haystack):
            return MatchResult(False, 'hard_negative:schenectady_hashtag_spam')
        return MatchResult(True, f'entity_local:{entity.entity_id}')

    if _STRONG_POSITIVE.search(haystack):
        # Bare "Albany County" is also a strong token; still drop WY-tagged posts.
        if _albany_county_wy_conflict(haystack, author_handle):
            return MatchResult(False, 'entity_other:albany_county_wy')
        if _watervliet_mi_conflict(haystack):
            return MatchResult(False, 'entity_other:watervliet_mi')
        if _rensselaer_roblox_conflict(haystack):
            return MatchResult(False, 'hard_negative:rensselaer_roblox')
        if _schenectady_style_conflict(haystack):
            return MatchResult(False, 'hard_negative:schenectady_style')
        if _schenectady_hashtag_spam_conflict(haystack):
            return MatchResult(False, 'hard_negative:schenectady_hashtag_spam')
        if _canadian_capital_region_conflict(haystack, author_handle):
            return MatchResult(False, 'hard_negative:canadian_capital_region')
        if _md_dc_capital_region_conflict(haystack, author_handle):
            return MatchResult(False, 'hard_negative:md_dc_capital_region')
        if _louisiana_capital_region_conflict(haystack, author_handle):
            return MatchResult(False, 'hard_negative:louisiana_capital_region')
        if _pennsylvania_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:pennsylvania_capital_region')
        if _california_capital_region_conflict(haystack, author_handle):
            return MatchResult(False, 'hard_negative:california_capital_region')
        if _korea_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:korea_capital_region')
        if _ukraine_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:ukraine_capital_region')
        if _russia_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:russia_capital_region')
        if _sudan_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:sudan_capital_region')
        if _iceland_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:iceland_capital_region')
        if _france_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:france_capital_region')
        if _belgium_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:belgium_capital_region')
        if _india_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:india_capital_region')
        if _finland_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:finland_capital_region')
        if _denmark_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:denmark_capital_region')
        if _alberta_capital_region_conflict(haystack, author_handle):
            return MatchResult(False, 'hard_negative:alberta_capital_region')
        if _bulgaria_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:bulgaria_capital_region')
        if _japan_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:japan_capital_region')
        if _michigan_capital_region_conflict(haystack, author_handle):
            return MatchResult(False, 'hard_negative:michigan_capital_region')
        if _australia_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:australia_capital_region')
        if _georgia_atlanta_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:georgia_atlanta_capital_region')
        if _virginia_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:virginia_capital_region')
        if _virginia_capital_district_conflict(haystack):
            return MatchResult(False, 'hard_negative:virginia_capital_district')
        if _germany_capital_region_conflict(haystack):
            return MatchResult(False, 'hard_negative:germany_capital_region')
        if _colombia_capital_district_conflict(haystack):
            return MatchResult(False, 'hard_negative:colombia_capital_district')
        if _png_capital_district_conflict(haystack):
            return MatchResult(False, 'hard_negative:png_capital_district')
        if _japan_capital_district_conflict(haystack):
            return MatchResult(False, 'hard_negative:japan_capital_district')
        if _burnt_hills_descriptive_conflict(haystack):
            return MatchResult(False, 'hard_negative:burnt_hills_descriptive')
        if _loudonville_oh_conflict(haystack, author_handle):
            return MatchResult(False, 'hard_negative:loudonville_oh')
        if _clifton_park_uk_conflict(haystack, author_handle):
            return MatchResult(False, 'hard_negative:clifton_park_uk')
        if _newtonville_ma_conflict(haystack):
            return MatchResult(False, 'hard_negative:newtonville_ma')
        if _ct_capital_district_conflict(haystack):
            return MatchResult(False, 'hard_negative:ct_capital_district')
        if _schaghticoke_ct_conflict(haystack):
            return MatchResult(False, 'hard_negative:schaghticoke_ct')
        if _albany_wire_remote_conflict(haystack):
            return MatchResult(False, 'hard_negative:albany_wire_remote')
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
        # Montreal Banque Scotia / Osheaga must not unlock Village of Scotia NY.
        if _scotia_montreal_conflict(haystack) and distinct <= {'scotia'}:
            return MatchResult(False, 'hard_negative:scotia_montreal')
        # European Malta AIS / Rotterdam shipping must not unlock multi-local.
        if _malta_europe_conflict(haystack) and distinct <= {'malta', 'rotterdam'}:
            return MatchResult(False, 'hard_negative:malta_europe')
        # Disney's Saratoga Springs Resort must not unlock Cap Region Saratoga.
        if _disney_saratoga_conflict(haystack, author_handle) and distinct <= {
            'saratoga',
            'saratoga springs',
        }:
            return MatchResult(False, 'hard_negative:disney_saratoga')
        multi_eligible = {name for name in distinct if name not in _MULTI_LOCAL_EXCLUDED}
        # Collapse nested tokens ("saratoga" ⊂ "saratoga springs") so a hyphenated
        # URL path cannot unlock multi_local from a single place name.
        multi_eligible = {
            name
            for name in multi_eligible
            if not any(name != other and name in other for other in multi_eligible)
        }
        if len(multi_eligible) >= 2:
            if _wi_troy_waterford_conflict(haystack, multi_eligible):
                return MatchResult(False, 'hard_negative:wi_troy_waterford')
            if _indiana_albany_saratoga_conflict(haystack, multi_eligible):
                return MatchResult(False, 'hard_negative:indiana_albany_saratoga')
            if _malta_europe_conflict(haystack) and ({'malta', 'rotterdam'} & multi_eligible):
                return MatchResult(False, 'hard_negative:malta_europe')
            if _scotia_montreal_conflict(haystack) and 'scotia' in multi_eligible:
                return MatchResult(False, 'hard_negative:scotia_montreal')
            if _disney_saratoga_conflict(haystack, author_handle) and (
                {'saratoga', 'saratoga springs'} & multi_eligible
            ):
                return MatchResult(False, 'hard_negative:disney_saratoga')
            if _albany_bay_area_conflict(haystack) and (
                {'albany', 'saratoga', 'saratoga springs'} & multi_eligible
            ):
                return MatchResult(False, 'hard_negative:albany_bay_area')
            if _troy_michigan_conflict(haystack) and 'troy' in multi_eligible:
                return MatchResult(False, 'hard_negative:troy_michigan')
            if _troy_sc_conflict(haystack, author_handle) and 'troy' in multi_eligible:
                return MatchResult(False, 'hard_negative:troy_sc')
            if _troy_person_name_conflict(haystack) and 'troy' in multi_eligible:
                return MatchResult(False, 'hard_negative:troy_person_name')
            if _troy_pa_conflict(haystack) and 'troy' in multi_eligible:
                return MatchResult(False, 'hard_negative:troy_pa')
            if _troy_road_ithaca_conflict(haystack) and 'troy' in multi_eligible:
                return MatchResult(False, 'hard_negative:troy_road_ithaca')
            return MatchResult(True, 'multi_local_places')

        # Prefer a non-collision token when several ambiguous names appear but
        # multi-local did not fire (e.g. Saratoga + DelMar racing tags).
        term = sorted(distinct, key=lambda name: (name in _MULTI_LOCAL_EXCLUDED, name))[0]
        # Bare "albany" is the noisiest token; require NY/local context.
        if term == 'albany':
            if _albany_bay_area_conflict(haystack):
                return MatchResult(False, 'hard_negative:albany_bay_area')
            if _gta_albany_conflict(haystack):
                return MatchResult(False, 'hard_negative:gta_albany')
            if _albany_ithaca_contrast_conflict(haystack):
                return MatchResult(False, 'hard_negative:albany_ithaca_contrast')
            if _albany_wire_remote_conflict(haystack):
                return MatchResult(False, 'hard_negative:albany_wire_remote')
            if _albany_bandscan_conflict(haystack):
                return MatchResult(False, 'hard_negative:albany_bandscan')
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

        if term == 'scotia' and _scotia_montreal_conflict(haystack):
            return MatchResult(False, 'hard_negative:scotia_montreal')

        if term == 'bethlehem' and _bethlehem_pa_conflict(haystack):
            return MatchResult(False, 'hard_negative:bethlehem_pa')

        if term == 'brunswick' and _brunswick_records_conflict(haystack):
            return MatchResult(False, 'hard_negative:brunswick_records')

        if term == 'brunswick' and _brunswick_ok_conflict(haystack):
            return MatchResult(False, 'hard_negative:brunswick_ok')

        if term == 'brunswick' and _brunswick_schools_other_conflict(haystack):
            return MatchResult(False, 'hard_negative:brunswick_schools_other')

        if term == 'green island' and _green_island_other_conflict(haystack):
            return MatchResult(False, 'hard_negative:green_island_other')

        if term == 'waterford' and _waterford_ct_conflict(haystack):
            return MatchResult(False, 'hard_negative:waterford_ct')

        if term == 'waterford' and _waterford_crystal_conflict(haystack):
            return MatchResult(False, 'hard_negative:waterford_crystal')

        if term == 'rensselaer' and _rensselaer_indiana_conflict(haystack):
            return MatchResult(False, 'hard_negative:rensselaer_indiana')

        if term in {'malta', 'rotterdam'} and _malta_europe_conflict(haystack):
            return MatchResult(False, 'hard_negative:malta_europe')

        if term in {'saratoga', 'saratoga springs'} and _disney_saratoga_conflict(
            haystack, author_handle
        ):
            return MatchResult(False, 'hard_negative:disney_saratoga')

        if term in {'saratoga', 'saratoga springs'} and _saratoga_park_ca_conflict(haystack):
            return MatchResult(False, 'hard_negative:saratoga_park_ca')

        if term in {'saratoga', 'saratoga springs'} and _albany_bay_area_conflict(haystack):
            return MatchResult(False, 'hard_negative:albany_bay_area')

        if term == 'stillwater' and _stillwater_film_conflict(haystack):
            return MatchResult(False, 'hard_negative:stillwater_film')

        if term == 'stillwater' and _stillwater_road_conflict(haystack, author_handle):
            return MatchResult(False, 'hard_negative:stillwater_road')

        if term == 'troy' and _troy_michigan_conflict(haystack):
            return MatchResult(False, 'hard_negative:troy_michigan')

        if term == 'troy' and _troy_sc_conflict(haystack, author_handle):
            return MatchResult(False, 'hard_negative:troy_sc')

        if term == 'troy' and _troy_person_name_conflict(haystack):
            return MatchResult(False, 'hard_negative:troy_person_name')

        if term == 'troy' and _troy_pa_conflict(haystack):
            return MatchResult(False, 'hard_negative:troy_pa')

        if term == 'troy' and _troy_road_ithaca_conflict(haystack):
            return MatchResult(False, 'hard_negative:troy_road_ithaca')

        if _NY_CONTEXT.search(place_haystack) or _STRONG_POSITIVE.search(haystack):
            return MatchResult(True, f'ambiguous_with_context:{term}')
        prior = _soft_prior_ambiguous(author_did, soft_prior_dids, term)
        if prior:
            return prior
        clf = _classifier_keep(haystack, term=term, model=classifier_model)
        return clf if clf else MatchResult(False, f'ambiguous_no_context:{term}')

    clf = _classifier_keep(haystack, term=None, model=classifier_model)
    return clf if clf else MatchResult(False, 'no_match')
