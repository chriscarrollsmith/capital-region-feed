"""Gazetteer entity disambiguation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from server.gazetteer import default_gazetteer, load_gazetteer
from server.matcher import match_post


def test_default_gazetteer_other_beats_local_substring() -> None:
    gaz = default_gazetteer()
    hit = gaz.lookup('Reporting from the National Capital Region this morning.')
    assert hit is not None
    assert hit.region == 'other'
    assert hit.entity_id == 'national_capital_region'


def test_match_post_entity_other_and_local() -> None:
    other = match_post('Street festival vibes in Albany Park this weekend.')
    assert other.matched is False
    assert other.reason == 'entity_other:albany_park_chicago'

    local = match_post('Niskayuna library board meets Thursday.')
    assert local.matched is True
    assert local.reason == 'entity_local:niskayuna_ny'


def test_albany_county_wyoming_not_capital_region() -> None:
    contiguous = match_post('Road work begins in Albany County, WY next month.')
    assert contiguous.matched is False
    assert contiguous.reason == 'entity_other:albany_county_wy'

    hashtags = match_post(
        'Albany County is backing a major road project with Wyo Silver. '
        '#AlbanyCounty #WY #InfrastructureImprovement'
    )
    assert hashtags.matched is False
    assert hashtags.reason == 'entity_other:albany_county_wy'

    nws_bracket = match_post(
        'CYS issues A THUNDERSTORM WILL IMPACT SOUTHEASTERN ALBANY COUNTY '
        'THROUGH 830 AM MDT for Laramie Valley, South Laramie Range [WY]'
    )
    assert nws_bracket.matched is False
    assert nws_bracket.reason == 'entity_other:albany_county_wy'

    ny = match_post('Albany County executives meet downtown tomorrow. #AlbanyNY')
    assert ny.matched is True


def test_load_gazetteer_rejects_unknown_region(tmp_path: Path) -> None:
    path = tmp_path / 'bad.json'
    path.write_text(
        '{"entities":[{"id":"x","region":"mars","surfaces":["x"]}]}',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='unknown region'):
        load_gazetteer(path)
