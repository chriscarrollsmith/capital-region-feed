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


def test_load_gazetteer_rejects_unknown_region(tmp_path: Path) -> None:
    path = tmp_path / 'bad.json'
    path.write_text(
        '{"entities":[{"id":"x","region":"mars","surfaces":["x"]}]}',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='unknown region'):
        load_gazetteer(path)
