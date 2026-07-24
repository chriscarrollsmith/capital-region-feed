import json
from pathlib import Path

import pytest

from server.matcher import extract_alt_text, match_post

CASES = json.loads(
    (Path(__file__).resolve().parents[1] / 'data' / 'eval_cases.json').read_text(
        encoding='utf-8'
    )
)

ALLOWLIST_HANDLES = {
    'news10.bsky.social',
    'timesunion.com',
    'albany-ny.bsky.social',
    'cbs6albany.bsky.social',
    'wrgb.bsky.social',
}


@pytest.mark.parametrize('case', CASES, ids=[c['id'] for c in CASES])
def test_eval_case(case):
    result = match_post(
        case.get('text', ''),
        alt_text=case.get('alt_text', ''),
        author_did=case.get('author_did'),
        author_handle=case.get('author_handle'),
        allowlist_handles=ALLOWLIST_HANDLES,
    )
    assert result.matched is bool(case['expected']), (
        f"{case['id']}: expected={case['expected']} got={result.matched} "
        f"reason={result.reason} note={case.get('note')}"
    )


def test_extract_alt_text_from_images():
    embed = {
        '$type': 'app.bsky.embed.images',
        'images': [{'alt': 'Sunset over Albany, NY', 'image': {}}],
    }
    assert 'Albany, NY' in extract_alt_text(embed)
