import json
from pathlib import Path
from typing import Any

import pytest
from server.matcher import extract_alt_text, match_post

ALL_CASES = json.loads(
    (Path(__file__).resolve().parents[1] / 'data' / 'eval_cases.json').read_text(encoding='utf-8')
)
# Gap cases (regression=false) are measured by scripts/eval_filter.py but not
# asserted here — they document known recall misses for later backlog items.
CASES = [c for c in ALL_CASES if c.get('regression', True)]

ALLOWLIST_HANDLES = {
    'news10.bsky.social',
    'timesunion.com',
    'albany-ny.bsky.social',
    'cbs6albany.bsky.social',
    'wrgb.bsky.social',
}


@pytest.mark.parametrize('case', CASES, ids=[c['id'] for c in CASES])
def test_eval_case(case: dict[str, Any]) -> None:
    result = match_post(
        case.get('text', ''),
        alt_text=case.get('alt_text', ''),
        author_did=case.get('author_did'),
        author_handle=case.get('author_handle'),
        allowlist_handles=ALLOWLIST_HANDLES,
    )
    assert result.matched is bool(case['expected']), (
        f'{case["id"]}: expected={case["expected"]} got={result.matched} '
        f'reason={result.reason} note={case.get("note")}'
    )


def test_extract_alt_text_from_images() -> None:
    embed = {
        '$type': 'app.bsky.embed.images',
        'images': [{'alt': 'Sunset over [REDACTED]', 'image': {}}],
    }
    assert '[REDACTED]' in extract_alt_text(embed)
