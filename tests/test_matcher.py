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
    result = match_post(
        case.get('text', ''),
        alt_text=case.get('alt_text', ''),
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
    assert blocked.reason == 'hard_negative'


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


def test_extract_alt_text_from_images() -> None:
    embed = {
        '$type': 'app.bsky.embed.images',
        'images': [{'alt': 'Sunset over [REDACTED]', 'image': {}}],
    }
    assert '[REDACTED]' in extract_alt_text(embed)
