"""Unit tests for allowlist file loading and DID file generation helpers."""

from __future__ import annotations

from pathlib import Path

from scripts.resolve_allowlist_dids import build_did_file
from server.allowlists import (
    load_allowlist_dids,
    load_allowlist_handles,
    load_blocklist_dids,
    load_blocklist_handles,
    load_list_file,
)


def test_load_list_file_skips_comments_and_blanks(tmp_path: Path) -> None:
    path = tmp_path / 'list.txt'
    path.write_text('# comment\n\nalpha\nBeta\n', encoding='utf-8')
    assert load_list_file(path) == ['alpha', 'Beta']


def test_production_allowlists_are_nonempty_and_aligned() -> None:
    handles = load_allowlist_handles()
    dids = load_allowlist_dids()
    assert handles
    assert dids
    assert all(h == h.lower() for h in handles)
    assert all(d.startswith('did:') for d in dids)
    # Every checked-in DID should correspond to a curated handle (file comments).
    # Count equality keeps the resolve script honest after handle edits.
    assert len(dids) == len(handles)


def test_production_blocklists_are_aligned() -> None:
    handles = load_blocklist_handles()
    dids = load_blocklist_dids()
    assert all(h == h.lower() for h in handles)
    assert all(d.startswith('did:') for d in dids)
    assert len(dids) == len(handles)
    assert 'bypophoenix.bsky.social' in handles
    assert 'did:plc:vdykwvsuhnim6beywhcqje7r' in dids


def test_build_did_file_includes_handle_provenance() -> None:
    text = build_did_file(
        [
            ('timesunion.com', 'did:plc:exampletimesunion'),
            ('news10.bsky.social', 'did:plc:examplenews10'),
        ]
    )
    assert '# timesunion.com' in text
    assert 'did:plc:exampletimesunion' in text
    assert 'resolve_allowlist_dids.py' in text
