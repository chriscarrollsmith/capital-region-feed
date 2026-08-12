"""Unit tests for author blocklist and content mute helpers."""

from __future__ import annotations

from server.content_filters import author_is_blocked, text_is_muted


def test_author_is_blocked_by_did_or_handle() -> None:
    dids = {'did:plc:blocked000000000000000001'}
    handles = {'bad.example'}
    assert author_is_blocked(
        'did:plc:blocked000000000000000001',
        blocklist_dids=dids,
        blocklist_handles=handles,
    )
    assert author_is_blocked(
        'did:plc:other',
        'Bad.Example',
        blocklist_dids=dids,
        blocklist_handles=handles,
    )
    assert not author_is_blocked(
        'did:plc:other',
        'ok.example',
        blocklist_dids=dids,
        blocklist_handles=handles,
    )


def test_text_is_muted_matches_acab_assertions() -> None:
    assert text_is_muted('ACAB — no justice in this town.')
    assert text_is_muted('Chanting #acab downtown tonight.')
    assert text_is_muted('Remember: A.C.A.B.')
    assert text_is_muted('All cops are bastards, full stop.')
    assert not text_is_muted('A macabre night at the Troy cinema.')
    assert not text_is_muted('The project is about to finish.')


def test_text_is_muted_honors_keyword_substrings() -> None:
    assert text_is_muted('Hello world', keywords=('world',))
    assert not text_is_muted('Hello world', keywords=('xyz',))
