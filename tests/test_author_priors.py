"""Unit tests for soft author prior counters."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from server.author_priors import author_has_soft_prior, is_strong_match_reason, record_strong_match
from server.database import AuthorLocalStats, Post, SubscriptionState, db, utc_now


@pytest.fixture()
def isolated_db() -> Any:
    db.connect(reuse_if_open=True)
    db.drop_tables([Post, SubscriptionState, AuthorLocalStats], safe=True)
    db.create_tables([Post, SubscriptionState, AuthorLocalStats])
    yield
    AuthorLocalStats.delete().execute()
    Post.delete().execute()


def test_is_strong_match_reason() -> None:
    assert is_strong_match_reason('strong_positive')
    assert is_strong_match_reason('ambiguous_with_context:troy')
    assert is_strong_match_reason('event_local_venue:proctors')
    assert is_strong_match_reason('entity_local:schenectady_ny')
    assert not is_strong_match_reason('allowlist_did')
    assert not is_strong_match_reason('entity_other:albany_park_chicago')
    assert not is_strong_match_reason('soft_prior_ambiguous:troy')
    assert not is_strong_match_reason('classifier:local_micro')
    assert not is_strong_match_reason('bare_albany')


def test_record_strong_match_earns_soft_prior(isolated_db: None) -> None:
    from server import config

    did = 'did:plc:priorearnertest00000000001'
    assert not author_has_soft_prior(did)
    for _ in range(config.SOFT_PRIOR_MIN_STRONG - 1):
        record_strong_match(did)
        assert not author_has_soft_prior(did)
    record_strong_match(did)
    assert author_has_soft_prior(did)


def test_soft_prior_expires_outside_window(isolated_db: None) -> None:
    from server import config

    did = 'did:plc:priorexpiredtest0000000001'
    for _ in range(config.SOFT_PRIOR_MIN_STRONG):
        record_strong_match(did)
    assert author_has_soft_prior(did)

    row = AuthorLocalStats.get_by_id(did)
    row.last_strong_at = utc_now() - timedelta(days=config.SOFT_PRIOR_WINDOW_DAYS + 1)
    row.save()
    assert not author_has_soft_prior(did)


def test_stale_window_resets_count(isolated_db: None) -> None:
    from server import config

    did = 'did:plc:priorresettest000000000001'
    for _ in range(config.SOFT_PRIOR_MIN_STRONG):
        record_strong_match(did)
    row = AuthorLocalStats.get_by_id(did)
    row.last_strong_at = utc_now() - timedelta(days=config.SOFT_PRIOR_WINDOW_DAYS + 1)
    row.save()

    record_strong_match(did)
    row = AuthorLocalStats.get_by_id(did)
    assert row.strong_match_count == 1
    assert not author_has_soft_prior(did)
