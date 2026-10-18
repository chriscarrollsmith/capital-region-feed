"""Unit tests for decider-earned soft author priors."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from server.author_priors import author_has_soft_prior, record_decision
from server.database import AuthorDecision, AuthorLocalStats, Post, SubscriptionState, db, utc_now

_TABLES = [Post, SubscriptionState, AuthorLocalStats, AuthorDecision]


@pytest.fixture()
def isolated_db() -> Any:
    db.connect(reuse_if_open=True)
    db.drop_tables(_TABLES, safe=True)
    db.create_tables(_TABLES)
    yield
    AuthorDecision.delete().execute()
    AuthorLocalStats.delete().execute()
    Post.delete().execute()


def test_keeps_earn_prior_and_reject_revokes(isolated_db: None) -> None:
    from server import config

    did = 'did:plc:priorearnertest00000000001'
    assert not author_has_soft_prior(did)
    assert not author_has_soft_prior(None)
    record_decision(None, kept=True)
    assert AuthorDecision.select().count() == 0

    for _ in range(config.SOFT_PRIOR_MIN_KEEPS - 1):
        record_decision(did, kept=True)
        assert not author_has_soft_prior(did)
    record_decision(did, kept=True)
    assert author_has_soft_prior(did)

    record_decision(did, kept=False)
    assert not author_has_soft_prior(did)


def test_reject_outside_window_does_not_block(isolated_db: None) -> None:
    from server import config

    did = 'did:plc:prioroldreject00000000001'
    AuthorDecision.create(
        author_did=did,
        kept=False,
        decided_at=utc_now() - timedelta(days=config.SOFT_PRIOR_WINDOW_DAYS + 1),
    )
    for _ in range(config.SOFT_PRIOR_MIN_KEEPS):
        record_decision(did, kept=True)
    assert author_has_soft_prior(did)


def test_keeps_outside_window_do_not_count(isolated_db: None) -> None:
    from server import config

    did = 'did:plc:priorexpiredtest0000000001'
    old = utc_now() - timedelta(days=config.SOFT_PRIOR_WINDOW_DAYS + 1)
    for _ in range(config.SOFT_PRIOR_MIN_KEEPS):
        AuthorDecision.create(author_did=did, kept=True, decided_at=old)
    assert not author_has_soft_prior(did)


def test_prune_drops_decisions_outside_window(isolated_db: None) -> None:
    from server import config
    from server.database import prune_old_posts

    did = 'did:plc:prunedecision000000000001'
    AuthorDecision.create(
        author_did=did,
        kept=True,
        decided_at=utc_now() - timedelta(days=config.SOFT_PRIOR_WINDOW_DAYS + 1),
    )
    AuthorDecision.create(author_did=did, kept=True, decided_at=utc_now())
    prune_old_posts()
    remaining = list(AuthorDecision.select())
    assert len(remaining) == 1
    assert remaining[0].kept is True
