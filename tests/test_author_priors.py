"""Unit tests for decider-earned soft author priors."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from server.author_priors import (
    LANE_RESIDENT,
    author_has_soft_prior,
    author_is_resident,
    lane_for_reason,
    record_decision,
)
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


def test_resident_lane_reject_does_not_revoke_soft_prior(isolated_db: None) -> None:
    from server import config

    did = 'did:plc:residentlanereject00000001'
    for _ in range(config.SOFT_PRIOR_MIN_KEEPS):
        record_decision(did, kept=True)
    record_decision(did, kept=False, lane=LANE_RESIDENT)
    assert author_has_soft_prior(did)
    assert author_is_resident(did)


def test_resident_requires_mention_keeps_and_demotes_on_reject_rate(isolated_db: None) -> None:
    from server import config

    did = 'did:plc:residentdemote000000000001'
    assert not author_is_resident(did)
    for _ in range(config.RESIDENT_MIN_KEEPS - 1):
        record_decision(did, kept=True)
        assert not author_is_resident(did)
    record_decision(did, kept=True)
    assert author_is_resident(did)

    for _ in range(config.RESIDENT_DEMOTE_REJECTS):
        record_decision(did, kept=False, lane=LANE_RESIDENT)
    assert not author_is_resident(did)
    assert author_has_soft_prior(did)


def test_resident_keeps_offset_demotion(isolated_db: None) -> None:
    from server import config

    did = 'did:plc:residentkeeps000000000001'
    for _ in range(config.RESIDENT_MIN_KEEPS):
        record_decision(did, kept=True)
    # Outside the one-day cap, still inside the privilege window.
    when = utc_now() - timedelta(days=2)
    for _ in range(config.RESIDENT_DEMOTE_REJECTS):
        AuthorDecision.create(author_did=did, kept=True, decided_at=when, lane=LANE_RESIDENT)
        AuthorDecision.create(author_did=did, kept=False, decided_at=when, lane=LANE_RESIDENT)
    assert author_is_resident(did)


def test_resident_daily_cap_suspends_privilege(isolated_db: None) -> None:
    from server import config

    did = 'did:plc:residentcap0000000000001'
    for _ in range(config.RESIDENT_MIN_KEEPS):
        record_decision(did, kept=True)
    for _ in range(config.RESIDENT_MAX_PER_DAY):
        record_decision(did, kept=True, lane=LANE_RESIDENT)
    assert not author_is_resident(did)
    assert author_has_soft_prior(did)


def test_mention_reject_revokes_residency(isolated_db: None) -> None:
    from server import config

    did = 'did:plc:residentmentionreject00001'
    for _ in range(config.RESIDENT_MIN_KEEPS):
        record_decision(did, kept=True)
    record_decision(did, kept=False)
    assert not author_is_resident(did)
    assert not author_has_soft_prior(did)


def test_lane_for_reason_splits_resident_from_mention() -> None:
    assert lane_for_reason('decision_keep:resident:0.880') == LANE_RESIDENT
    assert lane_for_reason('decision_reject:resident:0.200') == LANE_RESIDENT
    assert lane_for_reason('decision_keep:strong_positive:0.880') == 'mention'
    assert lane_for_reason('decision_keep:soft_prior_ambiguous:troy:0.880') == 'mention'


def test_lane_column_backfill_counts_as_mention(isolated_db: None) -> None:
    from server import config
    from server.database import _ensure_columns, db

    did = 'did:plc:olddecisionlane000000001'
    db.execute_sql('DROP TABLE authordecision')
    db.execute_sql(
        """
        CREATE TABLE authordecision (
            id INTEGER PRIMARY KEY,
            author_did TEXT,
            kept INTEGER,
            decided_at DATETIME
        )
        """
    )
    stamp = utc_now()
    for _ in range(config.RESIDENT_MIN_KEEPS):
        db.execute_sql(
            'INSERT INTO authordecision (author_did, kept, decided_at) VALUES (?, 1, ?)',
            (did, stamp),
        )
    _ensure_columns()
    assert author_has_soft_prior(did)
    assert author_is_resident(did)


def test_unknown_lane_is_rejected(isolated_db: None) -> None:
    with pytest.raises(ValueError):
        record_decision('did:plc:badlane0000000000000001', kept=True, lane='other')


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
