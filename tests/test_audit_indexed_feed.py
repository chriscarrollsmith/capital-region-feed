"""Unit tests for indexed-feed audit helpers (no network)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from scripts.audit_indexed_feed import (
    AuditRow,
    evaluate_post,
    load_resident_dids_from_db,
    load_soft_prior_dids_from_db,
    load_uris_from_db,
    post_alt_text,
    purge_uris,
)


def test_post_alt_text_merges_record_and_view_embeds() -> None:
    post = {
        'record': {
            'text': 'hello',
            'embed': {
                'external': {
                    'title': 'Record title',
                    'description': 'Record desc',
                }
            },
        },
        'embed': {
            'external': {
                'title': 'View title',
                'description': 'View desc',
            }
        },
    }
    alt = post_alt_text(post)
    assert 'Record title' in alt
    assert 'View title' in alt


def test_evaluate_post_drops_replies_when_configured() -> None:
    post = {
        'uri': 'at://did:plc:x/app.bsky.feed.post/1',
        'author': {'did': 'did:plc:x', 'handle': 'x.bsky.social'},
        'record': {
            'text': 'Albany NY tonight',
            'reply': {'parent': {'uri': 'at://did:plc:x/app.bsky.feed.post/0'}},
        },
    }
    matched, reason = evaluate_post(
        post,
        allowlist_dids=set(),
        allowlist_handles=set(),
        soft_prior_dids=set(),
        ignore_replies=True,
    )
    assert matched is False
    assert reason == 'reply_ignored'


def test_evaluate_post_keeps_strong_local() -> None:
    post = {
        'uri': 'at://did:plc:x/app.bsky.feed.post/1',
        'author': {'did': 'did:plc:x', 'handle': 'x.bsky.social'},
        'record': {'text': 'Concert at the Empire State Plaza tonight'},
    }
    matched, reason = evaluate_post(
        post,
        allowlist_dids=set(),
        allowlist_handles=set(),
        soft_prior_dids=set(),
        ignore_replies=True,
        blocklist_dids=set(),
        blocklist_handles=set(),
    )
    assert matched is True
    assert reason.startswith('entity_local:') or reason == 'strong_positive'


def test_evaluate_post_drops_blocklisted_author() -> None:
    post = {
        'uri': 'at://did:plc:blocked/app.bsky.feed.post/1',
        'author': {'did': 'did:plc:blocked', 'handle': 'blocked.bsky.social'},
        'record': {'text': 'Concert at the Empire State Plaza tonight'},
    }
    matched, reason = evaluate_post(
        post,
        allowlist_dids=set(),
        allowlist_handles=set(),
        soft_prior_dids=set(),
        ignore_replies=True,
        blocklist_dids={'did:plc:blocked'},
        blocklist_handles=set(),
    )
    assert matched is False
    assert reason == 'blocklisted'


def test_evaluate_post_drops_acab_mute() -> None:
    post = {
        'uri': 'at://did:plc:x/app.bsky.feed.post/1',
        'author': {'did': 'did:plc:x', 'handle': 'x.bsky.social'},
        'record': {'text': 'ACAB at the Empire State Plaza tonight'},
    }
    matched, reason = evaluate_post(
        post,
        allowlist_dids=set(),
        allowlist_handles=set(),
        soft_prior_dids=set(),
        ignore_replies=True,
        blocklist_dids=set(),
        blocklist_handles=set(),
    )
    assert matched is False
    assert reason == 'muted'


def test_load_uris_and_purge(tmp_path: Path) -> None:
    db_path = tmp_path / 'feed.db'
    con = sqlite3.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE post (
            uri TEXT, cid TEXT, author_did TEXT, match_reason TEXT, indexed_at TEXT
        )
        """
    )
    con.executemany(
        'INSERT INTO post VALUES (?, ?, ?, ?, ?)',
        [
            ('at://a', 'c1', 'did:a', 'strong_positive', '2026-07-26 01:00:00'),
            ('at://b', 'c2', 'did:b', 'ambiguous_with_context:troy', '2026-07-26 00:00:00'),
        ],
    )
    con.commit()
    con.close()

    rows = load_uris_from_db(db_path)
    assert [uri for uri, _ in rows] == ['at://a', 'at://b']
    assert purge_uris(db_path, ['at://b']) == 1
    assert load_uris_from_db(db_path) == [('at://a', 'strong_positive')]


def test_load_soft_prior_dids_respects_threshold(tmp_path: Path) -> None:
    db_path = tmp_path / 'feed.db'
    con = sqlite3.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE authordecision (
            id INTEGER PRIMARY KEY,
            author_did TEXT,
            kept INTEGER,
            decided_at TEXT
        )
        """
    )
    now = datetime(2026, 7, 26, 12, 0, 0)
    stamp = now.isoformat(sep=' ')
    rows = [('did:plc:enough', 1, stamp) for _ in range(3)]
    rows.append(('did:plc:low', 1, stamp))
    rows.extend(('did:plc:rejected', 1, stamp) for _ in range(3))
    rows.append(('did:plc:rejected', 0, stamp))
    con.executemany(
        'INSERT INTO authordecision (author_did, kept, decided_at) VALUES (?, ?, ?)', rows
    )
    con.commit()
    con.close()

    dids = load_soft_prior_dids_from_db(
        db_path,
        min_keeps=3,
        window_days=30,
        now=now.replace(tzinfo=UTC).replace(tzinfo=None),
    )
    assert dids == {'did:plc:enough'}


def test_resident_lane_does_not_revoke_soft_prior(tmp_path: Path) -> None:
    db_path = tmp_path / 'feed.db'
    con = sqlite3.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE authordecision (
            id INTEGER PRIMARY KEY,
            author_did TEXT,
            kept INTEGER,
            decided_at TEXT,
            lane TEXT
        )
        """
    )
    now = datetime(2026, 7, 26, 12, 0, 0)
    stamp = now.isoformat(sep=' ')
    rows = [('did:plc:local', 1, stamp, 'mention') for _ in range(3)]
    rows.append(('did:plc:local', 0, stamp, 'resident'))
    con.executemany(
        'INSERT INTO authordecision (author_did, kept, decided_at, lane) VALUES (?, ?, ?, ?)',
        rows,
    )
    con.commit()
    con.close()
    moment = now.replace(tzinfo=UTC).replace(tzinfo=None)
    assert load_soft_prior_dids_from_db(db_path, min_keeps=3, window_days=30, now=moment) == {
        'did:plc:local'
    }
    assert load_resident_dids_from_db(
        db_path,
        min_keeps=3,
        demote_rejects=3,
        max_per_day=8,
        window_days=30,
        now=moment,
    ) == {'did:plc:local'}


def test_resident_demote_and_daily_cap(tmp_path: Path) -> None:
    db_path = tmp_path / 'feed.db'
    con = sqlite3.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE authordecision (
            id INTEGER PRIMARY KEY,
            author_did TEXT,
            kept INTEGER,
            decided_at TEXT,
            lane TEXT
        )
        """
    )
    now = datetime(2026, 7, 26, 12, 0, 0)
    stamp = now.isoformat(sep=' ')
    rows: list[tuple[str, int, str, str]] = [
        ('did:plc:demoted', 1, stamp, 'mention'),
        ('did:plc:demoted', 1, stamp, 'mention'),
        ('did:plc:demoted', 1, stamp, 'mention'),
        ('did:plc:demoted', 0, stamp, 'resident'),
        ('did:plc:demoted', 0, stamp, 'resident'),
        ('did:plc:demoted', 0, stamp, 'resident'),
        ('did:plc:capped', 1, stamp, 'mention'),
        ('did:plc:capped', 1, stamp, 'mention'),
        ('did:plc:capped', 1, stamp, 'mention'),
    ]
    rows.extend(('did:plc:capped', 1, stamp, 'resident') for _ in range(8))
    con.executemany(
        'INSERT INTO authordecision (author_did, kept, decided_at, lane) VALUES (?, ?, ?, ?)',
        rows,
    )
    con.commit()
    con.close()
    moment = now.replace(tzinfo=UTC).replace(tzinfo=None)
    assert (
        load_resident_dids_from_db(
            db_path,
            min_keeps=3,
            demote_rejects=3,
            max_per_day=8,
            window_days=30,
            now=moment,
        )
        == set()
    )


def test_evaluate_post_keeps_resident_no_placename_and_drops_hard_negative() -> None:
    did = 'did:plc:residentaudit0000000001'
    photo = {
        'uri': 'at://did:plc:residentaudit0000000001/app.bsky.feed.post/1',
        'author': {'did': did, 'handle': 'neighbor.bsky.social'},
        'record': {'text': 'The light on the way home was unreal.'},
    }
    matched, reason = evaluate_post(
        photo,
        allowlist_dids=set(),
        allowlist_handles=set(),
        soft_prior_dids=set(),
        resident_dids={did},
        ignore_replies=True,
        blocklist_dids=set(),
        blocklist_handles=set(),
    )
    assert matched is True
    assert reason == 'resident'
    closed, closed_reason = evaluate_post(
        photo,
        allowlist_dids=set(),
        allowlist_handles=set(),
        soft_prior_dids=set(),
        ignore_replies=True,
        blocklist_dids=set(),
        blocklist_handles=set(),
    )
    assert closed is False
    assert closed_reason == 'no_match'

    park = {
        'uri': 'at://did:plc:residentaudit0000000001/app.bsky.feed.post/2',
        'author': {'did': did, 'handle': 'neighbor.bsky.social'},
        'record': {'text': 'Nice afternoon walking around Albany Park.'},
    }
    dropped, drop_reason = evaluate_post(
        park,
        allowlist_dids=set(),
        allowlist_handles=set(),
        soft_prior_dids=set(),
        resident_dids={did},
        ignore_replies=True,
        blocklist_dids=set(),
        blocklist_handles=set(),
    )
    assert dropped is False
    assert drop_reason in {'hard_negative', 'entity_other:albany_park_chicago'}


def test_load_soft_prior_missing_table_is_empty(tmp_path: Path) -> None:
    db_path = tmp_path / 'feed.db'
    con = sqlite3.connect(str(db_path))
    con.execute('CREATE TABLE post (uri TEXT)')
    con.commit()
    con.close()
    assert load_soft_prior_dids_from_db(db_path) == set()


def test_audit_row_dataclass_fields() -> None:
    row = AuditRow(
        uri='at://x',
        author_did=None,
        author_handle=None,
        text_preview='',
        indexed_reason=None,
        matched=False,
        reason='not_found',
        status='not_found',
    )
    assert row.reason == 'not_found'
