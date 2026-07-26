"""Feed skeleton ranking mode tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
from server.database import AuthorLocalStats, Post, SubscriptionState, db


@pytest.fixture()
def isolated_db() -> Any:
    db.connect(reuse_if_open=True)
    db.drop_tables([Post, SubscriptionState, AuthorLocalStats], safe=True)
    db.create_tables([Post, SubscriptionState, AuthorLocalStats])
    yield
    AuthorLocalStats.delete().execute()
    Post.delete().execute()


def _insert(
    *,
    key: str,
    indexed_at: datetime,
    created_at: datetime | None = None,
    like_count: int = 0,
    repost_count: int = 0,
) -> None:
    Post.create(
        uri=f'at://did:plc:test/app.bsky.feed.post/{key}',
        cid=f'cid{key}',
        author_did='did:plc:test',
        created_at=created_at,
        indexed_at=indexed_at,
        like_count=like_count,
        repost_count=repost_count,
        match_reason='strong_positive',
    )


def test_handler_engagement_orders_by_score(isolated_db: None) -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    _insert(key='low', indexed_at=now, like_count=1, repost_count=0)
    _insert(key='high', indexed_at=now - timedelta(hours=1), like_count=2, repost_count=2)

    with patch('server.algos.feed.config.RANKING_MODE', 'engagement'):
        from server.algos.feed import handler

        body = handler(None, 10)

    uris = [item['post'] for item in body['feed']]
    assert uris[0].endswith('/high')
    assert uris[1].endswith('/low')
    assert '::' in body['cursor']
    assert body['cursor'].count('::') == 2


def test_handler_created_prefers_author_time(isolated_db: None) -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    _insert(
        key='older-created',
        indexed_at=now,
        created_at=now - timedelta(days=2),
    )
    _insert(
        key='newer-created',
        indexed_at=now - timedelta(hours=3),
        created_at=now - timedelta(hours=1),
    )

    with patch('server.algos.feed.config.RANKING_MODE', 'created'):
        from server.algos.feed import handler

        body = handler(None, 10)

    uris = [item['post'] for item in body['feed']]
    assert uris[0].endswith('/newer-created')
    assert uris[1].endswith('/older-created')


def test_handle_engagement_increments_counts(isolated_db: None) -> None:
    from server.indexer import handle_event

    now = datetime.now(UTC).replace(tzinfo=None)
    uri = 'at://did:plc:test/app.bsky.feed.post/eng1'
    Post.create(
        uri=uri,
        cid='cideng1',
        author_did='did:plc:test',
        indexed_at=now,
        like_count=0,
        repost_count=0,
    )
    handle_event(
        {
            'operation': 'create',
            'engagement': 'like',
            'subject_uri': uri,
        }
    )
    handle_event(
        {
            'operation': 'create',
            'engagement': 'repost',
            'subject_uri': uri,
        }
    )
    row = Post.get(Post.uri == uri)
    assert row.like_count == 1
    assert row.repost_count == 1
