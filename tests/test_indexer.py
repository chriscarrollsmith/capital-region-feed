"""Indexer path tests: allowlisted DIDs index without placename text."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from server.database import Post, SubscriptionState, db


@pytest.fixture()
def isolated_db() -> Any:
    """Rebuild post tables so each test starts empty."""
    db.connect(reuse_if_open=True)
    db.drop_tables([Post, SubscriptionState], safe=True)
    db.create_tables([Post, SubscriptionState])
    yield
    Post.delete().execute()


def _create_event(*, author: str, text: str, uri_key: str = 'abc') -> dict[str, Any]:
    now = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
    return {
        'operation': 'create',
        'uri': f'at://{author}/app.bsky.feed.post/{uri_key}',
        'cid': 'bafytestcid',
        'author': author,
        'record': {
            '$type': 'app.bsky.feed.post',
            'text': text,
            'createdAt': now,
        },
    }


def test_handle_event_indexes_allowlisted_did_without_placename(isolated_db: None) -> None:
    from server import config
    from server.indexer import handle_event

    assert config.ALLOWLIST_DIDS, 'production DID allowlist must be populated'
    author = sorted(config.ALLOWLIST_DIDS)[0]
    event = _create_event(author=author, text='Our newsletter is out for subscribers.')
    handle_event(event)

    row = Post.get(Post.uri == event['uri'])
    assert row.author_did == author
    assert row.match_reason == 'allowlist_did'


def test_handle_event_skips_unknown_author_without_placename(isolated_db: None) -> None:
    from server import config
    from server.indexer import handle_event

    event = _create_event(
        author='did:plc:notallowlisted000000000000',
        text='Our newsletter is out for subscribers.',
        uri_key='xyz',
    )
    handle_event(event)
    assert Post.select().where(Post.uri == event['uri']).count() == 0
    assert config.ALLOWLIST_DIDS
