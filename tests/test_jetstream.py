"""Jetstream consumer helpers: catch-up engagement skip and cursor lag."""

from __future__ import annotations

import time

import pytest
from server.database import SubscriptionState, db
from server.jetstream import (
    _SKIP_TO_LIVE_LAG_S,
    _parse_post_event,
    cursor_lag_seconds,
    maybe_skip_cursor_to_live,
)


@pytest.fixture(autouse=True)
def _jetstream_tables() -> None:
    db.create_tables([SubscriptionState])


def test_cursor_lag_seconds_from_explicit_cursor() -> None:
    now_us = 1_700_000_000_000_000
    cursor = now_us - 90_000_000  # 90s behind
    assert cursor_lag_seconds(cursor, now_us=now_us) == 90.0


def test_maybe_skip_cursor_to_live_clears_deep_lag() -> None:
    service = 'did:web:test-skip-live'
    now_us = 1_700_000_000_000_000
    deep_cursor = now_us - int((_SKIP_TO_LIVE_LAG_S + 60) * 1_000_000)
    SubscriptionState.delete().where(SubscriptionState.service == service).execute()
    SubscriptionState.create(service=service, cursor=deep_cursor)

    assert maybe_skip_cursor_to_live(service, deep_cursor, now_us=now_us) is None
    state = SubscriptionState.get(SubscriptionState.service == service)
    assert int(state.cursor) == 0


def test_maybe_skip_cursor_to_live_keeps_modest_lag() -> None:
    service = 'did:web:test-keep-cursor'
    now_us = 1_700_000_000_000_000
    modest_cursor = now_us - 60_000_000  # 60s
    SubscriptionState.delete().where(SubscriptionState.service == service).execute()
    SubscriptionState.create(service=service, cursor=modest_cursor)

    assert maybe_skip_cursor_to_live(service, modest_cursor, now_us=now_us) == modest_cursor
    state = SubscriptionState.get(SubscriptionState.service == service)
    assert int(state.cursor) == modest_cursor


def test_parse_skips_engagement_when_catching_up() -> None:
    like = {
        'kind': 'commit',
        'did': 'did:plc:liker',
        'time_us': int(time.time() * 1_000_000),
        'commit': {
            'collection': 'app.bsky.feed.like',
            'rkey': 'abc',
            'operation': 'create',
            'record': {'subject': {'uri': 'at://did:plc:x/app.bsky.feed.post/y'}},
        },
    }
    assert _parse_post_event(like, skip_engagement=True) is None
    parsed = _parse_post_event(like, skip_engagement=False)
    assert parsed is not None
    assert parsed['engagement'] == 'like'


def test_parse_still_returns_posts_when_skipping_engagement() -> None:
    post = {
        'kind': 'commit',
        'did': 'did:plc:author',
        'time_us': int(time.time() * 1_000_000),
        'commit': {
            'collection': 'app.bsky.feed.post',
            'rkey': 'post1',
            'operation': 'create',
            'cid': 'bafy',
            'record': {'text': 'hello', 'createdAt': '2026-07-31T12:00:00.000Z'},
        },
    }
    event = _parse_post_event(post, skip_engagement=True)
    assert event is not None
    assert event['uri'].endswith('/post1')
    assert 'engagement' not in event
