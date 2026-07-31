from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from server.database import AuthorLocalStats, Post, SubscriptionState, db


def _noop_jetstream(*_args: Any, **_kwargs: Any) -> None:
    return None


@pytest.fixture(autouse=True)
def _app_tables() -> None:
    db.create_tables([Post, SubscriptionState, AuthorLocalStats])


@patch('server.app.run_jetstream', _noop_jetstream)
def test_healthz() -> None:
    from server.app import app

    SubscriptionState.delete().execute()
    with TestClient(app) as client:
        response = client.get('/healthz')
    assert response.status_code == 200
    body = response.json()
    assert body['ok'] is True
    # No subscription cursor in the test DB → lag fields omitted.
    assert 'jetstream_lag_s' not in body


@patch('server.app.run_jetstream', _noop_jetstream)
def test_healthz_reports_jetstream_lag() -> None:
    import time

    from server import config
    from server.app import app

    SubscriptionState.delete().execute()
    # ~1 hour behind live.
    SubscriptionState.create(
        service=config.SERVICE_DID,
        cursor=int(time.time() * 1_000_000) - 3_600_000_000,
    )
    with TestClient(app) as client:
        response = client.get('/healthz')
    assert response.status_code == 200
    body = response.json()
    assert body['ok'] is True
    assert body['jetstream_ok'] is False
    assert body['jetstream_lag_s'] >= 3500
    SubscriptionState.delete().execute()


@patch('server.app.run_jetstream', _noop_jetstream)
def test_describe_feed_generator() -> None:
    from server import config
    from server.app import app

    with TestClient(app) as client:
        response = client.get('/xrpc/app.bsky.feed.describeFeedGenerator')
    assert response.status_code == 200
    body = response.json()
    assert body['did'] == config.SERVICE_DID
    assert {'uri': config.FEED_URI} in body['feeds']


@patch('server.app.run_jetstream', _noop_jetstream)
def test_get_feed_skeleton_unsupported_algorithm() -> None:
    from server.app import app

    with TestClient(app) as client:
        response = client.get(
            '/xrpc/app.bsky.feed.getFeedSkeleton',
            params={'feed': 'at://did:plc:unknown/app.bsky.feed.generator/nope'},
        )
    assert response.status_code == 400
    assert response.json()['error'] == 'UnsupportedAlgorithm'


@patch('server.app.run_jetstream', _noop_jetstream)
def test_did_json_for_did_web_hostname() -> None:
    from server import config
    from server.app import app

    with TestClient(app) as client:
        response = client.get('/.well-known/did.json')

    if config.SERVICE_DID.endswith(config.HOSTNAME):
        assert response.status_code == 200
        body = response.json()
        assert body['id'] == config.SERVICE_DID
        assert body['service'][0]['type'] == 'BskyFeedGenerator'
    else:
        assert response.status_code == 404
