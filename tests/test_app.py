from unittest.mock import patch

from fastapi.testclient import TestClient


def _noop_jetstream(*_args, **_kwargs):
    return None


@patch('server.app.run_jetstream', _noop_jetstream)
def test_healthz():
    from server.app import app

    with TestClient(app) as client:
        response = client.get('/healthz')
    assert response.status_code == 200
    assert response.json() == {'ok': True}


@patch('server.app.run_jetstream', _noop_jetstream)
def test_describe_feed_generator():
    from server import config
    from server.app import app

    with TestClient(app) as client:
        response = client.get('/xrpc/app.bsky.feed.describeFeedGenerator')
    assert response.status_code == 200
    body = response.json()
    assert body['did'] == config.SERVICE_DID
    assert {'uri': config.FEED_URI} in body['feeds']


@patch('server.app.run_jetstream', _noop_jetstream)
def test_get_feed_skeleton_unsupported_algorithm():
    from server.app import app

    with TestClient(app) as client:
        response = client.get(
            '/xrpc/app.bsky.feed.getFeedSkeleton',
            params={'feed': 'at://did:plc:unknown/app.bsky.feed.generator/nope'},
        )
    assert response.status_code == 400
    assert response.json()['error'] == 'UnsupportedAlgorithm'


@patch('server.app.run_jetstream', _noop_jetstream)
def test_did_json_for_did_web_hostname():
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
