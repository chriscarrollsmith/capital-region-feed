import os

import pytest

# Keep config imports safe if a test pulls in server packages beyond matcher.
os.environ.setdefault('HOSTNAME', 'test.example')
os.environ.setdefault(
    'FEED_URI',
    'at://did:plc:xndplob7sicvv6balxdzh3jk/app.bsky.feed.generator/aaagkkw3yejuk',
)
os.environ.setdefault('DATABASE_PATH', ':memory:')
# Before app import. A developer .env may still override this via load_dotenv;
# the autouse fixture below forces the offline default for every test.
os.environ.setdefault('DECISION_ENABLED', 'false')


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip live OpenRouter tests unless the mark expression selects them."""
    markexpr = str(config.getoption('markexpr') or '')
    selects_live = 'live' in markexpr and 'not live' not in markexpr
    if selects_live:
        return
    skip_live = pytest.mark.skip(reason='live OpenRouter tests run only with pytest -m live')
    for item in items:
        if item.get_closest_marker('live') is not None:
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def _decision_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the default suite off the network even if .env enables the decider."""
    monkeypatch.setenv('DECISION_ENABLED', 'false')
    from server.decider import clear_decider_cache

    clear_decider_cache()
