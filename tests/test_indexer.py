"""Indexer path tests: allowlisted DIDs and soft author priors."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from server.database import AuthorDecision, AuthorLocalStats, Post, SubscriptionState, db

_TABLES = [Post, SubscriptionState, AuthorLocalStats, AuthorDecision]


@pytest.fixture()
def isolated_db() -> Any:
    """Rebuild post tables so each test starts empty."""
    db.connect(reuse_if_open=True)
    db.drop_tables(_TABLES, safe=True)
    db.create_tables(_TABLES)
    yield
    AuthorDecision.delete().execute()
    AuthorLocalStats.delete().execute()
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


def test_handle_event_skips_blocklisted_author_even_with_local_text(isolated_db: None) -> None:
    from server import config
    from server.indexer import handle_event

    assert config.BLOCKLIST_DIDS, 'production DID blocklist must be populated'
    author = sorted(config.BLOCKLIST_DIDS)[0]
    event = _create_event(
        author=author,
        text='Live from Albany NY — Empire State Plaza tonight.',
        uri_key='blocked',
    )
    handle_event(event)
    assert Post.select().where(Post.uri == event['uri']).count() == 0


def test_handle_event_skips_acab_assertion_even_with_local_text(isolated_db: None) -> None:
    from server.indexer import handle_event

    author = 'did:plc:acabfiltertest0000000000001'
    event = _create_event(
        author=author,
        text='ACAB rally near the Empire State Plaza in Albany.',
        uri_key='acab',
    )
    handle_event(event)
    assert Post.select().where(Post.uri == event['uri']).count() == 0


def test_regex_keep_without_decider_records_no_author_decision(isolated_db: None) -> None:
    from server.indexer import handle_event

    author = 'did:plc:indexersoftprior0000000001'
    event = _create_event(author=author, text='Hello from Schenectady.', uri_key='regex')
    handle_event(event)
    assert Post.get(Post.uri == event['uri']).match_reason
    assert not str(Post.get(Post.uri == event['uri']).match_reason).startswith('decision_')
    assert AuthorDecision.select().count() == 0


def test_decider_keeps_earn_prior_and_reject_drops_post(
    isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server import config
    from server.author_priors import author_has_soft_prior
    from server.decider import FakeDecider, UnavailableDecider
    from server.indexer import handle_event

    author = 'did:plc:deciderprior0000000000001'
    assert author not in config.ALLOWLIST_DIDS
    holder: dict[str, FakeDecider | UnavailableDecider | None] = {'decider': None}
    monkeypatch.setattr('server.decider.get_default_decider', lambda: holder['decider'])

    holder['decider'] = FakeDecider(keep=True, score=0.91)
    for i in range(config.SOFT_PRIOR_MIN_KEEPS):
        handle_event(
            _create_event(
                author=author,
                text=f'Hello from Schenectady update {i}.',
                uri_key=f'keep{i}',
            )
        )
    assert author_has_soft_prior(author)

    bare = _create_event(author=author, text='Dinner in Troy tonight.', uri_key='bare')
    handle_event(bare)
    row = Post.get(Post.uri == bare['uri'])
    assert str(row.match_reason).startswith('decision_keep:')
    assert 'soft_prior_ambiguous' in str(row.match_reason)

    holder['decider'] = FakeDecider(keep=False, score=0.12)
    rejected = _create_event(
        author=author,
        text='Another hello from Schenectady.',
        uri_key='reject',
    )
    handle_event(rejected)
    assert Post.select().where(Post.uri == rejected['uri']).count() == 0
    assert not author_has_soft_prior(author)

    holder['decider'] = UnavailableDecider()
    unscored = _create_event(
        author=author,
        text='Hello from Schenectady again.',
        uri_key='unscored',
    )
    handle_event(unscored)
    assert Post.select().where(Post.uri == unscored['uri']).count() == 0
    assert AuthorDecision.select().where(AuthorDecision.kept == 1).count() == (
        config.SOFT_PRIOR_MIN_KEEPS + 1
    )


def test_handle_event_passes_thumbnail_bytes(
    isolated_db: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server.decider import Decision
    from server.indexer import handle_event

    jpeg = b'\xff\xd8\xff' + b'\x11' * 8
    seen: dict[str, object] = {}

    class _Recorder:
        def decide(self, haystack: str, *, images: tuple[bytes, ...] = ()) -> Decision:
            seen['haystack'] = haystack
            seen['images'] = images
            return Decision(keep=True, score=0.77)

    monkeypatch.setattr('server.decider.get_default_decider', lambda: _Recorder())

    def _load(author_did: str | None, embed: object | None, *, timeout_s: float | None = None):
        del author_did, timeout_s
        if embed is None:
            return ()
        return (jpeg,)

    monkeypatch.setattr('server.matcher.load_thumbnails', _load)
    author = 'did:plc:thumbindex00000000000001'
    event = _create_event(author=author, text='Hello from Schenectady.', uri_key='img')
    event['record']['embed'] = {
        '$type': 'app.bsky.embed.images',
        'images': [{'image': {'ref': {'$link': 'bafythumb'}}}],
    }
    handle_event(event)
    row = Post.get(Post.uri == event['uri'])
    assert str(row.match_reason).startswith('decision_keep:')
    assert seen['images'] == (jpeg,)
    assert 'Schenectady' in str(seen['haystack'])


def test_handle_event_archived_uses_event_time_not_wall_clock(isolated_db: None) -> None:
    """Catch-up must not drop posts that were fresh at firehose time_us."""
    from datetime import timedelta

    from server import config
    from server.indexer import handle_event

    author = sorted(config.ALLOWLIST_DIDS)[0]
    # Created ~30h before wall clock, but event time_us is only 2h after createdAt.
    created = datetime.now(UTC) - timedelta(hours=30)
    event_time = created + timedelta(hours=2)
    event = {
        'operation': 'create',
        'uri': f'at://{author}/app.bsky.feed.post/catchup1',
        'cid': 'bafytestcid',
        'author': author,
        'time_us': int(event_time.timestamp() * 1_000_000),
        'record': {
            '$type': 'app.bsky.feed.post',
            'text': 'Catch-up local bulletin for subscribers.',
            'createdAt': created.isoformat().replace('+00:00', 'Z'),
        },
    }
    handle_event(event)
    assert Post.select().where(Post.uri == event['uri']).count() == 1

    # Same createdAt with a live-tail time_us (wall-aligned) is archived.
    live = dict(event)
    live['uri'] = f'at://{author}/app.bsky.feed.post/catchup2'
    live['time_us'] = int(datetime.now(UTC).timestamp() * 1_000_000)
    handle_event(live)
    assert Post.select().where(Post.uri == live['uri']).count() == 0
