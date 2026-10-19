"""Content decider unit tests. No network."""

from __future__ import annotations

import base64

import pytest
from server.decider import (
    Decision,
    FakeDecider,
    OpenRouterDecider,
    UnavailableDecider,
    _request_body,
    decision_from_payload,
)
from server.matcher import match_post

_NOUL = {'answers': {'keep': {'type': 'noul', 'noul': 0.62}}}
_STRONG = 'Schenectady City Council meets Tuesday at City Hall.'
_BARE = 'Dinner in Troy tonight.'
_JPEG = b'\xff\xd8\xff' + b'\x00' * 16
_IMAGE_EMBED = {'images': [{'image': {'ref': {'$link': 'bafythumb'}}}]}


def test_noul_threshold_and_choice_payload() -> None:
    kept = decision_from_payload(_NOUL, keep_min=0.5)
    rejected = decision_from_payload(_NOUL, keep_min=0.7)
    assert kept.keep is True
    assert kept.score == 0.62
    assert rejected.keep is False
    assert rejected.score == 0.62
    with pytest.raises(ValueError):
        decision_from_payload(
            {'answers': {'keep': {'type': 'choice', 'choice': 'yes'}}},
            keep_min=0.5,
        )


def test_request_body_is_a_decision_not_a_chat_prompt() -> None:
    body = _request_body('perplexity/pplx-decider-v1.2', 'Schenectady tonight')
    assert 'messages' not in body
    assert body['model'] == 'perplexity/pplx-decider-v1.2'
    assert body['state'] == {'post': 'Schenectady tonight'}
    keep = body['questions']['keep']
    assert keep['type'] == 'noul'
    assert 'Capital Region' in keep['instructions']
    assert 'unclear' in keep['criteria']['false'].lower()


def test_request_body_sends_jpeg_as_base64_data_url() -> None:
    body = _request_body('perplexity/pplx-decider-v1.2', 'Schenectady tonight', (_JPEG,))
    assert 'messages' not in body
    state = body['state']
    assert isinstance(state, list)
    assert state[0] == 'Schenectady tonight'
    part = state[1]
    assert part['type'] == 'image_url'
    url = part['image_url']['url']
    prefix = 'data:image/jpeg;base64,'
    assert url.startswith(prefix)
    assert base64.b64decode(url.removeprefix(prefix)) == _JPEG


def test_request_body_image_only_omits_empty_text() -> None:
    body = _request_body('perplexity/pplx-decider-v1.2', '', (_JPEG,))
    assert body['state'] == [
        {
            'type': 'image_url',
            'image_url': {
                'url': 'data:image/jpeg;base64,' + base64.b64encode(_JPEG).decode('ascii'),
            },
        }
    ]
    with pytest.raises(ValueError):
        _request_body('perplexity/pplx-decider-v1.2', 'text', (b'not-an-image',))


def test_image_bytes_are_part_of_the_cache_key() -> None:
    calls = {'n': 0}

    def judge(text: str) -> dict[str, object]:
        calls['n'] += 1
        assert text == 'Schenectady tonight'
        return {'answers': {'keep': {'type': 'noul', 'noul': 0.8}}}

    decider = OpenRouterDecider(api_key='test', _judge=judge)
    first = decider.decide('Schenectady tonight', images=(_JPEG,))
    second = decider.decide('Schenectady tonight', images=(_JPEG,))
    other = decider.decide('Schenectady tonight', images=(_JPEG + b'\x01',))
    assert first is not None and second == first
    assert other is not None
    assert calls['n'] == 2


def test_openrouter_low_noul_rejects_and_caches() -> None:
    calls = {'n': 0}

    def judge(text: str) -> dict[str, object]:
        calls['n'] += 1
        assert text == 'Schenectady farmers market'
        return {'answers': {'keep': {'type': 'noul', 'noul': 0.2}}}

    decider = OpenRouterDecider(api_key='test', keep_min=0.5, _judge=judge)
    first = decider.decide('Schenectady farmers market')
    second = decider.decide('Schenectady farmers market')
    assert first is not None
    assert first.keep is False
    assert first.score == 0.2
    assert second == first
    assert calls['n'] == 1


def test_empty_haystack_is_not_sent_or_cached() -> None:
    calls = {'n': 0}

    def judge(text: str) -> dict[str, object]:
        calls['n'] += 1
        return {'answers': {'keep': {'type': 'noul', 'noul': 0.9}}}

    decider = OpenRouterDecider(api_key='test', _judge=judge)
    assert decider.decide('   ') is None
    assert decider.decide('   ') is None
    assert calls['n'] == 0


def test_unusable_payload_is_cached_as_decline() -> None:
    calls = {'n': 0}

    def judge(text: str) -> dict[str, object]:
        del text
        calls['n'] += 1
        return {'answers': {'keep': {'type': 'choice', 'choice': 'yes'}}}

    decider = OpenRouterDecider(api_key='test', _judge=judge)
    assert decider.decide('Schenectady farmers market') is None
    assert decider.decide('Schenectady farmers market') is None
    assert calls['n'] == 1


def test_fake_reject_and_keep_wrap_regex_reason() -> None:
    baseline = match_post(_STRONG)
    assert baseline.matched is True

    rejected = match_post(_STRONG, decider=FakeDecider(keep=False))
    assert rejected.matched is False
    assert rejected.reason.startswith(f'decision_reject:{baseline.reason}:')

    kept = match_post(_STRONG, decider=FakeDecider(keep=True, score=0.88))
    assert kept.matched is True
    assert kept.reason == f'decision_keep:{baseline.reason}:0.880'


def test_bare_ambiguous_does_not_call_decider() -> None:
    baseline = match_post(_BARE)
    assert baseline.matched is False
    flipped = match_post(_BARE, decider=FakeDecider(keep=True))
    assert flipped.matched is False
    assert flipped.reason == baseline.reason


def test_soft_prior_bare_ambiguous_is_decided() -> None:
    did = 'did:plc:softdecider0000000000001'
    kept = match_post(
        _BARE,
        author_did=did,
        soft_prior_dids={did},
        decider=FakeDecider(keep=True, score=0.88),
    )
    assert kept.matched is True
    assert kept.reason.startswith('decision_keep:')
    assert 'soft_prior_ambiguous' in kept.reason

    rejected = match_post(
        _BARE,
        author_did=did,
        soft_prior_dids={did},
        decider=FakeDecider(keep=False, score=0.2),
    )
    assert rejected.matched is False
    assert rejected.reason.startswith('decision_reject:')
    assert 'soft_prior_ambiguous' in rejected.reason


def test_unavailable_decider_is_unscored() -> None:
    baseline = match_post(_STRONG)
    result = match_post(_STRONG, decider=UnavailableDecider())
    assert result.matched is False
    assert result.reason == f'decision_unscored:{baseline.reason}'


def test_hard_negative_is_not_a_candidate() -> None:
    text = 'Nice day in Albany Park.'
    baseline = match_post(text)
    assert baseline.matched is False
    result = match_post(text, decider=FakeDecider(keep=True))
    assert result.matched is False
    assert result.reason == baseline.reason


def test_fake_reject_substrings_win() -> None:
    decider = FakeDecider(
        keep=True,
        keep_substrings=('schenectady',),
        reject_substrings=('hate',),
    )
    rejected = decider.decide('I hate Schenectady')
    assert rejected is not None and rejected.keep is False
    kept = decider.decide('Schenectady city council')
    assert kept is not None and kept.keep is True
    neither = decider.decide('hello troy')
    assert neither is not None and neither.keep is False


def test_default_decider_follows_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from server.decider import clear_decider_cache, get_default_decider

    monkeypatch.setenv('DECISION_ENABLED', 'false')
    clear_decider_cache()
    assert get_default_decider() is None

    monkeypatch.setenv('DECISION_ENABLED', 'true')
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    clear_decider_cache()
    missing = get_default_decider()
    assert isinstance(missing, UnavailableDecider)
    assert get_default_decider() is missing

    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key')
    monkeypatch.setenv('DECISION_MODEL', 'perplexity/pplx-decider-v1.2')
    clear_decider_cache()
    client = get_default_decider()
    assert isinstance(client, OpenRouterDecider)
    assert client.model == 'perplexity/pplx-decider-v1.2'
    assert get_default_decider() is client
    clear_decider_cache()


class _RecordingDecider:
    def __init__(self) -> None:
        self.images: tuple[bytes, ...] | None = None

    def decide(self, haystack: str, *, images: tuple[bytes, ...] = ()) -> Decision:
        del haystack
        self.images = images
        return Decision(keep=True, score=0.88)


def test_match_post_passes_thumbnail_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('server.matcher.load_thumbnails', lambda *args, **kwargs: (_JPEG,))
    recorder = _RecordingDecider()
    result = match_post(
        _STRONG,
        author_did='did:plc:thumb',
        embed=_IMAGE_EMBED,
        decider=recorder,
    )
    assert recorder.images == (_JPEG,)
    assert result.matched is True
    assert result.reason.startswith('decision_keep:')


def test_thumbnail_fetch_failure_is_unscored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('server.matcher.load_thumbnails', lambda *args, **kwargs: None)
    result = match_post(
        _STRONG,
        author_did='did:plc:thumb',
        embed=_IMAGE_EMBED,
        decider=FakeDecider(keep=True),
    )
    assert result.matched is False
    baseline = match_post(_STRONG)
    assert result.reason == f'decision_unscored:{baseline.reason}'


def test_decider_off_and_non_candidates_do_not_fetch_thumbnails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object, **kwargs: object) -> tuple[bytes, ...]:
        raise AssertionError('thumbnail fetch')

    monkeypatch.setattr('server.matcher.load_thumbnails', boom)
    kept = match_post(_STRONG, author_did='did:plc:thumb', embed=_IMAGE_EMBED)
    assert kept.matched is True
    assert not kept.reason.startswith('decision_')
    dropped = match_post(
        _BARE,
        author_did='did:plc:thumb',
        embed=_IMAGE_EMBED,
        decider=FakeDecider(keep=True),
    )
    assert dropped.matched is False
    assert dropped.reason == match_post(_BARE).reason
