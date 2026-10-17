"""Content decider unit tests. No network."""

from __future__ import annotations

import pytest
from server.decider import (
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
    body = _request_body('perplexity/pplx-decider-v1.1-27b', 'Schenectady tonight')
    assert 'messages' not in body
    assert body['model'] == 'perplexity/pplx-decider-v1.1-27b'
    assert body['state'] == {'post': 'Schenectady tonight'}
    keep = body['questions']['keep']
    assert keep['type'] == 'noul'
    assert 'Capital Region' in keep['instructions']
    assert 'unclear' in keep['criteria']['false'].lower()


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
    monkeypatch.setenv('DECISION_MODEL', 'perplexity/pplx-decider-v1.1-27b')
    clear_decider_cache()
    client = get_default_decider()
    assert isinstance(client, OpenRouterDecider)
    assert client.model == 'perplexity/pplx-decider-v1.1-27b'
    assert get_default_decider() is client
    clear_decider_cache()
