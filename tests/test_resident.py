"""Resident-tier routing. The labeled set is the local-life policy.

Cases in ``data/resident_cases.json`` document which no-placename posts should
reach the resident decider and which outcomes that prompt is written to
produce. This module checks the floor and the prompt selection. It does not
call OpenRouter; a scripted decider stands in for the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from server.decider import PROMPT_MENTION, PROMPT_RESIDENT, Decision
from server.matcher import match_post

_CASES = json.loads(
    (Path(__file__).resolve().parents[1] / 'data' / 'resident_cases.json').read_text(
        encoding='utf-8'
    )
)
_DID = 'did:plc:residentfixture0000000001'


class _ScriptedDecider:
    def __init__(self, keep: bool) -> None:
        self.keep = keep
        self.prompts: list[str] = []

    def decide(
        self,
        haystack: str,
        *,
        images: tuple[bytes, ...] = (),
        prompt: str = PROMPT_MENTION,
    ) -> Decision:
        del haystack, images
        self.prompts.append(prompt)
        return Decision(keep=self.keep, score=0.91 if self.keep else 0.12)


def test_resident_cases_cover_keeps_and_drops() -> None:
    resident = [case for case in _CASES if case['kind'] == 'resident']
    assert any(case['expected'] is True for case in resident)
    assert any(case['expected'] is False for case in resident)
    assert any(case['kind'] == 'hard_negative' for case in _CASES)
    assert any(case['kind'] == 'mention' for case in _CASES)


def test_labeled_resident_cases_fail_closed_and_use_resident_prompt() -> None:
    for case in _CASES:
        text = str(case['text'])
        kind = str(case['kind'])
        if kind == 'hard_negative':
            decider = _ScriptedDecider(keep=True)
            result = match_post(
                text,
                author_did=_DID,
                resident_dids={_DID},
                decider=decider,
            )
            assert result.matched is False, case['id']
            assert decider.prompts == [], case['id']
            assert result.reason.startswith(('hard_negative', 'entity_other:')), case['id']
            continue

        if kind == 'mention':
            decider = _ScriptedDecider(keep=True)
            result = match_post(
                text,
                author_did=_DID,
                resident_dids={_DID},
                decider=decider,
            )
            assert result.matched is True, case['id']
            assert decider.prompts == [PROMPT_MENTION], case['id']
            assert not result.reason.startswith('decision_keep:resident:'), case['id']
            continue

        closed = match_post(text, author_did=_DID, resident_dids={_DID})
        assert closed.matched is False, case['id']
        outsider = match_post(text, author_did=_DID, decider=_ScriptedDecider(keep=True))
        assert outsider.matched is False, case['id']

        kept_decider = _ScriptedDecider(keep=True)
        kept = match_post(
            text,
            author_did=_DID,
            resident_dids={_DID},
            decider=kept_decider,
        )
        assert kept_decider.prompts == [PROMPT_RESIDENT], case['id']
        assert kept.matched is True, case['id']
        assert kept.reason.startswith('decision_keep:resident:'), case['id']

        dropped_decider = _ScriptedDecider(keep=False)
        dropped = match_post(
            text,
            author_did=_DID,
            resident_dids={_DID},
            decider=dropped_decider,
        )
        assert dropped_decider.prompts == [PROMPT_RESIDENT], case['id']
        assert dropped.matched is False, case['id']
        assert dropped.reason.startswith('decision_reject:resident:'), case['id']


def test_soft_prior_bare_name_stays_on_mention_prompt_for_a_resident() -> None:
    decider = _ScriptedDecider(keep=True)
    result = match_post(
        'Dinner in Troy tonight.',
        author_did=_DID,
        soft_prior_dids={_DID},
        resident_dids={_DID},
        decider=decider,
    )
    assert result.matched is True
    assert decider.prompts == [PROMPT_MENTION]
    assert 'soft_prior_ambiguous' in result.reason


def test_resident_photo_only_post_sends_the_thumbnail(monkeypatch: pytest.MonkeyPatch) -> None:
    jpeg = b'\xff\xd8\xff' + b'\x22' * 8
    seen: dict[str, object] = {}

    class _Recorder:
        def decide(
            self,
            haystack: str,
            *,
            images: tuple[bytes, ...] = (),
            prompt: str = PROMPT_MENTION,
        ) -> Decision:
            seen['haystack'] = haystack
            seen['images'] = images
            seen['prompt'] = prompt
            return Decision(keep=True, score=0.8)

    monkeypatch.setattr('server.matcher.load_thumbnails', lambda *args, **kwargs: (jpeg,))
    result = match_post(
        '',
        author_did=_DID,
        resident_dids={_DID},
        embed={'images': [{'image': {'ref': {'$link': 'bafythumb'}}}]},
        decider=_Recorder(),
    )
    assert result.matched is True
    assert result.reason.startswith('decision_keep:resident:')
    assert seen['images'] == (jpeg,)
    assert seen['prompt'] == PROMPT_RESIDENT


def test_resident_empty_post_without_image_is_unscored() -> None:
    from server.decider import OpenRouterDecider

    def judge(text: str) -> dict[str, object]:
        raise AssertionError(text)

    result = match_post(
        '',
        author_did=_DID,
        resident_dids={_DID},
        decider=OpenRouterDecider(api_key='test', _judge=judge),
    )
    assert result.matched is False
    assert result.reason == 'decision_unscored:resident'
