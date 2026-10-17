"""Live OpenRouter decider checks.

Run with ``pytest -m live`` and ``OPENROUTER_API_KEY`` set. The default suite
skips this module and does not call the network.
"""

from __future__ import annotations

import os

import pytest
from server.decider import OpenRouterDecider
from server.matcher import match_post

pytestmark = pytest.mark.live

_IN_SCOPE = 'Schenectady City Council meets Tuesday at City Hall to discuss the new CDTA bus route.'
_HOSTILE = 'I hate everyone in Schenectady. You are all garbage and this city should burn down.'


@pytest.fixture
def live_decider() -> OpenRouterDecider:
    key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not key:
        pytest.skip('OPENROUTER_API_KEY unset')
    return OpenRouterDecider(api_key=key, timeout_s=30)


def test_live_in_scope_post_is_kept(live_decider: OpenRouterDecider) -> None:
    floor = match_post(_IN_SCOPE)
    assert floor.matched is True
    decision = live_decider.decide(_IN_SCOPE)
    assert decision is not None and decision.keep is True
    result = match_post(_IN_SCOPE, decider=live_decider)
    assert result.matched is True
    assert result.reason.startswith('decision_keep:')


def test_live_hostile_post_is_rejected(live_decider: OpenRouterDecider) -> None:
    floor = match_post(_HOSTILE)
    assert floor.matched is True
    decision = live_decider.decide(_HOSTILE)
    assert decision is not None and decision.keep is False
    result = match_post(_HOSTILE, decider=live_decider)
    assert result.matched is False
    assert result.reason.startswith('decision_reject:')
