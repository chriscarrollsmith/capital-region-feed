"""Live OpenRouter decider checks.

Run with ``pytest -m live`` and ``OPENROUTER_API_KEY`` set. The default suite
skips this module and does not call the network.
"""

from __future__ import annotations

import os

import pytest
from server.decider import OpenRouterDecider
from server.matcher import match_post
from server.thumbnails import load_thumbnails

pytestmark = pytest.mark.live

_IN_SCOPE = 'Schenectady City Council meets Tuesday at City Hall to discuss the new CDTA bus route.'
_HOSTILE = 'I hate everyone in Schenectady. You are all garbage and this city should burn down.'
_THUMB_DID = 'did:plc:pjzh6ycskkvvi56nzch7tbhb'
_THUMB_CID = 'bafkreihfldlzs7q3ta5n33gfnhnh7jq5tapfug7uq5tkualwrcgqqskncq'
_THUMB_EMBED = {
    '$type': 'app.bsky.embed.images',
    'images': [{'alt': '', 'image': {'ref': {'$link': _THUMB_CID}, 'mimeType': 'image/jpeg'}}],
}


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


def test_live_thumbnail_base64_with_text(
    live_decider: OpenRouterDecider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real feed_thumbnail JPEG is scored with the post text.

    The photo can move ``noul``, so this asserts a scored decision rather than
    a particular keep or reject.
    """
    images = load_thumbnails(_THUMB_DID, _THUMB_EMBED, timeout_s=20)
    assert images is not None and len(images) == 1
    assert images[0].startswith(b'\xff\xd8\xff')
    decision = live_decider.decide(_IN_SCOPE, images=images)
    assert decision is not None
    assert 0.0 <= decision.score <= 1.0
    monkeypatch.setenv('DECISION_TIMEOUT_S', '20')
    result = match_post(
        _IN_SCOPE,
        author_did=_THUMB_DID,
        embed=_THUMB_EMBED,
        decider=live_decider,
    )
    assert result.reason.startswith(('decision_keep:', 'decision_reject:'))
    again = live_decider.decide(_IN_SCOPE, images=images)
    assert again is decision
