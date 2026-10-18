"""Perplexity content decider for regex-floor candidates.

The matcher proposes candidates. This module keep/rejects only those candidates
via OpenRouter's Decisions API (a ``noul`` probability). Missing credentials,
timeouts, transport errors, and unusable payloads decline. Callers treat
``None`` as a drop.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

PROMPT_VERSION = 'capital-region-keep-v1'

DEFAULT_MODEL = 'perplexity/pplx-decider-v1.2'
DEFAULT_API_URL = 'https://openrouter.ai/api/alpha/decisions'
DEFAULT_KEEP_MIN = 0.5
DEFAULT_TIMEOUT_S = 8.0
DEFAULT_MAX_TEXT_CHARS = 2000

DECISION_KEEP_PREFIX = 'decision_keep:'
DECISION_REJECT_PREFIX = 'decision_reject:'
DECISION_UNSCORED_PREFIX = 'decision_unscored:'

USER_AGENT = (
    'capital-region-feed-decider/1 (+https://github.com/chriscarrollsmith/capital-region-feed)'
)

# Editorial policy for this feed only. Bump PROMPT_VERSION when this text changes.
_KEEP_INSTRUCTIONS = (
    "This feed keeps constructive posts from and about New York's Capital Region "
    '(Albany, Troy, Schenectady, Saratoga Springs, and nearby): local news, civic life, '
    'culture, neighborhoods, everyday community posts, and events at Capital Region venues, '
    'including clearly local voices. Out of scope are other places that only share a name, '
    'national or off-region topics, hostile or abusive posts, spam, and anything too unclear '
    'to place in the Capital Region.'
)
_CRITERION_TRUE = (
    "Constructive and in scope for New York's Capital Region: local news, civic or community "
    'life, culture, events, or daily life in Albany, Troy, Schenectady, Saratoga, or nearby '
    'Capital Region places, including a clearly local voice or a regional venue event.'
)
_CRITERION_FALSE = (
    'Off-topic, hostile, abusive, spam, or unclear, or a known false friend of this feed: '
    "Albany Park (Chicago), New Albany, another state's Albany, the DC National Capital Region, "
    'NFL JC Latham, French colonie, Albany Road, Saratoga Springs Utah, or any other homograph '
    "that is not New York's Capital Region."
)

_TRUTHY = frozenset({'1', 'true', 't', 'yes', 'y'})
_DECLINE_ERRORS = (
    ValueError,
    urllib.error.URLError,
    TimeoutError,
    json.JSONDecodeError,
    KeyError,
    TypeError,
)

_singleton: DecisionBackend | None = None
_singleton_lock = threading.Lock()


@dataclass(frozen=True)
class Decision:
    keep: bool
    score: float


class DecisionBackend(Protocol):
    def decide(self, haystack: str, *, images: tuple[bytes, ...] = ()) -> Decision | None:
        """Return a keep/reject, or None when the post cannot be scored.

        ``images`` are raw thumbnail bytes sent with the post text. Text-only
        callers omit them.
        """


def decision_from_payload(payload: object, *, keep_min: float) -> Decision:
    """Parse a Decisions API body. Raises ValueError unless ``keep`` is in-range noul."""
    if not isinstance(payload, dict):
        raise ValueError('decision payload must be an object')
    answers = payload.get('answers')
    if not isinstance(answers, dict):
        raise ValueError('answers must be a dict')
    keep = answers.get('keep')
    if not isinstance(keep, dict):
        raise ValueError('answers.keep must be a dict')
    if keep.get('type') != 'noul':
        raise ValueError('answers.keep.type must be noul')
    noul = keep.get('noul')
    if isinstance(noul, bool) or not isinstance(noul, int | float):
        raise ValueError('noul must be a float in [0, 1]')
    score = float(noul)
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError('noul out of range')
    return Decision(keep=score >= keep_min, score=score)


def _image_mime(data: bytes) -> str:
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data.startswith((b'GIF87a', b'GIF89a')):
        return 'image/gif'
    if len(data) >= 12 and data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return 'image/webp'
    raise ValueError('unsupported image type')


def _image_part(data: bytes) -> dict[str, Any]:
    encoded = base64.b64encode(data).decode('ascii')
    return {
        'type': 'image_url',
        'image_url': {'url': f'data:{_image_mime(data)};base64,{encoded}'},
    }


def _request_body(
    model: str,
    text: str,
    images: tuple[bytes, ...] = (),
) -> dict[str, Any]:
    if images:
        parts: list[Any] = []
        if text:
            parts.append(text)
        parts.extend(_image_part(blob) for blob in images)
        state: Any = parts
    else:
        state = {'post': text}
    return {
        'model': model,
        'state': state,
        'questions': {
            'keep': {
                'type': 'noul',
                'instructions': _KEEP_INSTRUCTIONS,
                'criteria': {
                    'true': _CRITERION_TRUE,
                    'false': _CRITERION_FALSE,
                },
            }
        },
    }


class OpenRouterDecider:
    """Synchronous Decisions API client. Fail closed; cache scored and unscored results."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        api_url: str = DEFAULT_API_URL,
        keep_min: float = DEFAULT_KEEP_MIN,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
        _judge: Callable[[str], object] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.keep_min = keep_min
        self.timeout_s = timeout_s
        self.max_text_chars = max_text_chars
        self._judge = _judge
        self._cache: dict[str, Decision | None] = {}
        self._cache_lock = threading.Lock()

    def decide(self, haystack: str, *, images: tuple[bytes, ...] = ()) -> Decision | None:
        clipped = haystack.strip()[: self.max_text_chars]
        if not clipped and not images:
            return None
        image_token = ''
        if images:
            image_token = '|' + '|'.join(hashlib.sha256(blob).hexdigest() for blob in images)
        key = hashlib.sha256(
            f'{self.model}|{PROMPT_VERSION}|{self.keep_min:.3f}|{clipped}{image_token}'.encode()
        ).hexdigest()
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]
        try:
            payload = (
                self._judge(clipped) if self._judge is not None else self._post(clipped, images)
            )
            decision: Decision | None = decision_from_payload(payload, keep_min=self.keep_min)
        except _DECLINE_ERRORS:
            decision = None
        with self._cache_lock:
            # Another thread may have stored the same key; either result is usable.
            return self._cache.setdefault(key, decision)

    def _post(self, text: str, images: tuple[bytes, ...] = ()) -> object:
        body = json.dumps(_request_body(self.model, text, images)).encode('utf-8')
        request = urllib.request.Request(
            self.api_url,
            data=body,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'User-Agent': USER_AGENT,
            },
            method='POST',
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode('utf-8'))


class UnavailableDecider:
    """Enabled filter with no API key. Every candidate declines."""

    def decide(self, haystack: str, *, images: tuple[bytes, ...] = ()) -> Decision | None:
        del haystack, images
        return None


class FakeDecider:
    """In-memory decider for tests. Reject substrings win, then keep substrings."""

    def __init__(
        self,
        keep: bool,
        score: float = 0.91,
        keep_substrings: tuple[str, ...] = (),
        reject_substrings: tuple[str, ...] = (),
    ) -> None:
        self.keep = keep
        self.score = score
        self.keep_substrings = keep_substrings
        self.reject_substrings = reject_substrings

    def decide(self, haystack: str, *, images: tuple[bytes, ...] = ()) -> Decision | None:
        del images
        text = haystack.lower()
        if any(needle.lower() in text for needle in self.reject_substrings):
            return Decision(keep=False, score=self.score)
        if any(needle.lower() in text for needle in self.keep_substrings):
            return Decision(keep=True, score=self.score)
        if self.keep_substrings:
            return Decision(keep=False, score=self.score)
        return Decision(keep=self.keep, score=self.score)


def decision_enabled() -> bool:
    """Read ``DECISION_ENABLED`` at call time. Do not cache this at import."""
    raw = os.environ.get('DECISION_ENABLED', '')
    return raw.strip().lower() in _TRUTHY


def build_default_decider() -> DecisionBackend:
    """Build the process client from the environment. Missing key → unavailable."""
    api_key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not api_key:
        return UnavailableDecider()
    return OpenRouterDecider(
        api_key=api_key,
        model=os.environ.get('DECISION_MODEL', DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        api_url=os.environ.get('DECISION_API_URL', DEFAULT_API_URL).strip() or DEFAULT_API_URL,
        keep_min=_env_float('DECISION_KEEP_MIN', DEFAULT_KEEP_MIN),
        timeout_s=_env_float('DECISION_TIMEOUT_S', DEFAULT_TIMEOUT_S),
        max_text_chars=_env_int('DECISION_MAX_TEXT_CHARS', DEFAULT_MAX_TEXT_CHARS),
    )


def get_default_decider() -> DecisionBackend | None:
    """None when the filter is off. One process-wide client when it is on."""
    global _singleton
    if not decision_enabled():
        return None
    with _singleton_lock:
        if _singleton is None:
            _singleton = build_default_decider()
        return _singleton


def clear_decider_cache() -> None:
    """Drop the process-wide client so tests can rebuild it."""
    global _singleton
    with _singleton_lock:
        _singleton = None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)
