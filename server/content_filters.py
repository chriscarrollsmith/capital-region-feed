"""Index-time author and content filters outside the geography matcher.

Author blocklists drop curated bad-fit accounts even when they match Cap Region
cues. Content mutes drop posts with built-in prosocial patterns and optional
``MUTED_KEYWORDS`` substrings.
"""

from __future__ import annotations

import re

# Word-boundary patterns always applied (case-insensitive). Prefer regex over
# bare substrings so short tokens like "acab" do not hit words such as "macabre".
_BUILTIN_MUTE_RE = re.compile(
    r'(?i)(?:\ba\.?c\.?a\.?b\b|\ball\s+cops\s+are\s+bastards\b)',
)


def author_is_blocked(
    author_did: str | None,
    author_handle: str | None = None,
    *,
    blocklist_dids: set[str],
    blocklist_handles: set[str],
) -> bool:
    """True when the author DID or handle is on the feed blocklist."""
    if author_did and author_did in blocklist_dids:
        return True
    if author_handle and author_handle.lower() in blocklist_handles:
        return True
    return False


def text_is_muted(
    text: str,
    alt_text: str = '',
    *,
    keywords: tuple[str, ...] = (),
) -> bool:
    """True when post text/alt matches a built-in mute or env keyword."""
    haystack = f'{text} {alt_text}'
    if _BUILTIN_MUTE_RE.search(haystack):
        return True
    if not keywords:
        return False
    lower = haystack.lower()
    return any(keyword in lower for keyword in keywords)
