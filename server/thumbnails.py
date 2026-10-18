"""Bluesky feed-thumbnail bytes for the content decider.

Jetstream image embeds carry blob CIDs, not pixels. This module resolves those
CIDs to the public ``feed_thumbnail`` JPEG and returns the bytes. Callers
send them to the decider as base64. A missing CID, a missing author DID, or a
failed fetch returns None so the caller can fail closed.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from urllib.parse import quote

THUMBNAIL_HOST = 'https://cdn.bsky.app'
_MAX_IMAGES = 4
_MAX_BYTES = 1_048_576
_USER_AGENT = (
    'capital-region-feed-decider/1 (+https://github.com/chriscarrollsmith/capital-region-feed)'
)


def thumbnail_url(author_did: str, cid: str) -> str:
    """Public JPEG thumbnail for one blob on this author's repo."""
    did = quote(author_did, safe=':')
    blob = quote(cid, safe='')
    return f'{THUMBNAIL_HOST}/img/feed_thumbnail/plain/{did}/{blob}@jpeg'


def image_blob_cids(embed: object | None) -> list[str] | None:
    """CIDs for this post's own images and link-card thumbs.

    Quoted records are not walked. Returns [] when there is nothing to fetch.
    Returns None when an image slot is present but has no CID.
    """
    cids: list[str] = []
    broken = False

    def add_blob(blob: object) -> None:
        nonlocal broken
        if broken or len(cids) >= _MAX_IMAGES:
            return
        cid = _blob_cid(blob)
        if cid is None:
            broken = True
            return
        cids.append(cid)

    def walk(node: object) -> None:
        if broken or not isinstance(node, dict):
            return
        images = node.get('images')
        if isinstance(images, list):
            for image in images:
                if not isinstance(image, dict):
                    add_blob(None)
                    continue
                add_blob(image.get('image'))
        external = node.get('external')
        if isinstance(external, dict) and 'thumb' in external:
            add_blob(external.get('thumb'))
        media = node.get('media')
        if isinstance(media, dict):
            walk(media)

    if not isinstance(embed, dict):
        return []
    walk(embed)
    if broken:
        return None
    return cids


def load_thumbnails(
    author_did: str | None,
    embed: object | None,
    *,
    timeout_s: float | None = None,
) -> tuple[bytes, ...] | None:
    """Fetch thumbnails, or None when an image was required and could not be read."""
    cids = image_blob_cids(embed)
    if cids is None:
        return None
    if not cids:
        return ()
    if not author_did:
        return None
    timeout = _timeout_s() if timeout_s is None else timeout_s
    try:
        return tuple(_fetch_one(author_did, cid, timeout_s=timeout) for cid in cids)
    except urllib.error.URLError, TimeoutError, ValueError:
        return None


def _fetch_one(author_did: str, cid: str, *, timeout_s: float) -> bytes:
    request = urllib.request.Request(
        thumbnail_url(author_did, cid),
        headers={'User-Agent': _USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        data = response.read(_MAX_BYTES + 1)
    if not isinstance(data, bytes) or not data or len(data) > _MAX_BYTES:
        raise ValueError('thumbnail missing or oversized')
    if _image_mime(data) is None:
        raise ValueError('thumbnail is not a supported image')
    return data


def _blob_cid(blob: object) -> str | None:
    if not isinstance(blob, dict):
        return None
    ref = blob.get('ref')
    if isinstance(ref, str) and ref:
        return ref
    if isinstance(ref, dict):
        link = ref.get('$link')
        if isinstance(link, str) and link:
            return link
    cid = blob.get('cid')
    if isinstance(cid, str) and cid:
        return cid
    return None


def _image_mime(data: bytes) -> str | None:
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data.startswith((b'GIF87a', b'GIF89a')):
        return 'image/gif'
    if len(data) >= 12 and data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return 'image/webp'
    return None


def _timeout_s() -> float:
    """Read ``DECISION_TIMEOUT_S`` at call time. Invalid values raise ValueError."""
    raw = os.environ.get('DECISION_TIMEOUT_S', '')
    if not raw.strip():
        return 8.0
    return float(raw)
