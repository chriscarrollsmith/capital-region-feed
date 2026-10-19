"""Feed-thumbnail CID resolution. No network."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest
from server.thumbnails import image_blob_cids, load_thumbnails, thumbnail_url

_DID = 'did:plc:exampleauthor'
_CID = 'bafkreihfldlzs7q3ta5n33gfnhnh7jq5tapfug7uq5tkualwrcgqqskncq'
_JPEG = b'\xff\xd8\xff' + b'\x00' * 16


def test_thumbnail_url_keeps_did_colons_and_requests_jpeg() -> None:
    url = thumbnail_url(_DID, _CID)
    assert url == (f'https://cdn.bsky.app/img/feed_thumbnail/plain/{_DID}/{_CID}@jpeg')


def test_image_cids_from_ref_link_string_and_cid_field() -> None:
    embed = {
        '$type': 'app.bsky.embed.images',
        'images': [
            {'image': {'ref': {'$link': 'bafy-link'}}},
            {'image': {'ref': 'bafy-string'}},
            {'image': {'cid': 'bafy-cid'}},
        ],
    }
    assert image_blob_cids(embed) == ['bafy-link', 'bafy-string', 'bafy-cid']


def test_record_with_media_skips_quoted_images() -> None:
    embed = {
        '$type': 'app.bsky.embed.recordWithMedia',
        'record': {'record': {'embed': {'images': [{'image': {'ref': {'$link': 'quoted-cid'}}}]}}},
        'media': {'images': [{'image': {'ref': {'$link': 'media-cid'}}}]},
    }
    assert image_blob_cids(embed) == ['media-cid']


def test_external_thumb_is_included_and_missing_thumb_is_empty() -> None:
    thumb = {'$type': 'app.bsky.embed.external', 'external': {'thumb': {'cid': 'thumb-cid'}}}
    assert image_blob_cids(thumb) == ['thumb-cid']
    card = {'external': {'uri': 'https://example.com', 'title': 'Card'}}
    assert image_blob_cids(card) == []


def test_missing_cid_fails_closed_and_absent_embed_is_empty() -> None:
    assert image_blob_cids({'images': [{'alt': 'no blob'}]}) is None
    assert image_blob_cids({'images': ['not-a-dict']}) is None
    assert image_blob_cids(None) == []
    assert image_blob_cids('nope') == []
    assert image_blob_cids({'text': 'hello'}) == []


def test_image_cap_is_four() -> None:
    images = [{'image': {'ref': {'$link': f'cid-{i}'}}} for i in range(6)]
    assert image_blob_cids({'images': images}) == [f'cid-{i}' for i in range(4)]


def test_load_thumbnails_reads_jpeg_from_feed_thumbnail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self, n: int) -> bytes:
            assert n > len(_JPEG)
            return _JPEG

    def urlopen(request: urllib.request.Request, timeout: float | None = None) -> _Response:
        seen['url'] = request.full_url
        seen['timeout'] = timeout
        seen['ua'] = request.get_header('User-agent')
        return _Response()

    monkeypatch.setattr('server.thumbnails.urllib.request.urlopen', urlopen)
    embed = {'images': [{'image': {'ref': {'$link': _CID}}}]}
    got = load_thumbnails(_DID, embed, timeout_s=3.5)
    assert got == (_JPEG,)
    url = str(seen['url'])
    assert url.endswith('@jpeg')
    assert _DID in url
    assert _CID in url
    assert seen['timeout'] == 3.5
    assert 'capital-region-feed-decider' in str(seen['ua'])


def test_load_thumbnails_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embed = {'images': [{'image': {'ref': {'$link': _CID}}}]}

    def urlopen(request: urllib.request.Request, timeout: float | None = None) -> object:
        del request, timeout
        raise urllib.error.URLError('cdn down')

    monkeypatch.setattr('server.thumbnails.urllib.request.urlopen', urlopen)
    assert load_thumbnails(_DID, embed) is None

    class _Big:
        def __enter__(self) -> _Big:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self, n: int) -> bytes:
            return b'\xff\xd8\xff' + (b'\x00' * n)

    monkeypatch.setattr(
        'server.thumbnails.urllib.request.urlopen',
        lambda request, timeout=None: _Big(),
    )
    assert load_thumbnails(_DID, embed) is None

    class _Text:
        def __enter__(self) -> _Text:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self, n: int) -> bytes:
            del n
            return b'not an image'

    monkeypatch.setattr(
        'server.thumbnails.urllib.request.urlopen',
        lambda request, timeout=None: _Text(),
    )
    assert load_thumbnails(_DID, embed) is None
    assert load_thumbnails(None, embed) is None
    assert load_thumbnails('', embed) is None
    assert load_thumbnails(_DID, None) == ()
    assert load_thumbnails(_DID, {'images': [{'alt': 'broken'}]}) is None
