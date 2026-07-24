#!/usr/bin/env python3
"""Fetch the live SkyFeed Albany feed and print candidate eval rows as JSON lines."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request

DEFAULT_FEED = 'at://did:plc:xndplob7sicvv6balxdzh3jk/app.bsky.feed.generator/aaagkkw3yejuk'  # pragma: allowlist secret


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feed', default=DEFAULT_FEED)
    parser.add_argument('--limit', type=int, default=50)
    args = parser.parse_args()

    url = 'https://public.api.bsky.app/xrpc/app.bsky.feed.getFeed?' + urllib.parse.urlencode(
        {'feed': args.feed, 'limit': args.limit}
    )
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read().decode('utf-8'))

    for item in payload.get('feed', []):
        post = item.get('post') or {}
        record = post.get('record') or {}
        author = post.get('author') or {}
        row = {
            'id': post.get('uri', ''),
            'text': record.get('text', ''),
            'author_handle': author.get('handle'),
            'author_did': author.get('did'),
            'expected': None,
            'note': 'label me: true=keep, false=drop',
        }
        print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
