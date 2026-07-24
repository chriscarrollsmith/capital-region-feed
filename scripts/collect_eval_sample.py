#!/usr/bin/env python3
"""Fetch a live Bluesky feed and print candidate eval rows as JSON lines."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--feed',
        default=os.environ.get('FEED_URI'),
        help='Feed URI to sample (default: FEED_URI env)',
    )
    parser.add_argument('--limit', type=int, default=50)
    args = parser.parse_args()

    if not args.feed:
        print('Pass --feed or set FEED_URI', file=sys.stderr)
        return 1

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
