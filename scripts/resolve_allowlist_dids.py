#!/usr/bin/env python3
"""Resolve ``data/allowlist_handles.txt`` into ``data/allowlist_dids.txt``.

Jetstream only supplies author DIDs, so the live indexer needs DID membership.
Handles stay the curated source of truth; re-run this after editing handles.

    uv run python scripts/resolve_allowlist_dids.py
    uv run python scripts/resolve_allowlist_dids.py --check   # exit 1 if stale
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.allowlists import DIDS_PATH, HANDLES_PATH, load_list_file  # noqa: E402

DEFAULT_API_HOST = 'https://api.bsky.app'
USER_AGENT = (
    'capital-region-feed-resolve/0.1 (+https://github.com/chriscarrollsmith/capital-region-feed)'
)


def resolve_handle(handle: str, *, api_host: str, timeout: float = 30.0) -> str:
    """Return the DID for a Bluesky handle via AppView getProfile."""
    url = (
        api_host.rstrip('/')
        + '/xrpc/app.bsky.actor.getProfile?'
        + urllib.parse.urlencode({'actor': handle})
    )
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode('utf-8'))
    did = payload.get('did')
    if not isinstance(did, str) or not did.startswith('did:'):
        raise ValueError(f'no DID in profile for {handle!r}')
    return did


def build_did_file(
    entries: list[tuple[str, str]],
    *,
    kind: str = 'allowlist',
    handles_name: str = 'allowlist_handles.txt',
    resolve_hint: str | None = None,
) -> str:
    hint = resolve_hint or 'uv run python scripts/resolve_allowlist_dids.py'
    lines = [
        f'# DID {kind} for Jetstream ingest (author field is DID-only).',
        f'# Generated from {handles_name} — re-run:',
        f'#   {hint}',
        '# Do not hand-edit DIDs unless a handle is unstable; prefer updating handles.',
        '',
    ]
    for handle, did in entries:
        lines.append(f'# {handle}')
        lines.append(did)
    lines.append('')
    return '\n'.join(lines)


def resolve_all(
    handles: list[str],
    *,
    api_host: str,
) -> tuple[list[tuple[str, str]], list[str]]:
    resolved: list[tuple[str, str]] = []
    errors: list[str] = []
    for handle in handles:
        try:
            did = resolve_handle(handle, api_host=api_host)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            errors.append(f'{handle}: {exc}')
            continue
        resolved.append((handle, did))
        print(f'{handle} -> {did}', file=sys.stderr)
    return resolved, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--handles',
        type=Path,
        default=HANDLES_PATH,
        help='Handle allowlist path (default: data/allowlist_handles.txt)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DIDS_PATH,
        help='DID allowlist path to write (default: data/allowlist_dids.txt)',
    )
    parser.add_argument(
        '--api-host',
        default=DEFAULT_API_HOST,
        help=f'Bluesky AppView host (default: {DEFAULT_API_HOST})',
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Do not write; exit 1 if resolved DIDs differ from the output file',
    )
    args = parser.parse_args()

    handles = load_list_file(args.handles)
    if not handles:
        print(f'no handles found in {args.handles}', file=sys.stderr)
        return 1

    resolved, errors = resolve_all(handles, api_host=args.api_host)
    if errors:
        for line in errors:
            print(f'error: {line}', file=sys.stderr)
        return 1
    if not resolved:
        print('no DIDs resolved', file=sys.stderr)
        return 1

    handles_name = args.handles.name
    kind = 'blocklist' if 'blocklist' in handles_name else 'allowlist'
    resolve_hint = (
        'uv run python scripts/resolve_allowlist_dids.py '
        f'--handles {args.handles} --output {args.output}'
        if kind == 'blocklist'
        else 'uv run python scripts/resolve_allowlist_dids.py'
    )
    content = build_did_file(
        resolved,
        kind=kind,
        handles_name=handles_name,
        resolve_hint=resolve_hint,
    )
    if args.check:
        current = args.output.read_text(encoding='utf-8') if args.output.is_file() else ''
        if current != content:
            print(f'{args.output} is stale; run without --check to refresh', file=sys.stderr)
            return 1
        print(f'{args.output} is up to date', file=sys.stderr)
        return 0

    args.output.write_text(content, encoding='utf-8')
    print(f'wrote {len(resolved)} DIDs to {args.output}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
