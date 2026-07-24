#!/usr/bin/env python3
"""Publish or update the Bluesky feed generator record.

To cut over the existing SkyFeed Albany feed, keep RECORD_NAME=aaagkkw3yejuk
and set HOSTNAME/SERVICE_DID to your Fly app. putRecord overwrites the same
rkey, so the public feed URI stays stable for subscribers.
"""

from __future__ import annotations

import os
import sys

from atproto import Client, models
from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 't', 'yes', 'y'}


def main() -> int:
    handle = os.environ.get('HANDLE')
    password = os.environ.get('PASSWORD')
    hostname = os.environ.get('HOSTNAME')
    record_name = os.environ.get('RECORD_NAME', 'aaagkkw3yejuk')
    display_name = os.environ.get('DISPLAY_NAME', 'Albany, NY')
    description = os.environ.get(
        'DESCRIPTION',
        "Posts from and about New York's Capital Region.",
    )
    avatar_path = os.environ.get('AVATAR_PATH')
    service_did = os.environ.get('SERVICE_DID') or (f'did:web:{hostname}' if hostname else None)
    accepts_interactions = _bool_env('ACCEPTS_INTERACTIONS')

    missing = [n for n, v in {
        'HANDLE': handle,
        'PASSWORD': password,
        'HOSTNAME': hostname,
        'SERVICE_DID': service_did,
        'RECORD_NAME': record_name,
        'DISPLAY_NAME': display_name,
    }.items() if not v]
    if missing:
        print(f'Missing required env vars: {", ".join(missing)}', file=sys.stderr)
        return 1

    client = Client()
    client.login(handle, password)

    avatar_blob = None
    if avatar_path:
        with open(avatar_path, 'rb') as handle_fp:
            avatar_blob = client.upload_blob(handle_fp.read()).blob

    # Preserve createdAt when updating an existing record so the feed age stays stable.
    existing_created_at = None
    try:
        existing = client.com.atproto.repo.get_record(
            models.ComAtprotoRepoGetRecord.Params(
                repo=client.me.did,
                collection=models.ids.AppBskyFeedGenerator,
                rkey=record_name,
            )
        )
        value = existing.value
        existing_created_at = getattr(value, 'created_at', None) or (
            value.get('createdAt') if isinstance(value, dict) else None
        )
    except Exception:  # noqa: BLE001 - first publish has no record yet
        existing_created_at = None

    response = client.com.atproto.repo.put_record(
        models.ComAtprotoRepoPutRecord.Data(
            repo=client.me.did,
            collection=models.ids.AppBskyFeedGenerator,
            rkey=record_name,
            record=models.AppBskyFeedGenerator.Record(
                did=service_did,
                display_name=display_name,
                description=description,
                avatar=avatar_blob,
                accepts_interactions=accepts_interactions,
                created_at=existing_created_at or client.get_current_time_iso(),
            ),
        )
    )

    print('Successfully published/updated feed generator record.')
    print(f'Feed URI: {response.uri}')
    print(f'Service DID: {service_did}')
    print(f'Endpoint: https://{hostname}')
    print()
    print('Set FEED_URI to the Feed URI above in Fly secrets / .env, then redeploy.')
    print('After cutover, Bluesky AppView should call your service instead of SkyFeed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
