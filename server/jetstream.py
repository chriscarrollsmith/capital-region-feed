"""Jetstream JSON subscriber for posts plus like/repost engagement."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlencode

import websockets

from server import config
from server.database import SubscriptionState, db
from server.logger import logger

OnPostEvent = Callable[[dict[str, Any]], None]

_WANTED_COLLECTIONS = (
    'app.bsky.feed.post',
    'app.bsky.feed.like',
    'app.bsky.feed.repost',
)


def _build_url(cursor: int | None = None) -> str:
    params = [('wantedCollections', collection) for collection in _WANTED_COLLECTIONS]
    if cursor is not None and cursor > 0:
        params.append(('cursor', str(cursor)))
    return f'{config.JETSTREAM_URL}?{urlencode(params)}'


def _get_cursor(service_name: str) -> int | None:
    state = SubscriptionState.get_or_none(SubscriptionState.service == service_name)
    if state is None:
        SubscriptionState.create(service=service_name, cursor=0)
        return None
    return int(state.cursor) or None


def _save_cursor(service_name: str, cursor: int) -> None:
    SubscriptionState.update(cursor=cursor).where(
        SubscriptionState.service == service_name
    ).execute()


def _parse_post_event(message: dict[str, Any]) -> dict[str, Any] | None:
    if message.get('kind') != 'commit':
        return None

    commit = message.get('commit') or {}
    collection = commit.get('collection')
    did = message.get('did')
    rkey = commit.get('rkey')
    operation = commit.get('operation')
    if not did or not rkey or not operation or not collection:
        return None

    if collection == 'app.bsky.feed.post':
        uri = f'at://{did}/app.bsky.feed.post/{rkey}'
        event: dict[str, Any] = {
            'operation': operation,
            'uri': uri,
            'cid': commit.get('cid'),
            'author': did,
            'time_us': message.get('time_us'),
        }
        if operation == 'create':
            event['record'] = commit.get('record') or {}
        return event

    if collection in {'app.bsky.feed.like', 'app.bsky.feed.repost'}:
        if operation != 'create':
            return None
        record = commit.get('record') or {}
        subject = record.get('subject') or {}
        subject_uri = subject.get('uri')
        if not subject_uri:
            return None
        kind = 'like' if collection.endswith('.like') else 'repost'
        return {
            'operation': operation,
            'engagement': kind,
            'subject_uri': subject_uri,
            'author': did,
            'uri': f'at://{did}/{collection}/{rkey}',
            'time_us': message.get('time_us'),
        }

    return None


async def _consume(
    service_name: str, on_event: OnPostEvent, stop_event: asyncio.Event | None
) -> None:
    backoff = 1
    while stop_event is None or not stop_event.is_set():
        cursor = _get_cursor(service_name)
        url = _build_url(cursor)
        logger.info('connecting to jetstream cursor=%s', cursor)
        try:
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=20, max_size=8_000_000
            ) as ws:
                backoff = 1
                last_persist = time.monotonic()
                latest_cursor = cursor or 0
                async for raw in ws:
                    if stop_event is not None and stop_event.is_set():
                        break
                    try:
                        message = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    time_us = message.get('time_us')
                    if isinstance(time_us, int):
                        latest_cursor = time_us

                    event = _parse_post_event(message)
                    if event is not None:
                        on_event(event)

                    now = time.monotonic()
                    if latest_cursor and now - last_persist >= 5:
                        _save_cursor(service_name, latest_cursor)
                        last_persist = now
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reconnect loop
            logger.warning('jetstream disconnected: %s; retrying in %ss', exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


def run(
    service_name: str,
    on_event: OnPostEvent,
    stop_event: threading.Event | None = None,
) -> None:
    """Blocking entrypoint for a background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async_stop = asyncio.Event()

    def _bridge_stop() -> None:
        if stop_event is None:
            return
        while not stop_event.is_set():
            time.sleep(0.25)
        loop.call_soon_threadsafe(async_stop.set)

    if stop_event is not None:
        threading.Thread(target=_bridge_stop, daemon=True).start()

    try:
        loop.run_until_complete(_consume(service_name, on_event, async_stop))
    finally:
        db.close()
        loop.close()
