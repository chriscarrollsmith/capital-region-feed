"""Jetstream JSON subscriber for app.bsky.feed.post."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Optional
from urllib.parse import urlencode

import websockets

from server import config
from server.database import SubscriptionState, db
from server.logger import logger

OnPostEvent = Callable[[dict[str, Any]], None]


def _build_url(cursor: Optional[int] = None) -> str:
    params = [('wantedCollections', 'app.bsky.feed.post')]
    if cursor is not None and cursor > 0:
        params.append(('cursor', str(cursor)))
    return f'{config.JETSTREAM_URL}?{urlencode(params)}'


def _get_cursor(service_name: str) -> Optional[int]:
    state = SubscriptionState.get_or_none(SubscriptionState.service == service_name)
    if state is None:
        SubscriptionState.create(service=service_name, cursor=0)
        return None
    return int(state.cursor) or None


def _save_cursor(service_name: str, cursor: int) -> None:
    SubscriptionState.update(cursor=cursor).where(
        SubscriptionState.service == service_name
    ).execute()


def _parse_post_event(message: dict[str, Any]) -> Optional[dict[str, Any]]:
    if message.get('kind') != 'commit':
        return None

    commit = message.get('commit') or {}
    if commit.get('collection') != 'app.bsky.feed.post':
        return None

    did = message.get('did')
    rkey = commit.get('rkey')
    operation = commit.get('operation')
    if not did or not rkey or not operation:
        return None

    uri = f'at://{did}/app.bsky.feed.post/{rkey}'
    event: dict[str, Any] = {
        'operation': operation,
        'uri': uri,
        'cid': commit.get('cid'),
        'author': did,
        'time_us': message.get('time_us'),
    }

    if operation == 'create':
        record = commit.get('record') or {}
        event['record'] = record
    return event


async def _consume(service_name: str, on_event: OnPostEvent, stop_event: Optional[asyncio.Event]) -> None:
    backoff = 1
    while stop_event is None or not stop_event.is_set():
        cursor = _get_cursor(service_name)
        url = _build_url(cursor)
        logger.info('connecting to jetstream cursor=%s', cursor)
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_size=8_000_000) as ws:
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


def run(service_name: str, on_event: OnPostEvent, stop_event=None) -> None:
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
        import threading

        threading.Thread(target=_bridge_stop, daemon=True).start()

    try:
        loop.run_until_complete(_consume(service_name, on_event, async_stop))
    finally:
        db.close()
        loop.close()
