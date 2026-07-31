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

# When replaying more than this far behind live, skip like/repost handling so
# catch-up stays on the asyncio event loop's happy path (websocket pings).
_ENGAGEMENT_CATCHUP_LAG_US = 120 * 1_000_000

# websockets keepalive: pings need the event loop free. Sync SQLite/matcher work
# must not run on the loop or ping_timeout fires during firehose catch-up.
_PING_INTERVAL_S = 20
_PING_TIMEOUT_S = 60


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


def cursor_lag_seconds(
    cursor: int | None = None,
    *,
    service_name: str | None = None,
    now_us: int | None = None,
) -> float | None:
    """Return how far the saved (or given) cursor is behind wall clock, in seconds."""
    if cursor is None:
        query = SubscriptionState.select()
        if service_name is not None:
            query = query.where(SubscriptionState.service == service_name)
        state = query.get_or_none()
        if state is None:
            return None
        cursor = int(state.cursor) or None
    if not cursor:
        return None
    now = int(time.time() * 1_000_000) if now_us is None else now_us
    return max(0.0, (now - cursor) / 1_000_000)


def _parse_post_event(
    message: dict[str, Any],
    *,
    skip_engagement: bool = False,
) -> dict[str, Any] | None:
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
        if skip_engagement or operation != 'create':
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
    loop = asyncio.get_running_loop()
    while stop_event is None or not stop_event.is_set():
        cursor = _get_cursor(service_name)
        url = _build_url(cursor)
        lag_s = cursor_lag_seconds(cursor)
        logger.info(
            'connecting to jetstream cursor=%s lag_s=%s',
            cursor,
            None if lag_s is None else round(lag_s, 1),
        )
        try:
            async with websockets.connect(
                url,
                ping_interval=_PING_INTERVAL_S,
                ping_timeout=_PING_TIMEOUT_S,
                max_size=8_000_000,
            ) as ws:
                backoff = 1
                last_persist = time.monotonic()
                latest_cursor = cursor or 0
                skipping_engagement = False
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
                        # Compare stream time to wall clock — deep catch-up floods
                        # likes/reposts and must not starve websocket keepalive.
                        lag_us = int(time.time() * 1_000_000) - time_us
                        want_skip = lag_us > _ENGAGEMENT_CATCHUP_LAG_US
                        if want_skip != skipping_engagement:
                            skipping_engagement = want_skip
                            logger.info(
                                'jetstream engagement %s (lag_s=%.1f)',
                                'paused' if want_skip else 'resumed',
                                lag_us / 1_000_000,
                            )

                    event = _parse_post_event(message, skip_engagement=skipping_engagement)
                    if event is not None:
                        # Keep ping/pong on the event loop; Peewee + matcher are sync.
                        await loop.run_in_executor(None, on_event, event)

                    now = time.monotonic()
                    if latest_cursor and now - last_persist >= 5:
                        await loop.run_in_executor(None, _save_cursor, service_name, latest_cursor)
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
