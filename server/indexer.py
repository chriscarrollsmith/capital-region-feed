"""Apply matcher rules to Jetstream post events and update the DB."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from server import config
from server.database import Post, db, prune_old_posts
from server.logger import logger
from server.matcher import extract_alt_text, match_post

_last_prune = datetime.utcnow()


def _parse_created_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # Bluesky timestamps are ISO-8601; normalize Z.
        normalized = value.replace('Z', '+00:00')
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        return None


def _is_archived(created_at: datetime | None) -> bool:
    if not created_at:
        return False
    return datetime.utcnow() - created_at > timedelta(days=1)


def _maybe_prune() -> None:
    global _last_prune
    now = datetime.utcnow()
    if now - _last_prune < timedelta(minutes=30):
        return
    deleted = prune_old_posts()
    _last_prune = now
    if deleted:
        logger.info('pruned %s old posts', deleted)


def handle_event(event: dict[str, Any]) -> None:
    operation = event.get('operation')
    uri = event.get('uri')
    if not uri or not operation:
        return

    if operation == 'delete':
        deleted = Post.delete().where(Post.uri == uri).execute()
        if deleted:
            logger.debug('deleted %s', uri)
        return

    if operation != 'create':
        return

    record = event.get('record') or {}
    if config.IGNORE_REPLY_POSTS and record.get('reply'):
        return

    created_at = _parse_created_at(record.get('createdAt'))
    if config.IGNORE_ARCHIVED_POSTS and _is_archived(created_at):
        return

    text = record.get('text') or ''
    alt_text = extract_alt_text(record.get('embed'))
    result = match_post(
        text,
        alt_text=alt_text,
        author_did=event.get('author'),
        allowlist_dids=config.ALLOWLIST_DIDS,
        allowlist_handles=config.ALLOWLIST_HANDLES,
    )
    if not result.matched:
        return

    reply = record.get('reply') or {}
    parent = (reply.get('parent') or {}).get('uri')
    root = (reply.get('root') or {}).get('uri')

    with db.atomic():
        Post.insert(
            uri=uri,
            cid=event.get('cid') or '',
            author_did=event.get('author'),
            reply_parent=parent,
            reply_root=root,
            match_reason=result.reason,
            created_at=created_at,
        ).on_conflict_ignore().execute()

    logger.info('indexed %s reason=%s', uri, result.reason)
    _maybe_prune()
