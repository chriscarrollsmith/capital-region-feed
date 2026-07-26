from datetime import datetime
from typing import Any

from server import config
from server.database import Post

uri = config.FEED_URI
CURSOR_EOF = 'eof'


def _engagement_score(post: Post) -> int:
    return int(post.like_count or 0) + 2 * int(post.repost_count or 0)


def handler(cursor: str | None, limit: int) -> dict[str, Any]:
    limit = max(1, min(limit, 100))
    mode = config.RANKING_MODE

    if mode == 'engagement':
        return _handler_engagement(cursor, limit)
    if mode == 'created':
        return _handler_created(cursor, limit)
    return _handler_indexed(cursor, limit)


def _handler_indexed(cursor: str | None, limit: int) -> dict[str, Any]:
    posts = Post.select().order_by(Post.indexed_at.desc(), Post.cid.desc()).limit(limit)

    if cursor:
        if cursor == CURSOR_EOF:
            return {'cursor': CURSOR_EOF, 'feed': []}
        cursor_parts = cursor.split('::')
        if len(cursor_parts) != 2:
            raise ValueError('Malformed cursor')
        indexed_at_ms, cid = cursor_parts
        indexed_at = datetime.fromtimestamp(int(indexed_at_ms) / 1000)
        posts = posts.where(
            ((Post.indexed_at == indexed_at) & (Post.cid < cid)) | (Post.indexed_at < indexed_at)
        )

    post_list = list(posts)
    feed = [{'post': post.uri} for post in post_list]
    next_cursor = CURSOR_EOF
    if post_list:
        last_post = post_list[-1]
        next_cursor = f'{int(last_post.indexed_at.timestamp() * 1000)}::{last_post.cid}'

    return {
        'cursor': next_cursor,
        'feed': feed,
    }


def _handler_created(cursor: str | None, limit: int) -> dict[str, Any]:
    # Prefer author created_at; fall back to indexed_at for legacy/null rows.
    posts = (
        Post.select()
        .order_by(
            Post.created_at.desc(nulls='LAST'),
            Post.indexed_at.desc(),
            Post.cid.desc(),
        )
        .limit(limit)
    )

    if cursor:
        if cursor == CURSOR_EOF:
            return {'cursor': CURSOR_EOF, 'feed': []}
        cursor_parts = cursor.split('::')
        if len(cursor_parts) != 2:
            raise ValueError('Malformed cursor')
        created_at_ms, cid = cursor_parts
        created_at = datetime.fromtimestamp(int(created_at_ms) / 1000)
        posts = posts.where(
            (
                (Post.created_at.is_null(False))
                & (
                    ((Post.created_at == created_at) & (Post.cid < cid))
                    | (Post.created_at < created_at)
                )
            )
            | (
                Post.created_at.is_null(True)
                & (
                    ((Post.indexed_at == created_at) & (Post.cid < cid))
                    | (Post.indexed_at < created_at)
                )
            )
        )

    post_list = list(posts)
    feed = [{'post': post.uri} for post in post_list]
    next_cursor = CURSOR_EOF
    if post_list:
        last_post = post_list[-1]
        stamp = last_post.created_at or last_post.indexed_at
        next_cursor = f'{int(stamp.timestamp() * 1000)}::{last_post.cid}'

    return {
        'cursor': next_cursor,
        'feed': feed,
    }


def _handler_engagement(cursor: str | None, limit: int) -> dict[str, Any]:
    # SQLite expression: likes + 2 * reposts, then recency, then cid.
    score_expr = (Post.like_count + 2 * Post.repost_count).alias('engagement_score')
    posts = (
        Post.select(Post, score_expr)
        .order_by(
            (Post.like_count + 2 * Post.repost_count).desc(),
            Post.indexed_at.desc(),
            Post.cid.desc(),
        )
        .limit(limit)
    )

    if cursor:
        if cursor == CURSOR_EOF:
            return {'cursor': CURSOR_EOF, 'feed': []}
        cursor_parts = cursor.split('::')
        if len(cursor_parts) != 3:
            raise ValueError('Malformed cursor')
        score_s, indexed_at_ms, cid = cursor_parts
        score = int(score_s)
        indexed_at = datetime.fromtimestamp(int(indexed_at_ms) / 1000)
        eng = Post.like_count + 2 * Post.repost_count
        posts = posts.where(
            (eng < score)
            | ((eng == score) & (Post.indexed_at == indexed_at) & (Post.cid < cid))
            | ((eng == score) & (Post.indexed_at < indexed_at))
        )

    post_list = list(posts)
    feed = [{'post': post.uri} for post in post_list]
    next_cursor = CURSOR_EOF
    if post_list:
        last_post = post_list[-1]
        next_cursor = (
            f'{_engagement_score(last_post)}::'
            f'{int(last_post.indexed_at.timestamp() * 1000)}::'
            f'{last_post.cid}'
        )

    return {
        'cursor': next_cursor,
        'feed': feed,
    }
