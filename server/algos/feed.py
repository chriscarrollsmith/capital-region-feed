from datetime import datetime
from typing import Optional

from server import config
from server.database import Post

uri = config.FEED_URI
CURSOR_EOF = 'eof'


def handler(cursor: Optional[str], limit: int) -> dict:
    limit = max(1, min(limit, 100))
    posts = (
        Post.select()
        .order_by(Post.indexed_at.desc(), Post.cid.desc())
        .limit(limit)
    )

    if cursor:
        if cursor == CURSOR_EOF:
            return {'cursor': CURSOR_EOF, 'feed': []}
        cursor_parts = cursor.split('::')
        if len(cursor_parts) != 2:
            raise ValueError('Malformed cursor')
        indexed_at_ms, cid = cursor_parts
        indexed_at = datetime.fromtimestamp(int(indexed_at_ms) / 1000)
        posts = posts.where(
            ((Post.indexed_at == indexed_at) & (Post.cid < cid))
            | (Post.indexed_at < indexed_at)
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
