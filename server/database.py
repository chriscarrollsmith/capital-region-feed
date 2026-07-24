from datetime import UTC, datetime, timedelta
from pathlib import Path

import peewee

from server import config


def utc_now() -> datetime:
    """Naive UTC timestamp compatible with existing SQLite rows."""
    return datetime.now(UTC).replace(tzinfo=None)


_db_path = Path(config.DATABASE_PATH)
_db_path.parent.mkdir(parents=True, exist_ok=True)

db = peewee.SqliteDatabase(
    str(_db_path),
    pragmas={
        'journal_mode': 'wal',
        'synchronous': 'normal',
    },
)


class BaseModel(peewee.Model):
    class Meta:
        database = db


class Post(BaseModel):
    uri = peewee.CharField(unique=True, index=True)
    cid = peewee.CharField()
    author_did = peewee.CharField(null=True, index=True)
    reply_parent = peewee.CharField(null=True, default=None)
    reply_root = peewee.CharField(null=True, default=None)
    match_reason = peewee.CharField(null=True, default=None)
    created_at = peewee.DateTimeField(null=True, index=True)
    indexed_at = peewee.DateTimeField(default=utc_now, index=True)


class SubscriptionState(BaseModel):
    service = peewee.CharField(unique=True)
    cursor = peewee.BigIntegerField()


if db.is_closed():
    db.connect()
    db.create_tables([Post, SubscriptionState])


def prune_old_posts(retention_days: int | None = None) -> int:
    days = config.POST_RETENTION_DAYS if retention_days is None else retention_days
    cutoff = utc_now() - timedelta(days=days)
    query = Post.delete().where(Post.indexed_at < cutoff)
    return query.execute()
