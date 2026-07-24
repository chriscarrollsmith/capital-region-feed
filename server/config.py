import os
from pathlib import Path

from dotenv import load_dotenv

from server.logger import logger

# Override process env so a shell HOSTNAME cannot beat .env / Fly config.
load_dotenv(override=True)


def _get_bool_env_var(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {'1', 'true', 't', 'yes', 'y'}


def _load_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()
    lines: set[str] = set()
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        lines.add(line)
    return lines


# FEEDGEN_HOSTNAME avoids colliding with the OS/shell HOSTNAME variable.
HOSTNAME = os.environ.get('FEEDGEN_HOSTNAME') or os.environ.get('HOSTNAME')
if not HOSTNAME or HOSTNAME == 'cursor':
    raise RuntimeError(
        'Set FEEDGEN_HOSTNAME (or HOSTNAME) to your public feedgen hostname '
        '(e.g. capital-region-feed.fly.dev).'
    )

SERVICE_DID = os.environ.get('SERVICE_DID') or f'did:web:{HOSTNAME}'

FEED_URI = os.environ.get('FEED_URI')
if not FEED_URI:
    raise RuntimeError(
        'Set FEED_URI after publishing (or to the existing SkyFeed URI for cutover testing).'
    )

IGNORE_ARCHIVED_POSTS = _get_bool_env_var(os.environ.get('IGNORE_ARCHIVED_POSTS', 'true'))
IGNORE_REPLY_POSTS = _get_bool_env_var(os.environ.get('IGNORE_REPLY_POSTS', 'true'))
POST_RETENTION_DAYS = int(os.environ.get('POST_RETENTION_DAYS', '7'))
DATABASE_PATH = os.environ.get('DATABASE_PATH', 'feed_database.db')
JETSTREAM_URL = os.environ.get(
    'JETSTREAM_URL',
    'wss://jetstream2.us-east.bsky.network/subscribe',
)
PORT = int(os.environ.get('PORT', '8080'))

_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
ALLOWLIST_DIDS = _load_lines(_DATA_DIR / 'allowlist_dids.txt')
ALLOWLIST_HANDLES = {h.lower() for h in _load_lines(_DATA_DIR / 'allowlist_handles.txt')}

logger.info(
    'config loaded hostname=%s service_did=%s allowlist_dids=%d allowlist_handles=%d',
    HOSTNAME,
    SERVICE_DID,
    len(ALLOWLIST_DIDS),
    len(ALLOWLIST_HANDLES),
)
