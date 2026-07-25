import os

from dotenv import load_dotenv

from server.allowlists import load_allowlist_dids, load_allowlist_handles
from server.logger import logger

# Override process env so a shell HOSTNAME cannot beat .env / Fly config.
load_dotenv(override=True)


def _get_bool_env_var(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {'1', 'true', 't', 'yes', 'y'}


# FEEDGEN_HOSTNAME avoids colliding with the OS/shell HOSTNAME variable.
_hostname = os.environ.get('FEEDGEN_HOSTNAME') or os.environ.get('HOSTNAME')
if not _hostname or _hostname == 'cursor':
    raise RuntimeError(
        'Set FEEDGEN_HOSTNAME (or HOSTNAME) to your public feedgen hostname (e.g. [REDACTED]).'
    )
HOSTNAME: str = _hostname

SERVICE_DID = os.environ.get('SERVICE_DID') or f'did:web:{HOSTNAME}'

_feed_uri = os.environ.get('FEED_URI')
if not _feed_uri:
    raise RuntimeError(
        'Set FEED_URI after publishing (or to the existing SkyFeed URI for cutover testing).'
    )
FEED_URI: str = _feed_uri

IGNORE_ARCHIVED_POSTS = _get_bool_env_var(os.environ.get('IGNORE_ARCHIVED_POSTS', 'true'))
IGNORE_REPLY_POSTS = _get_bool_env_var(os.environ.get('IGNORE_REPLY_POSTS', 'true'))
POST_RETENTION_DAYS = int(os.environ.get('POST_RETENTION_DAYS', '7'))
# Soft priors: authors with this many strong text matches in the window may keep
# bare ambiguous place names (never hard negatives). See server/author_priors.py.
SOFT_PRIOR_MIN_STRONG = int(os.environ.get('SOFT_PRIOR_MIN_STRONG', '3'))
SOFT_PRIOR_WINDOW_DAYS = int(os.environ.get('SOFT_PRIOR_WINDOW_DAYS', '30'))
DATABASE_PATH = os.environ.get('DATABASE_PATH', 'feed_database.db')
JETSTREAM_URL = os.environ.get(
    'JETSTREAM_URL',
    'wss://jetstream2.us-east.bsky.network/subscribe',
)
PORT = int(os.environ.get('PORT', '8080'))

ALLOWLIST_DIDS = load_allowlist_dids()
ALLOWLIST_HANDLES = load_allowlist_handles()

logger.info(
    'config loaded hostname=%s service_did=%s allowlist_dids=%d allowlist_handles=%d '
    'soft_prior_min=%d soft_prior_window_days=%d',
    HOSTNAME,
    SERVICE_DID,
    len(ALLOWLIST_DIDS),
    len(ALLOWLIST_HANDLES),
    SOFT_PRIOR_MIN_STRONG,
    SOFT_PRIOR_WINDOW_DAYS,
)
