import os

from dotenv import load_dotenv

from server.allowlists import (
    load_allowlist_dids,
    load_allowlist_handles,
    load_blocklist_dids,
    load_blocklist_handles,
)
from server.decider import DEFAULT_KEEP_MIN, DEFAULT_MODEL, decision_enabled
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
# Soft priors: this many recent decider keeps, and zero rejects, let a bare
# ambiguous place reach the decider. Regex matches do not count.
_prior_min = os.environ.get('SOFT_PRIOR_MIN_KEEPS')
if _prior_min is None:
    _prior_min = os.environ.get('SOFT_PRIOR_MIN_STRONG', '3')
SOFT_PRIOR_MIN_KEEPS = int(_prior_min)
SOFT_PRIOR_WINDOW_DAYS = int(os.environ.get('SOFT_PRIOR_WINDOW_DAYS', '30'))
DATABASE_PATH = os.environ.get('DATABASE_PATH', 'feed_database.db')
JETSTREAM_URL = os.environ.get(
    'JETSTREAM_URL',
    'wss://jetstream2.us-east.bsky.network/subscribe',
)
PORT = int(os.environ.get('PORT', '8080'))
RANKING_MODE = os.environ.get('RANKING_MODE', 'indexed').strip().lower()
if RANKING_MODE not in {'indexed', 'created', 'engagement'}:
    raise RuntimeError(
        f'RANKING_MODE must be indexed, created, or engagement (got {RANKING_MODE!r})'
    )
# Comma-separated substrings; matched posts containing any are not indexed.
# Built-in word-boundary mutes (e.g. ACAB) live in server.content_filters.
MUTED_KEYWORDS = tuple(
    part.strip().lower() for part in os.environ.get('MUTED_KEYWORDS', '').split(',') if part.strip()
)

ALLOWLIST_DIDS = load_allowlist_dids()
ALLOWLIST_HANDLES = load_allowlist_handles()
BLOCKLIST_DIDS = load_blocklist_dids()
BLOCKLIST_HANDLES = load_blocklist_handles()

_decision_model = os.environ.get('DECISION_MODEL', DEFAULT_MODEL).strip() or DEFAULT_MODEL
_decision_keep_raw = os.environ.get('DECISION_KEEP_MIN')
_decision_keep_min = (
    DEFAULT_KEEP_MIN
    if _decision_keep_raw is None or not _decision_keep_raw.strip()
    else float(_decision_keep_raw)
)
_openrouter_key_state = 'set' if os.environ.get('OPENROUTER_API_KEY', '').strip() else 'missing'

logger.info(
    'config loaded hostname=%s service_did=%s allowlist_dids=%d allowlist_handles=%d '
    'blocklist_dids=%d blocklist_handles=%d soft_prior_min_keeps=%d soft_prior_window_days=%d '
    'ranking_mode=%s muted_keywords=%d decision_enabled=%s decision_model=%s '
    'openrouter_api_key=%s decision_keep_min=%s',
    HOSTNAME,
    SERVICE_DID,
    len(ALLOWLIST_DIDS),
    len(ALLOWLIST_HANDLES),
    len(BLOCKLIST_DIDS),
    len(BLOCKLIST_HANDLES),
    SOFT_PRIOR_MIN_KEEPS,
    SOFT_PRIOR_WINDOW_DAYS,
    RANKING_MODE,
    len(MUTED_KEYWORDS),
    decision_enabled(),
    _decision_model,
    _openrouter_key_state,
    _decision_keep_min,
)
