"""Soft author priors earned from repeated strong local text matches.

Hard allowlists (``allowlist_*.txt``) always-keep curated accounts. Soft priors
are a separate tier: authors who repeatedly match via strong text cues can keep
ambiguous bare place-name posts, without unlocking hard negatives or every
no-placename post.
"""

from __future__ import annotations

from datetime import timedelta

from server import config
from server.database import AuthorLocalStats, db, utc_now

# Text-earned locality only — not allowlist / soft-prior / classifier reasons
# (avoids loops where second-stage keeps inflate soft priors).
_STRONG_MATCH_REASONS = frozenset(
    {
        'strong_positive',
        'strong_positive_over_negative',
        'colonie_local',
        'multi_local_places',
        'albany_with_ny_context',
        'albany_with_local_cue',
        'colonie_with_context',
    }
)


def is_strong_match_reason(reason: str | None) -> bool:
    if not reason:
        return False
    if reason in _STRONG_MATCH_REASONS:
        return True
    return reason.startswith('ambiguous_with_context:') or reason.startswith('event_local_venue:')


def author_has_soft_prior(author_did: str | None) -> bool:
    """True when the author has enough recent strong text matches."""
    if not author_did:
        return False
    row = AuthorLocalStats.get_or_none(AuthorLocalStats.author_did == author_did)
    if row is None or row.last_strong_at is None:
        return False
    window = timedelta(days=config.SOFT_PRIOR_WINDOW_DAYS)
    if utc_now() - row.last_strong_at > window:
        return False
    return row.strong_match_count >= config.SOFT_PRIOR_MIN_STRONG


def record_strong_match(author_did: str | None) -> None:
    """Increment (or reset) the author's strong-match counter in the window."""
    if not author_did:
        return
    now = utc_now()
    window = timedelta(days=config.SOFT_PRIOR_WINDOW_DAYS)
    with db.atomic():
        row, created = AuthorLocalStats.get_or_create(
            author_did=author_did,
            defaults={'strong_match_count': 1, 'last_strong_at': now},
        )
        if created:
            return
        if row.last_strong_at is None or now - row.last_strong_at > window:
            row.strong_match_count = 1
        else:
            row.strong_match_count += 1
        row.last_strong_at = now
        row.save()
