"""Soft author priors earned from decider outcomes.

Hard allowlists (``allowlist_*.txt``) always reach the candidate floor. Soft
priors are a separate tier: an author with enough recent decider keeps and no
recent decider rejects can send a bare ambiguous place name to the decider.
The prior does not skip the decider, and it does not override hard negatives
or keep arbitrary no-placename posts. Regex-only keeps do not earn a prior.
"""

from __future__ import annotations

from datetime import timedelta

from server import config
from server.database import AuthorDecision, utc_now


def author_has_soft_prior(author_did: str | None) -> bool:
    """True when recent keeps meet the minimum and there are no recent rejects."""
    if not author_did:
        return False
    cutoff = utc_now() - timedelta(days=config.SOFT_PRIOR_WINDOW_DAYS)
    base = AuthorDecision.select().where(
        (AuthorDecision.author_did == author_did) & (AuthorDecision.decided_at >= cutoff)
    )
    if base.where(AuthorDecision.kept == 0).exists():
        return False
    keeps = base.where(AuthorDecision.kept == 1).count()
    return keeps >= config.SOFT_PRIOR_MIN_KEEPS


def record_decision(author_did: str | None, *, kept: bool) -> None:
    """Append one decider outcome. Timeouts and regex-only keeps are not recorded."""
    if not author_did:
        return
    AuthorDecision.create(author_did=author_did, kept=kept, decided_at=utc_now())
