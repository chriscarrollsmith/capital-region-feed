"""Soft author priors and the resident tier, both earned from decider outcomes.

Hard allowlists (``allowlist_*.txt``) always reach the candidate floor. Soft
priors are a separate tier: an author with enough recent mention-lane decider
keeps and no recent mention-lane rejects can send a bare ambiguous place name
to the decider. The prior does not skip the decider, and it does not override
hard negatives or keep posts with no place name.

The resident tier uses that same mention-lane evidence. Once earned, posts the
floor would otherwise drop for lack of a place name become candidates for a
separate decider prompt that asks whether the post is local life. The decider
stays authoritative, and the path stays closed when the decider is off.
Resident-lane rejects do not revoke the soft prior. A high resident-lane reject
rate, or too many resident-lane decisions in a day, suspends the privilege.
Mention-lane rejects revoke both. Regex-only keeps earn neither.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import peewee

from server import config
from server.database import AuthorDecision, utc_now
from server.decider import DECISION_KEEP_PREFIX, DECISION_REJECT_PREFIX

LANE_MENTION = 'mention'
LANE_RESIDENT = 'resident'
_LANES = frozenset({LANE_MENTION, LANE_RESIDENT})


def _mention_decisions(author_did: str, cutoff: datetime) -> peewee.ModelSelect[AuthorDecision]:
    return AuthorDecision.select().where(
        (AuthorDecision.author_did == author_did)
        & (AuthorDecision.decided_at >= cutoff)
        & ((AuthorDecision.lane == LANE_MENTION) | AuthorDecision.lane.is_null())
    )


def author_has_soft_prior(author_did: str | None) -> bool:
    """True when recent mention-lane keeps meet the minimum and none were rejects."""
    if not author_did:
        return False
    cutoff = utc_now() - timedelta(days=config.SOFT_PRIOR_WINDOW_DAYS)
    base = _mention_decisions(author_did, cutoff)
    if base.where(AuthorDecision.kept == 0).exists():
        return False
    keeps = base.where(AuthorDecision.kept == 1).count()
    return keeps >= config.SOFT_PRIOR_MIN_KEEPS


def author_is_resident(author_did: str | None) -> bool:
    """True when mention-lane evidence earns a no-placename decider pass.

    Resident-lane rejects suspend the privilege once they outnumber resident
    keeps and reach ``RESIDENT_DEMOTE_REJECTS``. More than
    ``RESIDENT_MAX_PER_DAY`` resident decisions in the last day also suspends
    it, until that day rolls off. Mention-lane rejects revoke the privilege
    outright.
    """
    if not author_did:
        return False
    cutoff = utc_now() - timedelta(days=config.SOFT_PRIOR_WINDOW_DAYS)
    mention = _mention_decisions(author_did, cutoff)
    if mention.where(AuthorDecision.kept == 0).exists():
        return False
    mention_keeps = mention.where(AuthorDecision.kept == 1).count()
    if mention_keeps < config.RESIDENT_MIN_KEEPS:
        return False

    resident = AuthorDecision.select().where(
        (AuthorDecision.author_did == author_did)
        & (AuthorDecision.decided_at >= cutoff)
        & (AuthorDecision.lane == LANE_RESIDENT)
    )
    resident_keeps = resident.where(AuthorDecision.kept == 1).count()
    resident_rejects = resident.where(AuthorDecision.kept == 0).count()
    if resident_rejects >= config.RESIDENT_DEMOTE_REJECTS and resident_rejects > resident_keeps:
        return False

    day_ago = utc_now() - timedelta(days=1)
    recent = (
        AuthorDecision.select()
        .where(
            (AuthorDecision.author_did == author_did)
            & (AuthorDecision.decided_at >= day_ago)
            & (AuthorDecision.lane == LANE_RESIDENT)
        )
        .count()
    )
    return recent < config.RESIDENT_MAX_PER_DAY


def record_decision(
    author_did: str | None,
    *,
    kept: bool,
    lane: str = LANE_MENTION,
) -> None:
    """Append one decider outcome. Timeouts and regex-only keeps are not recorded."""
    if not author_did:
        return
    if lane not in _LANES:
        raise ValueError(f'unknown decision lane {lane!r}')
    AuthorDecision.create(
        author_did=author_did,
        kept=kept,
        decided_at=utc_now(),
        lane=lane,
    )


def lane_for_reason(reason: str) -> str:
    """Map a decider match reason to the lane that should be recorded."""
    body = reason
    for prefix in (DECISION_KEEP_PREFIX, DECISION_REJECT_PREFIX):
        if reason.startswith(prefix):
            body = reason[len(prefix) :]
            break
    provisional = body.rsplit(':', 1)[0]
    if provisional == 'resident':
        return LANE_RESIDENT
    return LANE_MENTION
