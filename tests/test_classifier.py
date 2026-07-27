"""Unit tests for the ambiguous-case linear classifier."""

from __future__ import annotations

from server.classifier import (
    ClassifierModel,
    classify_candidate,
    extract_features,
    load_model,
)
from server.matcher import match_post


def test_load_default_model() -> None:
    model = load_model()
    assert model.version == 'ambiguous_clf_v1'
    assert 'local_micro' in model.weights


def test_extract_features_event_and_micro() -> None:
    feats = extract_features(
        'Pine Hills block party this Saturday',
        term=None,
        has_event_cue=True,
        has_local_venue=False,
    )
    assert feats['local_micro'] == 1.0
    assert feats['event_and_micro'] == 1.0
    assert feats['albany_event_no_micro'] == 0.0


def test_classify_keeps_local_micro_event() -> None:
    decision = classify_candidate(
        'Pine Hills block party this Saturday — bring a dish!',
        term=None,
        has_event_cue=True,
        has_local_venue=False,
    )
    assert decision is not None
    assert decision.matched is True
    assert decision.reason == 'classifier:local_micro'


def test_collision_micros_need_cap_region_hint() -> None:
    """Central Ave / Lincoln Park / 4th Street alone must not keep off-region posts."""
    assert (
        match_post(
            'Forum Thursday August 6, 2026, 7:00pm at 274 Central Avenue, Hackensack, NJ'
        ).matched
        is False
    )
    assert match_post('TONIGHT karaoke at Lincoln Park, Chicago, IL 8pm').matched is False
    assert (
        match_post('Gallery at 3704 East 34th Street Minneapolis Hours Friday 10 am').matched
        is False
    )
    assert match_post('Assault on 14th Street NW July 24 at 10:13 PM').matched is False
    assert (
        match_post('7th Avenue between West 13th Street and West 14th Street. #doors').matched
        is False
    )

    # Collision micro still unlocks with an Albany / Cap Region hint.
    keep = match_post('Crash on Central Avenue in Albany this morning around 8:15am.')
    assert keep.matched is True
    assert keep.reason.startswith('classifier:')


def test_center_square_wire_byline_is_not_local_micro() -> None:
    """The Center Square news wire must not unlock November/event + micro keeps."""
    alt = (
        'GOP makes political push in House races amid lawmaker controversies '
        '(The Center Square) – Months out from the upcoming general election in '
        'November, a number of races for the Illinois House of Representatives '
        'are gaining momentum'
    )
    result = match_post(
        'GOP makes political push in House races amid lawmaker controversies',
        alt_text=alt,
    )
    assert result.matched is False

    # Real Albany Center Square neighborhood + event cue still keeps.
    porch = match_post('Center Square Porchfest is this weekend. Maps at noon.')
    assert porch.matched is True
    assert porch.reason == 'classifier:local_micro'


def test_classify_drops_bare_albany_event_without_micro() -> None:
    decision = classify_candidate(
        "Don't miss the Albany Veterans Day Parade this Saturday downtown!",
        term='albany',
        has_event_cue=True,
        has_local_venue=False,
    )
    assert decision is None


def test_classify_drops_venue_without_event() -> None:
    decision = classify_candidate(
        'Proctors is a beautiful historic building downtown.',
        term=None,
        has_event_cue=False,
        has_local_venue=True,
    )
    assert decision is None


def test_match_post_classifier_reason_for_lark_street() -> None:
    result = match_post('Show on Lark Street in Albany tonight. Doors at 8.')
    assert result.matched is True
    assert result.reason == 'classifier:ambiguous:albany'


def test_match_post_soft_prior_still_beats_classifier_path() -> None:
    did = 'did:plc:softpriortest0000000000001'
    result = match_post(
        'Dinner in Troy tonight.',
        author_did=did,
        soft_prior_dids={did},
    )
    assert result.matched is True
    assert result.reason == 'soft_prior_ambiguous:troy'


def test_match_post_hard_negative_never_reaches_classifier() -> None:
    result = match_post('Nice day in Albany Park near Lark Street tonight.')
    assert result.matched is False
    assert result.reason in {'hard_negative', 'entity_other:albany_park_chicago'}


def test_injected_model_can_force_drop() -> None:
    model = ClassifierModel(version='test', threshold=99.0, weights={'bias': 0.0})
    result = match_post(
        'Pine Hills block party this Saturday — bring a dish!',
        classifier_model=model,
    )
    assert result.matched is False
    assert result.reason == 'no_match'
