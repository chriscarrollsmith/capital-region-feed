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
