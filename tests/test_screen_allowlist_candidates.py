"""Unit tests for allowlist candidate screening heuristics."""

from __future__ import annotations

from scripts.screen_allowlist_candidates import collect_flags, recommend


def test_recommend_rejects_high_volume_and_bots() -> None:
    assert recommend(flags=['high_volume:40.0/d']) == 'reject'
    assert recommend(flags=['bot_or_auto']) == 'reject'
    assert recommend(flags=['fetch_error']) == 'reject'


def test_recommend_review_for_elevated_volume_or_slop() -> None:
    assert recommend(flags=['elevated_volume:9.0/d']) == 'review'
    assert recommend(flags=['slop:promo']) == 'review'


def test_recommend_skip_empty_and_likely_ok() -> None:
    assert recommend(flags=['inactive_or_empty']) == 'skip_empty'
    assert recommend(flags=[]) == 'likely_ok'
    assert recommend(flags=['low_diversity:0.3']) == 'review'


def test_collect_flags_detects_volume_bot_and_slop() -> None:
    flags, slop = collect_flags(
        handle='alerts-bot.example',
        display_name='NWS Bot',
        description='Unofficial bot. Not monitored.',
        ppd_7d=25.0,
        ppd_30d=22.0,
        unique_prefix_ratio=0.2,
        sample_n=40,
        sample_texts=['Call us today for a free estimate serving Albany!'],
    )
    assert 'bot_or_auto' in flags
    assert any(f.startswith('high_volume') for f in flags)
    assert any(f.startswith('slop:') for f in flags)
    assert 'bizspam' in slop
    assert any(f.startswith('low_diversity') for f in flags)


def test_collect_flags_clean_account() -> None:
    flags, slop = collect_flags(
        handle='dgazette.bsky.social',
        display_name='The Daily Gazette',
        description='Family-owned newspaper in Schenectady',
        ppd_7d=0.6,
        ppd_30d=0.7,
        unique_prefix_ratio=0.95,
        sample_n=20,
        sample_texts=['City council debates housing policy tonight.'],
    )
    assert flags == []
    assert slop == []
