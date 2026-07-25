"""Unit tests for stratified eval reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.eval_filter import Confusion, evaluate_cases

CASES_PATH = Path(__file__).resolve().parents[1] / 'data' / 'eval_cases.json'


def test_confusion_metrics() -> None:
    c = Confusion()
    c.add(predicted=True, expected=True)
    c.add(predicted=True, expected=False)
    c.add(predicted=False, expected=False)
    c.add(predicted=False, expected=True)
    assert c.tp == 1 and c.fp == 1 and c.tn == 1 and c.fn == 1
    assert c.precision == 0.5
    assert c.recall == 0.5
    assert c.f1 == 0.5
    assert c.accuracy == 0.5


def test_evaluate_cases_stratifies_and_respects_gaps() -> None:
    cases = [
        {
            'id': 'tp-text',
            'text': 'Hello from Schenectady',
            'expected': True,
            'signal': 'text',
            'bucket': 'strong_local',
            'split': 'dev',
            'regression': True,
        },
        {
            'id': 'fn-author',
            'text': 'Newsletter is out.',
            'author_handle': 'timesunion.com',
            'expected': True,
            'signal': 'author',
            'bucket': 'local_org_no_placename',
            'split': 'holdout',
            'regression': True,
        },
        {
            'id': 'gap-author',
            'text': 'Open mic tonight.',
            'author_handle': 'not-allowlisted.bsky.social',
            'expected': True,
            'signal': 'author',
            'bucket': 'local_org_no_placename',
            'split': 'dev',
            'regression': False,
        },
    ]
    report = evaluate_cases(cases)
    assert report.overall.tp == 2
    assert report.overall.fn == 1
    assert report.by_split['dev'].tp == 1
    assert report.by_split['holdout'].tp == 1
    assert report.by_signal['author'].tp == 1
    assert report.by_signal['author'].fn == 1
    assert report.by_bucket['local_org_no_placename'].fn == 1
    assert len(report.failures) == 1
    assert len(report.regression_failures) == 0


def test_eval_cases_have_required_strata() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding='utf-8'))
    assert cases, 'eval_cases.json must not be empty'

    required = {'signal', 'bucket', 'split'}
    signals = set()
    buckets = set()
    splits = set()
    for case in cases:
        missing = required - case.keys()
        assert not missing, f'{case["id"]} missing fields: {sorted(missing)}'
        assert case['split'] in {'dev', 'holdout'}, case['id']
        assert case['signal'] in {'text', 'author', 'event'}, case['id']
        signals.add(case['signal'])
        buckets.add(case['bucket'])
        splits.add(case['split'])

    assert 'author' in signals
    assert 'event' in signals
    assert 'local_org_no_placename' in buckets
    assert 'author_soft_prior' in buckets
    assert 'regional_event' in buckets
    assert splits == {'dev', 'holdout'}

    soft_prior_cases = [c for c in cases if c.get('bucket') == 'author_soft_prior']
    assert soft_prior_cases, 'expected soft-prior eval cases'
    assert any(c.get('soft_prior') is True and c.get('expected') is True for c in soft_prior_cases)

    fn_author = [
        c
        for c in cases
        if c.get('bucket') == 'local_org_no_placename' and c.get('expected') is True
    ]
    fn_event = [
        c for c in cases if c.get('bucket') == 'regional_event' and c.get('expected') is True
    ]
    assert len(fn_author) >= 3
    assert len(fn_event) >= 3
    assert any(c.get('split') == 'holdout' for c in fn_author)
    assert any(c.get('split') == 'holdout' for c in fn_event)
