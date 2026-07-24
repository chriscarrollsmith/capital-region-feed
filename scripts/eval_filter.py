#!/usr/bin/env python3
"""Evaluate matcher precision/recall against labeled fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.matcher import match_post  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--cases',
        type=Path,
        default=ROOT / 'data' / 'eval_cases.json',
        help='Path to labeled eval JSON',
    )
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    tp = fp = tn = fn = 0
    failures: list[str] = []

    for case in cases:
        result = match_post(
            case.get('text', ''),
            alt_text=case.get('alt_text', ''),
            author_did=case.get('author_did'),
            author_handle=case.get('author_handle'),
            allowlist_handles={
                'news10.bsky.social',
                'timesunion.com',
                'albany-ny.bsky.social',
                'cbs6albany.bsky.social',
                'wrgb.bsky.social',
            },
        )
        expected = bool(case['expected'])
        if result.matched and expected:
            tp += 1
        elif result.matched and not expected:
            fp += 1
            failures.append(
                f'FP {case["id"]}: reason={result.reason} :: {case.get("text", "")[:90]}'
            )
        elif not result.matched and not expected:
            tn += 1
        else:
            fn += 1
            failures.append(
                f'FN {case["id"]}: reason={result.reason} :: {case.get("text", "")[:90]}'
            )

        if args.verbose:
            mark = 'OK' if result.matched == expected else 'MISS'
            print(
                f'[{mark}] {case["id"]} expected={expected} got={result.matched} ({result.reason})'
            )

    total = len(cases)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    print(f'cases={total} tp={tp} fp={fp} tn={tn} fn={fn}')
    print(f'precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} accuracy={accuracy:.3f}')
    if failures:
        print('\nFailures:')
        for line in failures:
            print(f'  - {line}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
