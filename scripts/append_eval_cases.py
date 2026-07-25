#!/usr/bin/env python3
"""Merge hand-labeled JSONL candidates into ``data/eval_cases.json``.

Reads rows produced by ``scripts/collect_eval_sample.py`` (or compatible JSON
objects). Only rows with a boolean ``expected`` are appended. Existing ids are
left unchanged. Missing stratum fields get CLI defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / 'data' / 'eval_cases.json'

JsonObject = dict[str, Any]

VALID_SIGNALS = frozenset({'text', 'author', 'event'})
VALID_SPLITS = frozenset({'dev', 'holdout'})


def read_jsonl(path: Path | None) -> list[JsonObject]:
    if path is None:
        raw = sys.stdin.read()
    else:
        raw = path.read_text(encoding='utf-8')
    rows: list[JsonObject] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith('#'):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f'invalid JSON on line {line_no}: {exc}') from exc
        if not isinstance(payload, dict):
            raise ValueError(f'line {line_no} must be a JSON object')
        rows.append(payload)
    return rows


def normalize_case(
    row: JsonObject,
    *,
    default_signal: str,
    default_bucket: str,
    default_split: str,
    default_regression: bool,
) -> JsonObject | None:
    """Return a scored eval case, or None if the row is still unlabeled."""
    expected = row.get('expected')
    if expected is None:
        return None
    if not isinstance(expected, bool):
        raise ValueError(f'{row.get("id")!r}: expected must be a boolean, got {expected!r}')

    case_id = str(row.get('id') or '').strip()
    if not case_id:
        raise ValueError('labeled row is missing id')

    text = row.get('text')
    if not isinstance(text, str):
        raise ValueError(f'{case_id}: text must be a string')

    signal = str(row.get('signal') or default_signal)
    if signal not in VALID_SIGNALS:
        raise ValueError(f'{case_id}: invalid signal {signal!r}')

    split = str(row.get('split') or default_split)
    if split not in VALID_SPLITS:
        raise ValueError(f'{case_id}: invalid split {split!r}')

    bucket = str(row.get('bucket') or default_bucket).strip() or default_bucket
    regression = row.get('regression', default_regression)
    if not isinstance(regression, bool):
        raise ValueError(f'{case_id}: regression must be a boolean')

    case: JsonObject = {
        'id': case_id,
        'text': text,
        'expected': expected,
        'signal': signal,
        'bucket': bucket,
        'split': split,
        'regression': regression,
    }
    for key in ('author_handle', 'author_did', 'alt_text', 'note'):
        value = row.get(key)
        if value is not None and value != '':
            case[key] = value
    return case


def merge_cases(
    existing: list[JsonObject],
    incoming: list[JsonObject],
    *,
    default_signal: str,
    default_bucket: str,
    default_split: str,
    default_regression: bool,
) -> tuple[list[JsonObject], dict[str, int]]:
    by_id = {str(case['id']): case for case in existing if 'id' in case}
    stats = {'added': 0, 'skipped_unlabeled': 0, 'skipped_existing': 0}

    for row in incoming:
        case = normalize_case(
            row,
            default_signal=default_signal,
            default_bucket=default_bucket,
            default_split=default_split,
            default_regression=default_regression,
        )
        if case is None:
            stats['skipped_unlabeled'] += 1
            continue
        case_id = str(case['id'])
        if case_id in by_id:
            stats['skipped_existing'] += 1
            continue
        by_id[case_id] = case
        existing.append(case)
        stats['added'] += 1

    return existing, stats


def write_cases(path: Path, cases: list[JsonObject]) -> None:
    path.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--cases',
        type=Path,
        default=DEFAULT_CASES,
        help='Eval cases JSON to update (default: data/eval_cases.json)',
    )
    parser.add_argument(
        '--input',
        type=Path,
        help='Labeled JSONL file (default: stdin)',
    )
    parser.add_argument(
        '--default-signal',
        default='text',
        choices=sorted(VALID_SIGNALS),
        help='Signal used when a row omits one',
    )
    parser.add_argument(
        '--default-bucket',
        default='unspecified',
        help='Bucket used when a row omits one',
    )
    parser.add_argument(
        '--default-split',
        default='dev',
        choices=sorted(VALID_SPLITS),
        help='Split used when a row omits one',
    )
    parser.add_argument(
        '--default-regression',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Regression flag used when a row omits one (default: true)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Report merge stats without writing the cases file',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        incoming = read_jsonl(args.input)
    except ValueError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1

    if not args.cases.is_file():
        print(f'error: cases file not found: {args.cases}', file=sys.stderr)
        return 1

    existing = json.loads(args.cases.read_text(encoding='utf-8'))
    if not isinstance(existing, list):
        print(f'error: {args.cases} must contain a JSON array', file=sys.stderr)
        return 1

    try:
        merged, stats = merge_cases(
            existing,
            incoming,
            default_signal=args.default_signal,
            default_bucket=args.default_bucket,
            default_split=args.default_split,
            default_regression=args.default_regression,
        )
    except ValueError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1

    if not args.dry_run and stats['added']:
        write_cases(args.cases, merged)

    print(
        f'added={stats["added"]} '
        f'skipped_unlabeled={stats["skipped_unlabeled"]} '
        f'skipped_existing={stats["skipped_existing"]} '
        f'total={len(merged)}' + (' (dry-run)' if args.dry_run else '')
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
