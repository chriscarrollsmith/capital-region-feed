"""Checked-in place gazetteer for Capital Region entity disambiguation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAZETTEER_PATH = ROOT / 'data' / 'gazetteer' / 'places.json'


@dataclass(frozen=True)
class GazetteerHit:
    entity_id: str
    region: str
    surface: str


@dataclass(frozen=True)
class Gazetteer:
    """Longest-surface-first matcher over checked-in place entities."""

    patterns: tuple[tuple[re.Pattern[str], str, str], ...]

    def lookup(self, text: str) -> GazetteerHit | None:
        haystack = text or ''
        if not haystack:
            return None
        for pattern, entity_id, region in self.patterns:
            match = pattern.search(haystack)
            if match:
                return GazetteerHit(
                    entity_id=entity_id,
                    region=region,
                    surface=re.sub(r'\s+', ' ', match.group(0).lower()),
                )
        return None


def _surface_pattern(surface: str) -> re.Pattern[str]:
    # Word-boundary aware; allow flexible whitespace inside multi-word surfaces.
    parts = [re.escape(part) for part in surface.split() if part]
    if not parts:
        raise ValueError(f'empty gazetteer surface: {surface!r}')
    body = r'\s+'.join(parts)
    return re.compile(rf'(?<!\w){body}(?!\w)', re.IGNORECASE)


def load_gazetteer(path: Path | None = None) -> Gazetteer:
    gazetteer_path = path or DEFAULT_GAZETTEER_PATH
    payload = json.loads(gazetteer_path.read_text(encoding='utf-8'))
    entities = payload.get('entities') or []
    compiled: list[tuple[int, re.Pattern[str], str, str]] = []
    for entity in entities:
        entity_id = str(entity['id'])
        region = str(entity['region'])
        if region not in {'capital_ny', 'other'}:
            raise ValueError(f'unknown region {region!r} for {entity_id}')
        for surface in entity.get('surfaces') or []:
            surface_text = str(surface).strip().lower()
            if not surface_text:
                continue
            compiled.append((len(surface_text), _surface_pattern(surface_text), entity_id, region))
    # Longest surface first so "albany park" beats a future bare "albany".
    compiled.sort(key=lambda item: item[0], reverse=True)
    patterns = tuple((pattern, entity_id, region) for _, pattern, entity_id, region in compiled)
    return Gazetteer(patterns=patterns)


@lru_cache(maxsize=1)
def default_gazetteer() -> Gazetteer:
    return load_gazetteer()
