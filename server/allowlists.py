"""Load Capital Region author allowlists from ``data/``.

Jetstream ingest supplies author DIDs only, so production matching relies on
``allowlist_dids.txt``. Handles remain the curated source of truth; refresh DIDs
with ``scripts/resolve_allowlist_dids.py`` after editing the handle list.
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
HANDLES_PATH = DATA_DIR / 'allowlist_handles.txt'
DIDS_PATH = DATA_DIR / 'allowlist_dids.txt'


def load_list_file(path: Path) -> list[str]:
    """Load non-empty, non-comment lines from an allowlist-style file."""
    if not path.is_file():
        return []
    values: list[str] = []
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        values.append(line)
    return values


def load_allowlist_handles(path: Path | None = None) -> set[str]:
    """Return lowercased handles from the allowlist file."""
    return {h.lower() for h in load_list_file(path or HANDLES_PATH)}


def load_allowlist_dids(path: Path | None = None) -> set[str]:
    """Return DIDs from the allowlist file (exact spelling preserved)."""
    return set(load_list_file(path or DIDS_PATH))
