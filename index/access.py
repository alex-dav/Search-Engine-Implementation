"""Access functions for the unified Task 1 index package."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .io import load


_CacheSig = Tuple[int, int, int]
_package_cache: Dict[str, Tuple[_CacheSig, Dict[str, Any]]] = {}


def clear_package_cache(index_path: Optional[str] = None) -> None:
    """Clear cached index packages.

    Use this after rebuilding an index at the same path in a long-running
    process. If ``index_path`` is omitted, clear all cached packages.
    """
    if index_path is None:
        _package_cache.clear()
        return
    _package_cache.pop(str(Path(index_path).resolve()), None)


def _file_signature(path: Path) -> _CacheSig:
    """Return a cheap signature used to detect rebuilt index packages."""
    st = path.stat()
    return (int(st.st_mtime_ns), int(st.st_size), int(getattr(st, "st_ino", 0)))


def _load_package(index_path: str) -> Dict[str, Any]:
    """Load and cache one index package for repeated dictionary lookups.

    The cache is invalidated automatically when the file at ``index_path`` is
    rebuilt. This avoids accidentally reading a stale dev index when tests
    rebuild the same path for a different corpus.
    """
    path = Path(index_path)
    key = str(path.resolve())
    sig = _file_signature(path)
    cached = _package_cache.get(key)
    if cached is None or cached[0] != sig:
        _package_cache[key] = (sig, load(str(path)))
    return _package_cache[key][1]


def get_posting_list(term: str, index_path: str) -> List[int]:
    """Return sorted doc IDs for a term, or ``[]`` if missing."""
    package = _load_package(index_path)
    entry = package.get("unified", {}).get(term)
    if not isinstance(entry, dict):
        return []
    postings = entry.get("postings", {})
    return sorted(int(doc_id) for doc_id in postings)


def get_all_doc_ids(index_path: str) -> List[int]:
    """Return all document IDs in the indexed collection, in ascending order."""
    package = _load_package(index_path)
    doc_ids = package.get("__META__", {}).get("doc_ids", [])
    return sorted(int(doc_id) for doc_id in doc_ids)


def find_wildcard_matches(ngram: str, index_path: str) -> List[str]:
    """Return sorted terms for one character n-gram, or ``[]`` if missing."""
    package = _load_package(index_path)
    return list(package.get("wildcard", {}).get(ngram, []))
