"""Cross-platform serialization helpers for index packages.

This file is provided so Task 1 can focus on index contents, not pickle/gzip
details. You may use these helpers as-is.
"""

import gzip
import pickle
from pathlib import Path
from typing import Any


def dump(obj: Any, path: str) -> None:
    """Write a Python object to a gzip-compressed pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: str) -> Any:
    """Load an object written by dump(...)."""
    with gzip.open(path, "rb") as f:
        return pickle.load(f)
