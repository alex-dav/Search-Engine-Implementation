"""Task 1 index package containing unified term and wildcard indexes."""

from .builders import create_all_indexes
from .access import get_posting_list, find_wildcard_matches
from .io import dump, load

__all__ = [
    "create_all_indexes",
    "get_posting_list", 
    "find_wildcard_matches",
    "dump",
    "load"
]
