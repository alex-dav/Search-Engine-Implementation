"""Main query-processing entry point for Task 1."""

from typing import Set

from query_processing.boolean import process_boolean_query
from query_processing.detection import detect_query_type
from query_processing.wildcard import process_wildcard_query

# Task 1 overall process:
# query string -> query type -> matching query processor
# -> candidate document IDs -> Task 2 ranking


def convert_natural_language(nl_query: str) -> str:
    """Convert whitespace-separated terms to a Boolean OR query.

    Example:
        ``climate change policy`` -> ``climate OR change OR policy``
    """
    tokens = [tok for tok in nl_query.split() if tok]
    return " OR ".join(tokens)


def process_query(query: str, index_path: str) -> Set[int]:
    """Process a query with the appropriate Task 1 processor.

    This wrapper is used by the end-to-end search system. The individual
    processor functions can also be checked directly.
    """
    qtype = detect_query_type(query)
    if qtype == "boolean":
        return process_boolean_query(query, index_path)
    if qtype == "wildcard":
        return process_wildcard_query(query, index_path)
    boolean_query = convert_natural_language(query)
    return process_boolean_query(boolean_query, index_path) if boolean_query else set()
