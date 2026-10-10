"""Query type detection for the undergraduate Task 1."""

import re


def detect_query_type(query: str) -> str:
    """Return ``wildcard``, ``boolean``, or ``natural_language``.

    Detection is intentionally syntax-based:
    - ``*`` -> wildcard
    - uppercase ``AND``/``OR``/``NOT`` -> boolean
    - otherwise -> natural language

    Operators are case-sensitive in the assignment data.
    """
    if "*" in query:
        return "wildcard"
    if re.search(r"\b(AND|OR|NOT)\b", query):
        return "boolean"
    return "natural_language"
