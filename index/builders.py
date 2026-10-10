"""Unified index package builder for Task 1.

Build one on-disk package containing:
- ``unified``: term -> document frequency and per-document term frequency
- ``wildcard``: character n-gram -> sorted vocabulary terms

Tokens are already cleaned before they reach this function. Use them as-is.
"""

from collections import Counter
from typing import Any, Dict, List, Optional

from .io import dump


def create_all_indexes(
    tokenized_docs: List[List[str]],
    index_path: str,
    doc_ids: Optional[List[int]] = None,
) -> None:
    """Build and serialize the unified index package.

    Args:
        tokenized_docs: One token list per document. Do not clean or edit tokens here.
        index_path: Destination path for the serialized package.
        doc_ids: Optional integer document IDs aligned with ``tokenized_docs``.
            If omitted, use ``0..N-1``.
    """
    if doc_ids is None:
        doc_ids = list(range(len(tokenized_docs)))

    if len(doc_ids) != len(tokenized_docs):
        raise ValueError("doc_ids must have the same length as tokenized_docs")

    doc_ids = [int(doc_id) for doc_id in doc_ids]
    doc_lengths = {
        int(doc_id): len(tokens)
        for doc_id, tokens in zip(doc_ids, tokenized_docs)
    }
    avgdl = (sum(doc_lengths.values()) / len(doc_lengths)) if doc_lengths else 0.0

    # Keep this top-level schema aligned with the appendix.
    package: Dict[str, Any] = {
        "__META__": {
            "N": len(tokenized_docs),
            "doc_ids": doc_ids,
            "doc_lengths": doc_lengths,
            "avgdl": avgdl,
            "version": "1.0",
            "char_ngrams_max": 3,
        },
        "unified": {},
        "wildcard": {},
    }

    term_postings: Dict[str, Dict[int, int]] = {}
    for doc_id, tokens in zip(doc_ids, tokenized_docs):
        for term, tf in Counter(tokens).items():
            term_postings.setdefault(term, {})[doc_id] = int(tf)
    package["unified"] = {
        term: {
            "df": len(term_postings[term]),
            "postings": {
                doc_id: {"tf": term_postings[term][doc_id]}
                for doc_id in sorted(term_postings[term])
            },
        }
        for term in sorted(term_postings)
    }

    # Task 1 overall process:
    # provided unified index + vocabulary terms
    # -> boundary-padded character n-grams -> wildcard index
    # completed indexes + metadata -> serialized index package
    #
    # wildcard index:
    # Part A Task 2 provides the boundary-padded character n-gram foundation.
    # See the Week 04 lecture sections "k-gram Index" and
    # "Wildcard Expansion Pipeline".
    #
    # The package schema and serialization stage are already provided below.
    # (Task 1): fill the wildcard index, then serialize the package.
    # Required behaviour:
    # - posting lists are sorted, deduplicated integer doc IDs
    # - wildcard values are sorted, deduplicated vocabulary terms
    # - wildcard char n-grams come from "$" + term + "$", length 1..3
    # - do not index the unigram "$"
    gram_postings = unified_index_from_package(package)

    package["wildcard"] = {
        gram: sorted(set(terms)) for gram, terms in gram_postings.items()
    }

    dump(package, index_path)

def unified_index_from_package(package: Dict[str, Any]) -> Dict[str, Any]:
    gram_postings = {}
    for term in package["unified"]:
        padded_term = f"${term}$"
        for n in range(1, 4):
            for i in range(len(padded_term) - n + 1):
                gram = padded_term[i: i + n]
                if gram == "$":
                    continue
                gram_postings.setdefault(gram, []).append(term)
    return gram_postings
