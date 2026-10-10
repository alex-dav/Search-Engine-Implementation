"""TF-IDF variants (Task A-3)."""
from typing import Dict, List, Tuple

import numpy as np


def tfidf_variants(
        docs: List[List[str]],
        tf_mode: str = "len",
        k: float = 1.2
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build a TF-IDF document-term matrix.
    Args:
        docs: One token list per document.
        tf_mode: Term-frequency mode, one of 'len', 'log', or 'bm25'.
            The 'log' mode uses the natural logarithm.
        k: BM25 term-frequency saturation parameter.
    Returns:
        A pair (X, vocab). X has one row per document and one column per
        vocabulary term. vocab contains every unique input term and assigns
        columns in alphabetical term order.
    """
    if tf_mode not in ("len", "log", "bm25"):
        raise ValueError(f"Unknown tf_mode: {tf_mode}")

    # Build vocabulary
    vocab = build_vocab(docs)
    N = len(docs)
    # Compute inverse document frequencies
    idf = compute_idf(docs, vocab)
    X = np.zeros((N, len(vocab)), dtype=np.float32)
    for i, doc in enumerate(docs):
        if not doc:
            continue

        # Compute term frequencies
        for term in doc:
            if term in vocab:
                X[i, vocab[term]] += 1.0

        # Compute TF-IDF values
        if tf_mode == "len":
            X[i] = (X[i] / float(len(doc))) * idf
        elif tf_mode == "log":
            nonzero = X[i] > 0
            X[i, nonzero] = (1 + np.log(X[i, nonzero])) * idf[nonzero]
            X[i, ~nonzero] = 0
        else:
            # BM25 mode
            X[i] = ((k + 1) * X[i]) / (k + X[i]) * idf

    return X, vocab

def build_vocab(tokenised_docs: List[List[str]]) -> Dict[str, int]:
    """Build a deterministic vocabulary dictionary from tokenised documents.

    The dictionary maps token -> column index.
    Terms are sorted so the same corpus always gives the same mapping.
    """
    terms = sorted({token for doc in tokenised_docs for token in doc})
    return {term: idx for idx, term in enumerate(terms)}

def compute_idf(tokenised_docs: List[List[str]], vocab: Dict[str, int]) -> np.ndarray:
    """Compute IDF values aligned with the vocabulary dictionary.

    Formula: idf(t) = log(N / df(t)).
    Output shape: (len(vocab),).
    """
    n_docs = len(tokenised_docs)
    df = np.zeros(len(vocab), dtype=np.float32)

    # Compute document frequencies
    for tokens in tokenised_docs:
        for token in set(tokens):
            if token in vocab:
                df[vocab[token]] += 1.0

    safe_df = np.maximum(df, 1.0)
    return np.log(n_docs / safe_df).astype(np.float32)


def compute_raw_tf(
    docs: List[List[str]], vocab: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute raw term-frequency matrix and per-document lengths.

    Returns:
        (X, doc_lengths) where X[i, j] is the raw count of term j in doc i,
        and doc_lengths[i] is the number of tokens in doc i.
    """
    N = len(docs)
    X = np.zeros((N, len(vocab)), dtype=np.float32)
    doc_lengths = np.zeros(N, dtype=np.float32)
    for i, doc in enumerate(docs):
        doc_lengths[i] = len(doc) if doc else 1.0
        if not doc:
            continue
        for term in doc:
            if term in vocab:
                X[i, vocab[term]] += 1.0
    return X, doc_lengths


def tfidf_multi_mode(
    docs: List[List[str]],
    modes: Tuple[str, ...] = ("len", "log"),
    k: float = 1.2,
) -> Dict[str, Tuple[np.ndarray, Dict[str, int]]]:
    """Build TF-IDF matrices for multiple modes using a shared vocab and IDF.

    Args:
        docs: One token list per document.
        modes: Which TF modes to compute (subset of 'len', 'log', 'bm25').
        k: BM25 term-frequency saturation parameter.

    Returns:
        A dict mapping mode name -> (X_matrix, vocab).
    """
    vocab = build_vocab(docs)
    N = len(docs)
    idf = compute_idf(docs, vocab)

    results: Dict[str, Tuple[np.ndarray, Dict[str, int]]] = {}
    for mode in modes:
        if mode not in ("len", "log", "bm25"):
            raise ValueError(f"Unknown tf_mode: {mode}")
        X = np.zeros((N, len(vocab)), dtype=np.float32)
        for i, doc in enumerate(docs):
            if not doc:
                continue
            for term in doc:
                if term in vocab:
                    X[i, vocab[term]] += 1.0
            if mode == "len":
                X[i] = (X[i] / float(len(doc))) * idf
            elif mode == "log":
                nonzero = X[i] > 0
                X[i, nonzero] = (1 + np.log(X[i, nonzero])) * idf[nonzero]
                X[i, ~nonzero] = 0
            else:  # bm25
                X[i] = ((k + 1) * X[i]) / (k + X[i]) * idf
        results[mode] = (X, vocab)

    return results
