"""Ranking module for Task 2.

Provides a BM25 ranker built on the unified index statistics stored in the
index package, plus the original keyword-match baseline for comparison.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from index.access import _load_package
from utils.tfidf import tfidf_variants, tfidf_multi_mode, compute_idf
from utils.embeddings import semantic_vector


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine similarity between two vectors using NumPy."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    norm_x = float(np.linalg.norm(x))
    norm_y = float(np.linalg.norm(y))
    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0
    return float(np.dot(x, y) / (norm_x * norm_y))


def _score_binary_match(query_toks: Sequence[str], doc_toks: Sequence[str]) -> float:
    """Return 1.0 if the query and document share any token, else 0.0."""
    if not query_toks or not doc_toks:
        return 0.0
    return 1.0 if (set(query_toks) & set(doc_toks)) else 0.0


def _rank_keyword_match(
    query_toks: List[str],
    candidate_docs: List[List[str]],
    doc_ids: List[int],
) -> Tuple[List[int], List[float]]:
    """Weak runnable baseline: any shared query token gets score 1.0."""
    scored = []
    for doc_id, doc_toks in zip(doc_ids, candidate_docs):
        scored.append((int(doc_id), _score_binary_match(query_toks, doc_toks)))

    scored.sort(key=lambda item: (-item[1], item[0]))
    ranked_ids = [doc_id for doc_id, _ in scored]
    scores = [float(score) for _, score in scored]
    return ranked_ids, scores


def rank_tfidf(
    query_toks: List[str],
    candidate_docs: List[List[str]],
    doc_ids: List[int],
    doc_cache: Optional[Dict[str, object]] = None,
) -> Tuple[List[int], List[float]]:
    """Rank candidates using TF-IDF (length mode) cosine similarity.

    Args:
        doc_cache: Optional pre-computed dict with keys 'raw_tf', 'vocab',
            'doc_lengths', 'doc_id_order'. If provided, skips vocab/TF
            recomputation and only computes IDF from the candidate subset.
    """
    if (doc_cache is not None
            and "raw_tf" in doc_cache
            and "vocab" in doc_cache
            and "doc_lengths" in doc_cache
            and "doc_id_order" in doc_cache):
        vocab = doc_cache["vocab"]
        all_raw_tf = doc_cache["raw_tf"]
        all_doc_lengths = doc_cache["doc_lengths"]
        all_ids = doc_cache["doc_id_order"]
        id_to_idx = {int(did): i for i, did in enumerate(all_ids)}
        candidate_row_indices = [id_to_idx[int(did)] for did in doc_ids]

        idf = compute_idf(candidate_docs, vocab)
        idf = np.nan_to_num(idf, nan=0.0, posinf=0.0, neginf=0.0)
        idf_mask = (idf > 0).astype(np.float32)
        n_terms = len(vocab)

        scored = []
        for i, doc_id in enumerate(doc_ids):
            ri = candidate_row_indices[i]
            raw_row = all_raw_tf[ri]
            doc_len = all_doc_lengths[ri]
            tfidf_row = (raw_row / doc_len) * idf

            qv = np.zeros(n_terms, dtype=np.float32)
            for t in query_toks:
                if t in vocab:
                    qv[vocab[t]] += 1.0
            qv *= idf_mask

            scored.append((int(doc_id), cosine_similarity(qv, tfidf_row)))
    else:
        X, vocab = tfidf_variants(candidate_docs, tf_mode="len")
        n_terms = len(vocab)

        query_vec = np.zeros(n_terms, dtype=np.float32)
        for t in query_toks:
            if t in vocab:
                query_vec[vocab[t]] += 1.0

        scored = []
        for i, doc_id in enumerate(doc_ids):
            scored.append((int(doc_id), cosine_similarity(query_vec, X[i])))

    scored.sort(key=lambda item: (-item[1], item[0]))
    ranked_ids = [doc_id for doc_id, _ in scored]
    scores = [float(score) for _, score in scored]
    return ranked_ids, scores


def rank_hybrid(
    query_toks: List[str],
    candidate_docs: List[List[str]],
    doc_ids: List[int],
    inverted_index_path: str,
    doc_cache: Optional[Dict[str, object]] = None,
) -> Tuple[List[int], List[float]]:
    """Rank candidates by combining power-transformed, min-max normalised signals.

    Args:
        doc_cache: Optional pre-computed document vectors. If provided, must be
            a dict with keys 'raw_tf', 'vocab', 'doc_lengths', 'doc_id_order'.
            Skips vocab/TF recomputation and computes IDF from candidate subset.
    """
    package = _load_package(inverted_index_path)
    meta = package.get("__META__", {})
    assert meta.get("N", 0) > 0, "Unified index must have N > 0"

    # TF-IDF signals: pre-compute vocab + raw TF once, compute IDF per query
    if (doc_cache is not None
            and "raw_tf" in doc_cache
            and "vocab" in doc_cache
            and "doc_lengths" in doc_cache
            and "doc_id_order" in doc_cache):
        vocab = doc_cache["vocab"]
        all_raw_tf = doc_cache["raw_tf"]
        all_doc_lengths = doc_cache["doc_lengths"]
        all_ids = doc_cache["doc_id_order"]
        id_to_idx = {int(did): i for i, did in enumerate(all_ids)}
        candidate_row_indices = [id_to_idx[int(did)] for did in doc_ids]

        idf = compute_idf(candidate_docs, vocab)
        idf = np.nan_to_num(idf, nan=0.0, posinf=0.0, neginf=0.0)
        idf_mask = (idf > 0).astype(np.float32)
        n_terms = len(vocab)

        # TF-IDF raw frequency (length mode)
        qv_len = np.zeros(n_terms, dtype=np.float32)
        for t in query_toks:
            if t in vocab:
                qv_len[vocab[t]] += 1.0
        qv_len *= idf_mask
        tfidf_raw = []
        for ri in candidate_row_indices:
            raw_row = all_raw_tf[ri]
            doc_len = all_doc_lengths[ri]
            tfidf_row = (raw_row / doc_len) * idf
            tfidf_raw.append(cosine_similarity(qv_len, tfidf_row))

        # TF-IDF (log mode)
        qv_log = np.zeros(n_terms, dtype=np.float32)
        for t in query_toks:
            if t in vocab:
                qv_log[vocab[t]] += 1.0
        qv_log *= idf_mask
        tfidf_log = []
        for ri in candidate_row_indices:
            raw_row = all_raw_tf[ri]
            log_tf = np.zeros(n_terms, dtype=np.float32)
            nonzero = raw_row > 0
            log_tf[nonzero] = (1 + np.log(raw_row[nonzero])) * idf[nonzero]
            tfidf_log.append(cosine_similarity(qv_log, log_tf))
    else:
        tfidf_results = tfidf_multi_mode(candidate_docs, modes=("len", "log"))
        X_len, vocab_len = tfidf_results["len"]
        X_log, vocab_log = tfidf_results["log"]
        candidate_row_indices = list(range(len(doc_ids)))

        qv_len = np.zeros(len(vocab_len), dtype=np.float32)
        for t in query_toks:
            if t in vocab_len:
                qv_len[vocab_len[t]] += 1.0
        tfidf_raw = [cosine_similarity(qv_len, X_len[ri]) for ri in candidate_row_indices]

        qv_log = np.zeros(len(vocab_log), dtype=np.float32)
        for t in query_toks:
            if t in vocab_log:
                qv_log[vocab_log[t]] += 1.0
        tfidf_log = [cosine_similarity(qv_log, X_log[ri]) for ri in candidate_row_indices]

    # Dense-vector semantic (tfidf-weighted method) - always per-query
    doc_vecs = semantic_vector(candidate_docs, method="tfidf_weighted")
    query_dense = semantic_vector([query_toks], method="tfidf_weighted")[0]
    dense_raw = [
        cosine_similarity(query_dense, doc_vecs[i]) for i in range(len(doc_ids))
    ]

    # Power transform normalise
    p_tfidf, p_log, p_dense = 2.0, 2.0, 4.0
    t_arr = np.array([max(0.0, s) ** p_tfidf for s in tfidf_raw])
    l_arr = np.array([max(0.0, s) ** p_log for s in tfidf_log])
    d_arr = np.array([max(0.0, s) ** p_dense for s in dense_raw])

    # Min-max normalise
    def _minmax(a: np.ndarray) -> np.ndarray:
        lo, hi = a.min(), a.max()
        if hi - lo < 1e-12:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    n_t = _minmax(t_arr)
    n_l = _minmax(l_arr)
    n_d = _minmax(d_arr)

    # Weighted combination
    w_tfidf, w_log, w_dense = 0.35, 0.35, 0.30
    combined = w_tfidf * n_t + w_log * n_l + w_dense * n_d

    scored = [(int(doc_ids[i]), float(combined[i])) for i in range(len(doc_ids))]
    scored.sort(key=lambda item: (-item[1], item[0]))
    ranked_ids = [doc_id for doc_id, _ in scored]
    scores = [float(score) for _, score in scored]
    return ranked_ids, scores


def rank_documents(
    query_toks: List[str],
    candidate_docs: List[List[str]],
    doc_ids: List[int],
    inverted_index_path: str,
    method: str = "default",
    doc_cache: Optional[Dict[str, object]] = None,
) -> Tuple[List[int], List[float]]:
    """Rank every candidate document.

    Args:
        query_toks: Tokenized query terms.
        candidate_docs: Tokenized candidate documents aligned with ``doc_ids``.
        doc_ids: Candidate document IDs.
        inverted_index_path: Path to the unified index package.
        method: Method selector. ``default`` is an alias set inside this
            function.
        doc_cache: Optional pre-computed document-level vectors. If provided,
            the ranker will skip recomputing TF-IDF matrices and semantic
            vectors for documents.
            Add extra ``elif`` branches using real method names for experiments,
            then point ``best_method`` to your best-performing final method.

    Returns:
        ``(ranked_doc_ids, scores)`` for all candidate docs. ``ranked_doc_ids``
        must be a permutation of the input ``doc_ids``.
    """
    if len(candidate_docs) != len(doc_ids):
        raise ValueError("candidate_docs and doc_ids must have the same length")

    # Task 2 overall process:
    # query tokens + candidate documents -> one score per candidate
    # -> ranked document IDs + aligned scores
    #
    # TF-IDF/vector route:
    # You may adapt your Part A Task 3 TF-IDF and Task 4 semantic-vector
    # implementations for this route.
    # See the Week 03 workshop sections "Implement TF-IDF", "Implement GloVe
    # mean pooling", and "Cosine similarity and ranking comparison". The
    # cosine helper compares two completed representations. You still construct
    # query and document representations, score every candidate, sort all
    # documents, and connect the selected method to default.
    #
    # BM25 route:
    # The unified index uses the Week 05 term-statistics structure: each term
    # stores df and per-document tf. ``index.io.load(inverted_index_path)``
    # loads the package. See the Week 05 workshop sections "Inspect
    # index-stored statistics" and "Implement BM25 scoring". Adapt that work
    # to this ranking contract if you choose BM25.
    #
    # Adapt one or more routes to this candidate-ranking contract. Return every
    # candidate, keep scores aligned with document IDs, and make best_method
    # identify the method represented by "default".

    best_method = "hybrid"

    if method == "default":
        method = best_method

    if method == "keyword_match":
        return _rank_keyword_match(query_toks, candidate_docs, doc_ids)

    elif method == "tfidf":
        return rank_tfidf(query_toks, candidate_docs, doc_ids, doc_cache=doc_cache)

    elif method == "hybrid":
        return rank_hybrid(query_toks, candidate_docs, doc_ids, inverted_index_path, doc_cache=doc_cache)

    raise ValueError(f"Unknown ranking method: {method}")
