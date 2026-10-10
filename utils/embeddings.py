"""Light-weight GloVe loader & semantic aggregation (Task A-4)."""
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

# ----------------------------------------------------------------------
# 1.  Load the provided local GloVe subset
# ----------------------------------------------------------------------
_GLOVE_PATH = Path(__file__).resolve().parents[1] / "data" / "glove" / "glove_subset_200d.pkl"


def _ensure_glove() -> Dict[str, np.ndarray]:
    if not _GLOVE_PATH.exists():
        raise FileNotFoundError(
            f"Missing GloVe subset: {_GLOVE_PATH}. "
            "Keep data/glove/glove_subset_200d.pkl in your submission folder."
        )

    with _GLOVE_PATH.open("rb") as f:
        loaded = pickle.load(f)

    if not isinstance(loaded, dict):
        raise TypeError(f"GloVe subset must be a dict, got {type(loaded)}")

    vocab: Dict[str, np.ndarray] = {
        str(word): np.asarray(vec, dtype=float)
        for word, vec in loaded.items()
    }

    if not vocab:
        raise ValueError(f"GloVe subset is empty: {_GLOVE_PATH}")

    # deterministic random <unk>
    if "<unk>" not in vocab:
        rng = np.random.default_rng(seed=42)
        dim = len(next(iter(vocab.values())))
        vocab["<unk>"] = rng.normal(0.0, 0.05, size=dim)

    return vocab


_glove_cache: Dict[str, np.ndarray] | None = None
_DIM: int | None = None


def _get_glove() -> Dict[str, np.ndarray]:
    """Lazily load and cache the GloVe vectors on first access."""
    global _glove_cache, _DIM
    if _glove_cache is None:
        _glove_cache = _ensure_glove()
        _DIM = len(next(iter(_glove_cache.values())))
    return _glove_cache


def _get_dim() -> int:
    """Return the GloVe embedding dimension, triggering load if needed."""
    if _DIM is None:
        _get_glove()
    return _DIM  # type: ignore[return-value]


# ----------------------------------------------------------------------
# 2.  Public helper
# ----------------------------------------------------------------------
def _key(tok: str) -> str:
    """Return the exact vocabulary key, or '<unk>' for an OOV token."""
    glove = _get_glove()
    return tok if tok in glove else "<unk>"


# ----------------------------------------------------------------------
# 3.  Main entry
# ----------------------------------------------------------------------
def semantic_vector(docs: List[List[str]], method: str = "meanmax") -> np.ndarray:
    """Build one semantic vector per document.

    Args:
        docs: One token list per document.
        method: Aggregation method, either 'tfidf_weighted' or 'meanmax'.

    Returns:
        A matrix with shape (N, D) for 'tfidf_weighted' or (N, 2D) for
        'meanmax', where N is the number of documents and D is the embedding
        dimension.
    """
    if method not in ("tfidf_weighted", "meanmax"):
            raise ValueError(f"Unknown method: {method}")
    
    if method == "tfidf_weighted":
        return tfidf_weighted_vector(docs)

    # meanmax method
    else:
        return meanmax_vector(docs)

def tfidf_weighted_vector(docs: List[List[str]]) -> np.ndarray:
    """Build a TF-IDF weighted semantic vector for each document.
    Args:
        docs: One token list per document.
    Returns:
        A matrix with shape (N, D), where N is the number of documents and D is the embedding dimension.
    """
    # global document frequency for IDF
    glove = _get_glove()
    dim = _get_dim()
    n = len(docs)
    df = Counter()
    for doc in docs:
        df.update(set(doc))
    n_docs = len(docs)

    X = np.zeros((n, dim), dtype=np.float32)
    for i, doc in enumerate(docs):
        if not doc:
            continue
        tf = Counter(doc)
        total = sum(tf.values())
        for tok, count in tf.items():
            # Use the smoothed IDF formula to avoid division by zero and handle unseen terms
            idf = np.log((n_docs + 1) / (df[tok] + 1)) + 1
            # Raw relative frequency (count / total tokens in the document)
            weight = (count / total) * idf
            X[i] += weight * glove[_key(tok)]
    return X

def meanmax_vector(docs: List[List[str]]) -> np.ndarray:
    """Build a mean-max semantic vector for each document.
    Args:
        docs: One token list per document.
    Returns:
        A matrix with shape (N, 2D), where N is the number of documents and D is the embedding dimension.
    """
    if not docs:
        dim = _get_dim()
        return np.zeros((len(docs), 2 * dim), dtype=np.float32)

    glove = _get_glove()
    dim = _get_dim()
    doc_vectors = np.zeros((len(docs), 2 * dim), dtype=np.float32)
    for i, doc in enumerate(docs):
        if not doc:
            doc_vectors[i] = np.zeros(2 * dim, dtype=np.float32)
            continue

        unk = glove.get("<unk>", np.zeros(dim, dtype=np.float32))
        vecs = [glove.get(_key(token), unk) for token in doc]
        mean_vec = np.mean(vecs, axis=0).astype(np.float32)
        max_vec = np.max(vecs, axis=0).astype(np.float32)
        # Concatenate mean and max vectors to form the final document vector
        doc_vectors[i] = np.concatenate([mean_vec, max_vec])

    return doc_vectors
