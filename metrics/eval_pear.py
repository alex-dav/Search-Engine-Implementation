#!/usr/bin/env python3
"""Evaluate dev Pearson correlation for ``ranking.rankers.rank_documents``.

Task 2 uses a fixed tokenization protocol so this helper measures ranking rather
than the student's Part A preprocessing choices.
"""

import html
import hashlib
import json
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from nltk.tokenize import word_tokenize

from index.access import clear_package_cache
from index.builders import create_all_indexes
from ranking.rankers import rank_documents


def _simple_tokens(text: str) -> List[str]:
    """Apply the fixed Task 2 tokenization used by the marking evaluator."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("*", " ")
    tokens = word_tokenize(text)
    return [
        token.lower()
        for token in tokens
        if re.fullmatch(r"[a-z0-9]+", token.lower())
    ]


def _is_natural_query(query: str) -> bool:
    """Return whether a query belongs to the Task 2 natural-language split."""
    if re.search(r"\b(AND|OR|NOT)\b", query):
        return False
    return not any(symbol in query for symbol in ('*', '"', '(', ')'))


def _pearson(pred: List[float], gold: List[float]) -> float:
    y_pred = np.asarray(pred, dtype=float)
    y_gold = np.asarray(gold, dtype=float)
    if y_pred.size < 2 or np.std(y_pred) == 0 or np.std(y_gold) == 0:
        return 0.0
    r = float(np.corrcoef(y_pred, y_gold)[0, 1])
    return 0.0 if np.isnan(r) else r


def _stable_candidate_order(qid: str, doc_ids: List[int]) -> List[int]:
    """Return a reproducible order that carries no relevance information."""
    digest = hashlib.sha256(str(qid).encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    ordered = list(doc_ids)
    rng.shuffle(ordered)
    return ordered


def _load_docs(path: str = "data/dev_ranking/documents.jsonl") -> Dict[int, str]:
    docs: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            docs[int(obj["id"])] = str(obj["text"])
    return docs


def _load_queries(path: str = "data/dev_ranking/queries.json") -> Dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(item["qid"]): str(item["query"]) for item in data}


def _try_build_case_index(
    candidate_docs: List[List[str]],
    doc_ids: List[int],
    index_path: str,
) -> None:
    """Build the candidate-set index passed to ``rank_documents``."""
    path = Path(index_path)
    if path.exists():
        if not path.is_file():
            raise IsADirectoryError(f"Dev index path is not a file: {path}")
        path.unlink()

    create_all_indexes(candidate_docs, index_path, doc_ids=doc_ids)
    clear_package_cache(index_path)


def main() -> int:
    method = sys.argv[1] if len(sys.argv) > 1 else "default"
    docs = _load_docs()
    queries = _load_queries()
    relevance = json.loads(
        Path("data/dev_ranking/relevance_judge.json").read_text(encoding="utf-8")
    )

    per_query = []
    natural_items = [
        item
        for item in relevance
        if _is_natural_query(queries[str(item["qid"])])
    ]
    with tempfile.TemporaryDirectory(prefix="partb_eval_pear_") as temp_dir:
        for item in natural_items:
            qid = str(item["qid"])
            doc_ids = _stable_candidate_order(
                qid,
                [int(doc_id) for doc_id in item["candidate_doc_ids"]],
            )
            candidate_docs = [_simple_tokens(docs[doc_id]) for doc_id in doc_ids]
            query_toks = _simple_tokens(queries[qid])
            index_path = str(Path(temp_dir) / f"index_{qid}.pkl.gz")
            _try_build_case_index(candidate_docs, doc_ids, index_path)

            ranked_ids, scores = rank_documents(
                query_toks,
                candidate_docs,
                doc_ids,
                index_path,
                method=method,
            )
            if len(ranked_ids) != len(doc_ids) or len(scores) != len(doc_ids):
                raise ValueError("rank_documents must return every candidate document")
            if set(map(int, ranked_ids)) != set(doc_ids):
                raise ValueError("ranked document IDs must be a permutation of the candidates")

            gold_map = {
                int(doc_id): float(score)
                for doc_id, score in item["relevance_scores"].items()
            }
            gold = [gold_map.get(int(doc_id), 0.0) for doc_id in ranked_ids]
            per_query.append(_pearson(scores, gold))

    avg = round(float(np.mean(per_query)), 3) if per_query else 0.0
    print(f"Method: {method}")
    print(f"Natural-language dev queries: {len(per_query)}")
    print(f"Average Pearson: {avg:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
