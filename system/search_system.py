#!/usr/bin/env python3
"""Batch search-system CLI for Task 3.

Usage:
  python -m system.search_system <queries_json> <documents_jsonl> <run_output_json> [method]

The optional method argument is the Task 3 optimisation selector:
- ``no_optimisation`` is the runnable starter baseline.
- ``default`` is your submitted final system. In the starter, it is initially
  the same as ``no_optimisation``; add your own method branch if you improve it.

The Task 2 ranker method remains ``default`` unless you explicitly extend
the ranking interface elsewhere.
"""

import json
import pathlib
import sys
from typing import Dict, List, Set, Tuple

from index.builders import create_all_indexes
from query_processing.query_process import process_query
from ranking.rankers import rank_documents
from utils.text_preprocessing import preprocess
from utils.query_expansion import expand_query_wordnet
from utils.tfidf import build_vocab, compute_raw_tf


_TASK1_FALLBACK_NOTE = (
    "Ranking all documents so Tasks 2 and 3 remain runnable. "
    "This fallback does not show that Task 1 is correct. "
    "Run python -m test_sanity.check_submission and fix any Task 1 failures "
    "before submission."
)
_task1_fallback_warning_shown = False


def _warn_task1_fallback(reason: str) -> None:
    """Show the Task 1 fallback warning once per CLI run."""
    global _task1_fallback_warning_shown
    if _task1_fallback_warning_shown:
        return
    print(f"WARNING: {reason} {_TASK1_FALLBACK_NOTE}")
    _task1_fallback_warning_shown = True


def _load_docs(path: str) -> Tuple[List[dict], List[str]]:
    """Load JSONL documents with ``id`` and ``text`` fields."""
    raw_docs: List[dict] = []
    texts: List[str] = []
    seen_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if "id" not in obj or "text" not in obj:
                raise ValueError(f"Line {line_num} must contain 'id' and 'text'")
            doc_id = int(obj["id"])
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            raw_docs.append({"id": doc_id, "text": str(obj["text"])})
            texts.append(str(obj["text"]))

    return raw_docs, texts


def _load_queries(path: str) -> List[dict]:
    """Load query JSON list with ``qid`` and ``query`` fields."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("queries_json must contain a JSON list")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "qid" not in item or "query" not in item:
            raise ValueError(f"Query item {i} must contain 'qid' and 'query'")
    return data


def _all_doc_candidates(doc_ids: List[int]) -> Set[int]:
    """Return all document IDs."""
    return set(doc_ids)


def _basic_tokenize(text: str) -> List[str]:
    """Tiny baseline tokenizer for the runnable starter."""
    return str(text).split()


def _tokenize_texts(texts: List[str]) -> List[List[str]]:
    """Tokenize with the weak default baseline."""
    return [_basic_tokenize(text) for text in texts]


def _build_index(tokenized_docs: List[List[str]], doc_ids: List[int]) -> Tuple[pathlib.Path, bool]:
    """Build the Task 1 index package used by query processing."""
    cache_dir = pathlib.Path(__file__).resolve().parent.parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    index_path = cache_dir / "unified_package.pkl.gz"

    try:
        if index_path.exists():
            if not index_path.is_file():
                raise IsADirectoryError(f"Index path exists but is not a file: {index_path}")
            index_path.unlink()
        create_all_indexes(tokenized_docs, str(index_path), doc_ids=doc_ids)
        return index_path, True
    except NotImplementedError:
        _warn_task1_fallback("Task 1 index build is not implemented.")
    except Exception as e:
        _warn_task1_fallback(
            f"Task 1 index build failed ({type(e).__name__}: {e})."
        )
    return index_path, False


def _candidate_ids_for_query(
    query_text: str,
    qid: str,
    index_path: pathlib.Path,
    index_ready: bool,
    doc_ids: List[int],
    system_method: str,
) -> Set[int]:
    """Select candidates, using all documents only when Task 1 is unavailable."""
    if system_method not in {"no_optimisation", "default", "expanded_query_optimisation"}:
        raise ValueError(
            f"Unknown system method: {system_method}. "
            "Add a method branch in system/search_system.py."
        )

    if not index_ready:
        return _all_doc_candidates(doc_ids)

    try:
        candidate_ids = process_query(query_text, str(index_path))
    except NotImplementedError:
        _warn_task1_fallback(
            f"Task 1 query processing is not implemented for {qid}."
        )
        candidate_ids = _all_doc_candidates(doc_ids)
    except Exception as e:
        _warn_task1_fallback(
            f"Task 1 query processing failed for {qid} "
            f"({type(e).__name__}: {e})."
        )
        candidate_ids = _all_doc_candidates(doc_ids)

    # A successful empty result means that the query matched no documents.
    return candidate_ids


def _rank_one_query(
    qid: str,
    query_toks: List[str],
    candidate_ids: Set[int],
    doc_to_tokens: Dict[int, List[str]],
    index_path: pathlib.Path,
    doc_cache: dict,
) -> dict:
    """Rank candidates and format one top-10 run-file result object."""
    candidate_ids_list = sorted(
        int(doc_id)
        for doc_id in candidate_ids
        if int(doc_id) in doc_to_tokens
    )
    candidate_docs = [doc_to_tokens[doc_id] for doc_id in candidate_ids_list]

    ranked_ids, scores = rank_documents(
        query_toks=query_toks,
        candidate_docs=candidate_docs,
        doc_ids=candidate_ids_list,
        inverted_index_path=str(index_path),
        method="default",
        doc_cache=doc_cache,
    )

    return {
        "qid": qid,
        "doc_ids": [int(doc_id) for doc_id in ranked_ids[:10]],
        "scores": [float(score) for score in scores[:10]],
    }


def main() -> int:
    if len(sys.argv) not in (4, 5):
        print("Usage: python -m system.search_system <queries_json> <documents_jsonl> <run_output_json> [method]")
        return 2

    queries_path, docs_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    system_method = sys.argv[4] if len(sys.argv) == 5 else "default"
    if system_method not in {"no_optimisation", "default", "expanded_query_optimisation"}:
        raise ValueError(
            f"Unknown system method: {system_method}. "
            "Use 'no_optimisation' or 'default', or add a new method branch."
        )

    queries = _load_queries(queries_path)
    raw_docs, doc_texts = _load_docs(docs_path)

    use_preprocess = system_method == "default"
    # Preprocessing optimisation = default
    tokenized_docs = preprocess(doc_texts) if use_preprocess else _tokenize_texts(doc_texts)
    doc_ids = [int(doc["id"]) for doc in raw_docs]
    doc_to_tokens: Dict[int, List[str]] = {
        doc_id: toks
        for doc_id, toks in zip(doc_ids, tokenized_docs)
    }

    index_path, index_ready = _build_index(tokenized_docs, doc_ids)

    # Pre-compute vocabulary, raw TF matrix, and doc lengths once (all queries share these)
    doc_cache: dict = {}
    vocab = build_vocab(tokenized_docs)
    raw_tf, doc_lengths = compute_raw_tf(tokenized_docs, vocab)
    doc_cache["vocab"] = vocab
    doc_cache["raw_tf"] = raw_tf
    doc_cache["doc_lengths"] = doc_lengths
    doc_cache["doc_id_order"] = doc_ids

    results = []
    for item in queries:
        qid = str(item["qid"])
        query_text = str(item["query"])
        # Preprocessing optimisation = default
        query_toks = preprocess([query_text])[0] if use_preprocess else _basic_tokenize(query_text)
        query_for_processing = " ".join(query_toks)
        candidate_ids = _candidate_ids_for_query(
            query_text=query_for_processing,
            qid=qid,
            index_path=index_path,
            index_ready=index_ready,
            doc_ids=doc_ids,
            system_method=system_method,
        )
        query_toks = expand_query_wordnet(query_toks) if system_method == "expanded_query_optimisation" else query_toks
        results.append(_rank_one_query(qid, query_toks, candidate_ids, doc_to_tokens, index_path, doc_cache))

    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} query results to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
