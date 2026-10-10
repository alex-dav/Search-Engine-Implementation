#!/usr/bin/env python3
"""Evaluate dev MAP for run JSON files in ``runs/``.

This uses the assignment MAP definition on the public dev files.
"""

import json
from pathlib import Path
from typing import Dict, List


def load_qrels(path: str = "data/dev/relevance_judge.json") -> Dict[str, Dict[str, int]]:
    """Load relevance scores; documents with score > 0 are relevant."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    qrels: Dict[str, Dict[str, int]] = {}
    for entry in data:
        qid = str(entry["qid"])
        rels: Dict[str, int] = {}
        for doc_id, score in entry["relevance_scores"].items():
            score_int = int(score)
            if score_int > 0:
                rels[str(doc_id)] = score_int
        qrels[qid] = rels
    return qrels


def average_precision(retrieved: List[str], relevant: Dict[str, int]) -> float:
    """Compute AP over the retrieved top results."""
    if not relevant:
        return 0.0
    relevant_docs = {doc_id for doc_id, score in relevant.items() if score > 0}
    if not relevant_docs:
        return 0.0

    precisions = []
    hits = 0
    seen = set()
    for rank, doc_id in enumerate(retrieved, 1):
        doc_key = str(doc_id)
        if doc_key in seen:
            continue
        seen.add(doc_key)
        if doc_key in relevant_docs:
            hits += 1
            precisions.append(hits / rank)

    return (sum(precisions) / len(relevant_docs)) if precisions else 0.0


def load_run(path: Path) -> Dict[str, List[str]]:
    """Load a run JSON file and keep the first 10 doc IDs per query."""
    data = json.loads(path.read_text(encoding="utf-8"))
    run: Dict[str, List[str]] = {}
    for item in data:
        qid = str(item["qid"])
        run[qid] = [str(doc_id) for doc_id in item["doc_ids"][:10]]
    return run


def mean_average_precision(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]]) -> float:
    """Compute MAP over every judged qid, treating missing results as empty."""
    aps = [
        average_precision(run.get(qid, []), relevant)
        for qid, relevant in qrels.items()
    ]
    return (sum(aps) / len(aps)) if aps else 0.0


def main() -> int:
    qrels_path = Path("data/dev/relevance_judge.json")
    runs_dir = Path("runs")
    if not qrels_path.exists():
        print(f"Missing {qrels_path}")
        return 2
    if not runs_dir.exists():
        print(f"Missing {runs_dir}")
        return 2

    qrels = load_qrels(str(qrels_path))
    run_files = sorted(runs_dir.glob("*.json"))
    if not run_files:
        print("No run JSON files found in runs/")
        return 1

    run_scores = [
        (path, mean_average_precision(load_run(path), qrels))
        for path in run_files
    ]

    print("Run file                       MAP")
    print("-----------------------------------")
    for path, score in run_scores:
        print(f"{path.name:<30} {score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
