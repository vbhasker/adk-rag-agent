"""Evaluate retrieval configurations against the golden dataset. No LLM needed for the default set.

    python eval_retrieval.py                     # all offline configs, k=3
    python eval_retrieval.py --k 5
    python eval_retrieval.py --llm               # also multi-query + HyDE via Gemini (needs a key)

Every configuration is a function: question -> ranked list of doc ids. Add your own to CONFIGS.
Results are written to eval/results/retrieval.json (the playground reads them).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

from hybrid import get_retriever
from metrics import all_metrics, dedupe_docs
from pipeline import run_pipeline
from query_transform import hyde, multi_query
from rerank import get_reranker

EVAL_DIR = Path(__file__).parent / "eval"
RESULTS_DIR = EVAL_DIR / "results"
DEPTH = 10  # chunks retrieved per question before mapping to docs


def load_golden() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (EVAL_DIR / "golden.jsonl").read_text().splitlines() if line.strip()]


def _docs(hits: list[dict[str, Any]]) -> list[str]:
    return dedupe_docs([h["source_id"].split("#")[0] for h in hits])


def _mode(mode: str, fusion: str = "rrf", context: str = "header") -> Callable[[str], list[str]]:
    return lambda q: dedupe_docs([c.chunk.doc_id for c in get_retriever(context).search(q, DEPTH, mode, fusion)])


def _pipe(**kwargs: Any) -> Callable[[str], list[str]]:
    return lambda q: _docs(run_pipeline(q, top_k=DEPTH, **kwargs)["final"])


CONFIGS: dict[str, Callable[[str], list[str]]] = {
    "1 · BM25": _mode("bm25"),
    "2 · Vector": _mode("vector"),
    "3 · Hybrid (weighted α=0.5)": _mode("hybrid", "weighted"),
    "4 · Hybrid (RRF)": _mode("hybrid"),
    "5 · Hybrid + rerank, no chunk context": _pipe(context_mode="none"),
    "6 · Hybrid + rerank": _pipe(context_mode="header"),
    "7 · Hybrid + rerank + parent docs": _pipe(context_mode="header", parent_docs=True),
}

LLM_CONFIGS: dict[str, Callable[[str], list[str]]] = {
    "8 · + multi-query (Gemini)": lambda q: _docs(run_pipeline(q, top_k=DEPTH, queries=multi_query(q)["queries"])["final"]),
    "9 · + HyDE (Gemini)": lambda q: _docs(run_pipeline(q, top_k=DEPTH, hyde_passage=hyde(q)["passage"])["final"]),
}


def evaluate(configs: dict[str, Callable[[str], list[str]]], k: int = 3) -> dict[str, Any]:
    golden = [g for g in load_golden() if g["relevant_docs"]]  # unanswerable questions have nothing to find
    report: dict[str, Any] = {
        "k": k, "questions": len(golden), "configs": {},
        "embedder": get_retriever().store.embedder_name, "reranker": get_reranker().name,
    }
    for name, fn in configs.items():
        fn(golden[0]["question"])  # warm-up: build indexes / load models outside the timing
        rows, t0 = [], time.perf_counter()
        for g in golden:
            ranked = fn(g["question"])
            rows.append({"id": g["id"], "type": g["type"], "question": g["question"], "relevant": g["relevant_docs"],
                         "ranked": ranked[:k], **all_metrics(ranked, set(g["relevant_docs"]), k)})
        metric_names = [m for m in rows[0] if "@" in m or m == "mrr"]
        summary = {m: round(mean(r[m] for r in rows), 3) for m in metric_names}
        by_type = {t: round(mean(r["mrr"] for r in rows if r["type"] == t), 3) for t in dict.fromkeys(r["type"] for r in rows)}
        report["configs"][name] = {
            "summary": summary, "mrrByType": by_type, "rows": rows,
            "msPerQuery": round((time.perf_counter() - t0) * 1000 / len(golden), 1),
        }
    return report


def print_report(report: dict[str, Any]) -> None:
    k = report["k"]
    cols = [f"hit@{k}", f"recall@{k}", "mrr", f"ndcg@{k}", f"precision@{k}"]
    print(f"\n{report['questions']} answerable golden questions, k={k} · embedder {report['embedder']} · reranker {report['reranker']}\n")
    print(f"{'configuration':42s}" + "".join(f"{c:>13s}" for c in cols) + f"{'ms/q':>8s}")
    for name, res in report["configs"].items():
        print(f"{name:42s}" + "".join(f"{res['summary'][c]:13.3f}" for c in cols) + f"{res['msPerQuery']:8.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--llm", action="store_true", help="include Gemini-powered configs")
    args = parser.parse_args()

    report = evaluate({**CONFIGS, **(LLM_CONFIGS if args.llm else {})}, args.k)
    print_report(report)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval.json").write_text(json.dumps(report, indent=1))
    print(f"\nSaved {RESULTS_DIR / 'retrieval.json'}")


if __name__ == "__main__":
    main()
