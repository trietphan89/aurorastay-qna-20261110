"""Reference-based DeepEval benchmark; never used as a live confidence score."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import pandas as pd

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

from rag_core import RULES, answer_query, load_credentials

WEIGHTS = {"faithfulness": .35, "contextual_recall": .25,
           "contextual_precision": .20, "answer_relevancy": .20}
THRESHOLDS = {"faithfulness": .85, "contextual_recall": .80,
              "contextual_precision": .75, "answer_relevancy": .75}
SPECS = {
    "faithfulness": ("Faithfulness", ["ACTUAL_OUTPUT", "RETRIEVAL_CONTEXT"], [
        "Identify the factual claims in the actual output.",
        "Check every claim against retrieval context. Penalize contradictions, unsupported numbers, conditions or exceptions.",
        "Ignore wording differences and citation formatting. Give a high score when all important claims are supported."]),
    "contextual_recall": ("Contextual Recall", ["EXPECTED_OUTPUT", "RETRIEVAL_CONTEXT"], [
        "Identify the important factual claims required by the expected output.",
        "Check whether retrieval context supports each required claim, semantically rather than by exact wording.",
        "Penalize important missing information. Give a high score when retrieval contains the information needed for the expected answer."]),
    "contextual_precision": ("Contextual Precision", ["INPUT", "RETRIEVAL_CONTEXT"], [
        "Identify which retrieved passages help answer the input question.",
        "Check whether useful passages are prioritized over unrelated or redundant material.",
        "Penalize retrieval dominated by irrelevant information. Give a high score when most retrieved information is useful."]),
    "answer_relevancy": ("Answer Relevancy", ["INPUT", "ACTUAL_OUTPUT"], [
        "Check whether the actual output addresses every part of the input question.",
        "Penalize off-topic, unnecessary material or failure to answer. Do not penalize a concise but sufficient answer.",
        "Give a high score when the answer directly addresses the question with an appropriate staff-facing response."]),
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ["rag_core.py", "retrieval.py"]}


def metric_factory(key: str):
    from deepeval.metrics import GEval
    from deepeval.models import OpenAIModel
    try:
        from deepeval.test_case import SingleTurnParams as Params
    except ImportError:
        from deepeval.test_case import LLMTestCaseParams as Params
    name, fields, steps = SPECS[key]
    judge = OpenAIModel(model="gpt-4o", temperature=0, **load_credentials())
    # Explicit steps avoid changing the grading instructions between cases.
    return GEval(name=name, evaluation_steps=steps,
                 evaluation_params=[getattr(Params, field) for field in fields],
                 threshold=THRESHOLDS[key], model=judge,
                 async_mode=False, verbose_mode=False)


def route(scores: dict[str, float | None]) -> tuple[float | None, str, bool]:
    if any(scores.get(key) is None for key in WEIGHTS):
        return None, "MANUAL LOOKUP", False
    value = sum(WEIGHTS[key] * float(scores[key]) for key in WEIGHTS)
    # The business criteria include the 0.75 boundary. Review counts as non-auto.
    label = "AUTO ANSWER" if value >= .75 else ("ASSISTED REVIEW" if value >= .50 else "MANUAL LOOKUP")
    strict_pass = all(float(scores[key]) >= THRESHOLDS[key] for key in WEIGHTS)
    return value, label, strict_pass


def freeze_configuration(config: dict[str, Any], corpus_hash: str, reason: str,
                         artifact_dir: str | Path = "artifacts") -> dict[str, Any]:
    dest = Path(artifact_dir) / "frozen_config.json"
    record = {"config": config, "corpus_hash": corpus_hash, "prompt_hash": digest(RULES),
              "metric_specs_hash": digest(SPECS), "implementation_hashes": implementation_hashes(),
              "frozen_at": now(), "reason": reason}
    if dest.exists():
        existing = json.loads(dest.read_text())
        if any(existing[key] != record[key] for key in ["config", "corpus_hash", "prompt_hash", "metric_specs_hash", "implementation_hashes"]):
            raise RuntimeError("Frozen test configuration cannot be silently changed.")
        return existing
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=2))
    return record


def evaluate_dataset(dataset_path: str | Path, vectorstore: Any, config: dict[str, Any],
                     split: str, corpus_hash: str, artifact_dir: str | Path = "artifacts",
                     workers: int = 2, prepare_only: bool = False) -> pd.DataFrame:
    from deepeval.test_case import LLMTestCase
    if split not in {"evaluation", "test"}:
        raise ValueError("Use evaluation or test split.")
    folder = Path(artifact_dir)
    if split == "test":
        frozen = json.loads((folder / "frozen_config.json").read_text())
        assert frozen["config"] == config and frozen["corpus_hash"] == corpus_hash
        assert frozen["prompt_hash"] == digest(RULES) and frozen["metric_specs_hash"] == digest(SPECS)
        assert frozen["implementation_hashes"] == implementation_hashes()
    data = pd.read_csv(dataset_path)
    assert len(data) and data[["question", "expected_answer", "relevant_context"]].notna().all().all()
    run = {"config": config, "corpus_hash": corpus_hash, "prompt_hash": digest(RULES),
           "metric_specs": SPECS, "judge": "gpt-4o", "deepeval": importlib.metadata.version("deepeval"),
           "implementation_hashes": implementation_hashes()}
    run_hash = digest(run)
    cache = folder / "runs" / f"{split}-{run_hash[:12]}"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "run_manifest.json").write_text(json.dumps({**run, "run_hash": run_hash}, indent=2))

    def process(index: int, row: pd.Series) -> dict[str, Any]:
        started = time.perf_counter()
        path = cache / f"case-{index+1:02}.json"
        row_hash = digest(row.to_dict())
        record = json.loads(path.read_text()) if path.exists() else {}
        if record:
            assert record["row_hash"] == row_hash
        if not record.get("generation"):
            generation = answer_query(row["question"], vectorstore=vectorstore, **config)
            record = {"case_id": f"{split[:1].upper()}{index+1:02}", "split": split,
                      "row_hash": row_hash, "recorded_at": now(),
                      "question": row["question"], "expected_answer": row["expected_answer"],
                      "reference_context": [x.strip() for x in row["relevant_context"].split("<>") if x.strip()],
                      "generation": generation, "metrics": {}}
            path.write_text(json.dumps(record, indent=2))
        if prepare_only:
            print(f"{record['case_id']} answer prepared for source review", flush=True)
            return {"case_id": record["case_id"], "question": record["question"],
                    "answer": record["generation"]["answer"], "record_path": str(path)}
        case = LLMTestCase(input=record["question"], actual_output=record["generation"]["answer"],
                           expected_output=record["expected_answer"],
                           retrieval_context=record["generation"]["relevant_chunks"],
                           context=record["reference_context"])
        for key in WEIGHTS:
            if record["metrics"].get(key, {}).get("score") is not None:
                continue
            metric = metric_factory(key)
            try:
                metric.measure(case)
                raw_score = float(metric.score)
                if not math.isfinite(raw_score) or not -1e-12 <= raw_score <= 1 + 1e-12:
                    raise ValueError("Judge score is outside the 0-1 range.")
                # G-Eval's probability-weighted mean may exceed an endpoint by
                # floating-point roundoff only. Preserve raw values for audit.
                score = min(1.0, max(0.0, raw_score))
                record["metrics"][key] = {"score": score, "reason": metric.reason,
                                            "raw_score": raw_score,
                                            "threshold": THRESHOLDS[key], "recorded_at": now()}
            except Exception as exc:
                record["metrics"][key] = {"score": None, "error_type": type(exc).__name__,
                                            "threshold": THRESHOLDS[key], "recorded_at": now()}
                path.write_text(json.dumps(record, indent=2))
                raise RuntimeError(f"{record['case_id']} {key}: {type(exc).__name__}") from exc
            path.write_text(json.dumps(record, indent=2))
        scores = {key: record["metrics"][key]["score"] for key in WEIGHTS}
        composite, routing, strict = route(scores)
        record.update({"composite": composite, "route": routing, "all_thresholds_pass": strict,
                       "processing_seconds_this_invocation": time.perf_counter() - started})
        path.write_text(json.dumps(record, indent=2))
        print(f"{record['case_id']} scored: {routing}", flush=True)
        return {"case_id": record["case_id"], "question": record["question"], **scores,
                "composite": composite, "route": routing, "all_thresholds_pass": strict,
                "generation_seconds": record["generation"]["elapsed_seconds"],
                "answer": record["generation"]["answer"],
                "chunk_ids": ", ".join(x["chunk_id"] for x in record["generation"]["sources"])}

    completed = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process, i, row) for i, row in data.iterrows()]
        for future in as_completed(futures):
            completed.append(future.result())
    frame = pd.DataFrame(completed).sort_values("case_id").reset_index(drop=True)
    if prepare_only:
        frame.to_csv(cache / "prepared_answers.csv", index=False)
        return frame
    assert len(frame) == len(data) and frame[list(WEIGHTS)].notna().all().all()
    frame.to_csv(folder / f"{split}_results.csv", index=False)
    summary = {"split": split, "n": len(frame), "run_hash": run_hash,
               "means": {key: float(frame[key].mean()) for key in WEIGHTS},
               "mean_composite": float(frame.composite.mean()),
               "auto_count": int((frame.route == "AUTO ANSWER").sum()),
               "auto_rate": float((frame.route == "AUTO ANSWER").mean()),
               "strict_all_metrics_count": int(frame.all_thresholds_pass.sum()),
               "thresholds": THRESHOLDS, "routing_boundary": "AUTO >= 0.75",
               "scope": "Reference-based benchmark eligibility, not calibrated live confidence or observed workload reduction."}
    (folder / f"{split}_summary.json").write_text(json.dumps(summary, indent=2))
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["evaluation", "test"], required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    from retrieval import load_index
    config = {"k": args.k, "model_name": "gpt-4o-mini", "temperature": 0.0,
              "top_p": 1.0, "max_tokens": 512}
    index = Path("artifacts/faiss_index")
    corpus_hash = digest({str(p.relative_to(index)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(index.rglob('*')) if p.is_file()})
    evaluate_dataset(Path("data") / f"{args.split}.csv", load_index(index), config,
                     args.split, corpus_hash, prepare_only=args.prepare_only)
