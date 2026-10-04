"""Evaluate saved offline artifacts. Default execution performs no inference."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from evaluation.data import (ROOT, read_json, resolve_file, validate_analysis,
                             validate_dataset, audio_duration, performance_metrics)
from evaluation.metrics import asr_metrics, diarization_metrics, detection_metrics, retrieval_metrics
from utils.console import configure_utf8_console


def evaluate(dataset: dict, analysis: dict, *, progress: dict | None = None,
             reliability: dict | None = None, live_retrieval: bool = False) -> dict:
    validate_dataset(dataset)
    validate_analysis(analysis)
    reference = dataset.get("reference_transcript")
    if isinstance(reference, list):
        reference = " ".join(t["text"] for t in sorted(reference, key=lambda t: t["start"]))
    hypothesis = " ".join(t["text"] for t in sorted(analysis["transcript"], key=lambda t: t["start"]))
    predicted = analysis["detected_issues"]
    matching = dataset.get("matching", {"mode": "time", "threshold": .5})
    detected = detection_metrics(dataset.get("reference_issues"), predicted, **matching)
    pairs = dict(detected.get("matches", []))
    queries = []
    service = None
    if live_retrieval:
        # Reuse the unchanged production preparation path, not another search.
        # HF offline flag prevents model downloads even on fallback code paths.
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from llm.config import QAConfig
        from llm.service import QuestionAnsweringService
        from rag.config import RAGConfig
        from rag.retriever import SemanticRetriever, IndexNotReadyError
        service = QuestionAnsweringService(retriever=SemanticRetriever(RAGConfig()),
                                           llm_client=None, config=QAConfig.from_env())
    for index, issue in enumerate(dataset.get("reference_issues") or []):
        expected = issue.get("expected_sources")
        if not expected:
            continue
        sources, available, origin = [], False, "not saved"
        if service:
            if not isinstance(issue.get("question"), str) or not issue["question"].strip():
                raise ValueError("Live retrieval requires each reference question")
            try:
                prepared = service.prepare(issue["question"], search_terms=issue.get("search_terms", []))
            except IndexNotReadyError as error:
                raise ValueError(str(error)) from None
            sources, available, origin = [asdict(s) for s in prepared.context_sources], True, "existing production prepare()"
        elif index in pairs:
            predicted_issue = predicted[pairs[index]]
            saved = (progress or {}).get("prepared", {}).get(predicted_issue["id"])
            if saved is not None:
                sources, available, origin = saved["context_sources"], True, "saved final context"
            elif "retrieved_candidates" in predicted_issue:
                sources, available, origin = predicted_issue["retrieved_candidates"], True, "saved reranked candidates (not exact final context)"
        queries.append({"reference_id": issue["id"], "expected": expected, "sources": sources,
                        "available": available, "ranking_source": origin})
    duration = analysis.get("audio_duration_sec")
    duration_source = "saved measured duration" if duration is not None else None
    if duration is None:
        duration, duration_source = audio_duration(resolve_file(dataset["audio_file"]) if dataset.get("audio_file") else None)
    if duration is not None and (not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0):
        raise ValueError("Measured audio duration must be positive")
    status_checks = []
    for index, issue in enumerate(dataset.get("reference_issues") or []):
        if issue.get("expected_status") is not None:
            actual = predicted[pairs[index]]["status"] if index in pairs else None
            status_checks.append({"reference_id": issue["id"], "expected": issue["expected_status"],
                                  "actual": actual, "matches": actual == issue["expected_status"]})
    return {"schema_version": 1, "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_id": dataset.get("id"), "llm_provider": analysis.get("llm_provider"),
            "llm_model": analysis.get("llm_model"), "asr": asr_metrics(reference, hypothesis),
            "diarization": diarization_metrics(dataset.get("reference_diarization"), analysis.get("diarization_segments"),
                collar=dataset.get("der", {}).get("collar_sec", .25),
                skip_overlap=dataset.get("der", {}).get("skip_overlap", False),
                evaluation_regions=dataset.get("evaluation_regions")),
            "question_detection": {**detected, "matching": matching},
            "retrieval": retrieval_metrics(queries),
            "performance": performance_metrics(analysis, duration, duration_source),
            "reliability": reliability or {"tests": None, "passed": None, "failed": None,
                                           "success_rate": None, "reason": "pytest results not provided"},
            "answer_status_checks": status_checks,
            "answer_facts": {"reason": "manual expert assessment required; no LLM-as-a-judge"},
            "observed_issues": [{"id": i["id"], "question": i["normalized_question"],
                                 "status": i["status"], "sources": len(i.get("cited_sources", []))} for i in predicted],
            "observed_statistics": analysis["statistics"]}


def save_results(summary: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in ("asr", "diarization", "question_detection", "retrieval", "performance", "reliability"):
        values = {k: v for k, v in summary[name].items() if not isinstance(v, (list, dict))}
        with (directory / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(values)); writer.writeheader(); writer.writerow(values)
    for name, rows in (("issues", summary["observed_issues"]), ("retrieval_queries", summary["retrieval"].get("per_query", []))):
        with (directory / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as file:
            fields = list(rows[0]) if rows else ["reference_id", "reason"]
            writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description="Локальная оценка сохранённых результатов; без LLM и аудиоинференса.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/datasets/construction_meeting.template.json")
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--reliability", type=Path, default=ROOT / "evaluation/results/reliability.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation/results/baseline")
    parser.add_argument("--live-retrieval", action="store_true", help="Явно запустить только существующий RAG с локальным cache; без LLM")
    args = parser.parse_args()
    try:
        dataset = read_json(args.dataset)
        validate_dataset(dataset)
        analysis_path = args.analysis or resolve_file(dataset["hypothesis"]["analysis_file"])
        progress_path = args.progress or (resolve_file(dataset["hypothesis"]["progress_file"]) if dataset.get("hypothesis", {}).get("progress_file") else None)
        if progress_path and not progress_path.is_file():
            raise FileNotFoundError(f"Progress не найден: {progress_path}")
        analysis = read_json(analysis_path)
        summary = evaluate(dataset, analysis, progress=read_json(progress_path) if progress_path else None,
                           reliability=read_json(args.reliability) if args.reliability.is_file() else None,
                           live_retrieval=args.live_retrieval)
        summary["artifacts"] = [{"file": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                for path in (args.dataset, analysis_path, progress_path) if path is not None]
        save_results(summary, args.output_dir)
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(f"Ошибка evaluation: {error}")
        return 1
    print("Раздел                 Результат")
    for name, key in (("asr", "wer"), ("diarization", "der"), ("question_detection", "f1"), ("retrieval", "hit_at_5"), ("performance", "total_sec"), ("reliability", "success_rate")):
        value = summary[name].get(key)
        print(f"{name:22} {key}={value}" if value is not None else f"{name:22} SKIPPED — {summary[name].get('reason')}")
    print(f"Summary: {(args.output_dir / 'summary.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
