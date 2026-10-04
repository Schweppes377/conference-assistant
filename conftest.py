"""All pytest tests are local; record reproducible reliability, not availability."""
import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

_reports = {}
_network_guard = None


def pytest_sessionstart(session):
    global _network_guard
    _reports.clear()
    _network_guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Pytest network forbidden, including collection"))
    _network_guard.start()


def pytest_addoption(parser):
    parser.addoption("--reliability-output", default="evaluation/results/reliability.json")


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.startswith("rag/test_store.py"):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def no_external_inference(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not use network or load inference models")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.delenv("SBER_ULTRA", raising=False)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    with patch("asr.transcriber._load_model", side_effect=forbidden), \
         patch("diarization.diarizer._load_pipeline", side_effect=forbidden), \
         patch("rag.embeddings.SentenceTransformerEmbedder._get_model", side_effect=forbidden):
        yield


def pytest_runtest_logreport(report):
    if report.when == "call" or report.failed or report.skipped:
        old = _reports.get(report.nodeid)
        if not old or old["outcome"] != "failed":
            _reports[report.nodeid] = {"outcome": report.outcome, "phase": report.when, "seconds": report.duration}


def pytest_sessionfinish(session, exitstatus):
    counts = {state: sum(r["outcome"] == state for r in _reports.values()) for state in ("passed", "failed", "skipped")}
    collection_errors = max(0, session.testsfailed - counts["failed"])
    denominator = counts["passed"] + counts["failed"] + collection_errors
    result = {"tests": len(_reports), **counts, "collection_errors": collection_errors,
              "success_rate": counts["passed"] / denominator if denominator else None,
              "exit_status": int(exitstatus), "measured_at": datetime.now(timezone.utc).isoformat(),
              "definition": "passed / (passed + failed + collection_errors); excludes skipped; not SLA",
              "network_allowed": False, "cases": _reports}
    path = Path(session.config.getoption("--reliability-output"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if _network_guard is not None:
        _network_guard.stop()
