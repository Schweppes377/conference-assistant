"""Local production contracts/failures; existing unittest suites are also collected."""
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.progress import read_input, save_checkpoint
from app.transcript import merge_transcript
from asr import ASRSegment
from diarization import DiarizationSegment
from rag.chunking import chunk_pages
from rag.models import PageText, SearchResult
from rag.pdf_loader import discover_pdf_files
from rag.retriever import SemanticRetriever, IndexNotReadyError
from rag.config import RAGConfig
from utils.audio import load_audio, AudioDecodingError


def test_timestamp_merge_accumulates_overlap_and_unknown():
    transcript = merge_transcript([ASRSegment(0, 4, "Текст"), ASRSegment(9, 10, "Далее")],
        [DiarizationSegment(0, 1, "B"), DiarizationSegment(1, 2, "B"), DiarizationSegment(2, 3, "A")])
    assert transcript[0].speaker == "B"
    assert transcript[1].speaker == "UNKNOWN"
    assert asdict(transcript[0]) == {"start": 0, "end": 4, "speaker": "B", "text": "Текст"}


def test_document_metadata_and_retrieval_contract():
    chunks = chunk_pages([PageText(Path("doc.pdf"), "Документ", "СП TEST", 7, "Требование к объекту.")], chunk_size=100, chunk_overlap=10)
    assert chunks[0].metadata()["page"] == 7
    assert chunks[0].metadata()["source_file"] == "doc.pdf"
    assert SearchResult(.1, chunks[0].text, chunks[0].metadata()).score == pytest.approx(.9)


def test_missing_documents_and_index_fail_before_loading_model(tmp_path):
    with pytest.raises(FileNotFoundError): discover_pdf_files(tmp_path / "absent")
    retriever = SemanticRetriever(RAGConfig(persist_dir=tmp_path / "absent"))
    with pytest.raises(IndexNotReadyError): retriever.retrieve("Требования", 5)
    assert retriever._embedder is None


def test_missing_and_empty_audio_do_not_invoke_models(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError): load_audio(tmp_path / "absent.wav")
    import utils.audio
    monkeypatch.setattr(utils.audio, "decode_audio", lambda *args, **kwargs: [])
    path = tmp_path / "empty.wav"; path.write_bytes(b"")
    with pytest.raises(AudioDecodingError): load_audio(path)


@pytest.mark.parametrize("payload,mode", [(42, "transcript"), ({"checkpoint_version": 1, "transcript": []}, "resume"),
    ([{"start": 0,"end": 1,"speaker": "S","text": " "}], "transcript"),
    ({"transcript": [{"start": 0,"end": 1,"speaker": "S","text": "x"}], "detected_issues": [None]}, "issues"),
    ({"checkpoint_version": 1,"transcript": [{"start": 0,"end": 1,"speaker": "S","text": "x"}], "answers": []}, "resume"),
    ({"checkpoint_version": 1,"transcript": [{"start": 0,"end": 1,"speaker": "S","text": "x"}], "answers": {"x": None}}, "resume")])
def test_bad_progress_and_analysis_friendly_value_error(tmp_path, payload, mode):
    path = tmp_path / "input.json"; path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError): read_input(path, mode=mode)


def test_invalid_json_and_atomic_progress_roundtrip(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError): read_input(path, mode="resume")
    state = {"checkpoint_version": 1, "transcript": [{"start": 0, "end": 1, "speaker": "S", "text": "Русский"}],
             "pending_issue_ids": ["ISSUE_002"], "answers": {}, "prepared": {}}
    save_checkpoint(path, state)
    assert read_input(path, mode="resume") == state
    assert not path.with_name(path.name + ".tmp").exists()


@pytest.mark.integration
def test_audio_metadata_stages_without_model_execution(tmp_path):
    import numpy as np
    from app.test_api_budget import FakeClient, FakeRetriever, pipeline
    from app.analysis_models import MeetingAnalysis
    path = tmp_path / "fixture.wav"; path.write_bytes(b"synthetic test placeholder")
    transcription = Mock()
    transcription.transcriber.transcribe.return_value = [ASRSegment(0, 1, "Проверить объект.")]
    transcription.diarizer.diarize.return_value = [DiarizationSegment(0, 1, "SPEAKER_00")]
    stages = []
    runner = pipeline(FakeClient(['{"issues":[]}']), FakeRetriever(), transcription_pipeline=transcription,
                      audio_loader=lambda *args, **kwargs: np.zeros(16000, dtype=np.float32),
                      on_stage=lambda n, text: stages.append(n))
    result = runner.analyze(path)
    assert stages == [1, 2, 3, 4, 5]
    assert result.audio_duration_sec == 1
    assert result.diarization_segments == [{"start": 0, "end": 1, "speaker": "SPEAKER_00"}]
    assert transcription.transcriber.transcribe.call_count == 1
    assert transcription.diarizer.diarize.call_count == 1


@pytest.mark.integration
def test_resume_explicit_asr_and_diarization_spies(tmp_path):
    from app.test_api_budget import FakeClient, FakeRetriever, pipeline, fixture, limited
    transcript, payload = fixture(2)
    checkpoint = tmp_path / "progress.json"
    first = pipeline(FakeClient([payload, "Ответ [SOURCE_1].", limited()]), FakeRetriever(), checkpoint_path=checkpoint)
    partial = first.analyze_transcript(transcript)
    assert partial.statistics.pending_issues == 1
    saved = read_input(checkpoint, mode="resume")
    first_answer = saved["answers"]["ISSUE_001"]
    transcription = Mock()
    client = FakeClient(["Ответ [SOURCE_1]."])
    runner = pipeline(client, FakeRetriever(), state=saved, checkpoint_path=checkpoint, transcription_pipeline=transcription)
    final = runner.analyze_transcript(transcript)
    transcription.transcriber.transcribe.assert_not_called()
    transcription.diarizer.diarize.assert_not_called()
    assert runner.state["answers"]["ISSUE_001"] == first_answer
    assert runner.state["pending_issue_ids"] == []
    assert len(client.prompts) == 1
    assert final.run_status == "completed"
