"""Synthetic fixtures test evaluators, not quality of the real meeting."""
import pytest

from evaluation.metrics import (normalize_text, edit_counts, asr_metrics,
                                diarization_metrics, detection_metrics, match_issues, retrieval_metrics)
from evaluation.data import performance_metrics, validate_dataset, validate_analysis
from evaluation.run import evaluate, save_results


def test_text_normalization_does_not_hide_words_or_yo():
    assert normalize_text("  ЁЖ,   дом! 520 м² ") == "ёж дом 520 м²"
    assert normalize_text("ёлка") != normalize_text("елка")


def test_edit_alignment_and_wer_counts():
    assert edit_counts(["a", "b", "c"], ["a", "x", "c", "d"]) == {"substitutions": 1, "deletions": 0, "insertions": 1}
    result = asr_metrics("один два три", "один четыре три пять")
    assert result["wer"] == pytest.approx(2/3)
    assert result["reference_word_count"] == 3
    assert result["hypothesis_word_count"] == 4
    assert result["substitutions"] == result["insertions"] == 1
    assert asr_metrics("кот", "код")["cer"] == pytest.approx(1/3)


def test_missing_and_empty_reference_not_fake_zero():
    assert asr_metrics(None, "текст")["wer"] is None
    assert asr_metrics("", "текст")["wer"] is None
    assert asr_metrics("слово", "")["deletions"] == 1
    assert diarization_metrics(None, None)["der"] is None
    assert detection_metrics(None, [])["f1"] is None
    assert retrieval_metrics([])["hit_at_1"] is None


def test_der_permutation_is_handled_by_standard_metric():
    reference = [{"start": 0, "end": 2, "speaker": "Мария"}, {"start": 2, "end": 4, "speaker": "Иван"}]
    hypothesis = [{"start": 0, "end": 2, "speaker": "SPEAKER_01"}, {"start": 2, "end": 4, "speaker": "SPEAKER_00"}]
    regions = [{"start": 0, "end": 4}]
    assert diarization_metrics(reference, hypothesis, collar=0, evaluation_regions=regions)["der"] == 0
    assert diarization_metrics(reference, hypothesis[:1], collar=0, evaluation_regions=regions)["der"] == pytest.approx(.5)


def test_der_overlap_and_uem():
    turns = [{"start": 0, "end": 3, "speaker": "A"}, {"start": 1, "end": 2, "speaker": "B"}]
    result = diarization_metrics(turns, turns, collar=0, evaluation_regions=[{"start": 0, "end": 3}])
    assert result["der"] == 0
    assert result["details"]["total"] == 4


def test_question_matching_one_to_one_duplicate_is_fp():
    reference = [{"id": "R1", "start": 0, "end": 4}]
    predicted = [{"start": 0, "end": 4}, {"start": 0, "end": 4}, {"start": 10, "end": 14}]
    result = detection_metrics(reference, predicted, mode="time", threshold=.5)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 0)
    assert result["precision"] == pytest.approx(1/3)
    assert result["recall"] == 1


def test_maximum_matching_not_greedy_order():
    reference = [{"segment_ids": ["a", "b"]}, {"segment_ids": ["a"]}]
    predicted = [{"segment_ids": ["a"]}, {"segment_ids": ["b"]}]
    assert len(match_issues(reference, predicted, mode="segments", threshold=.5)) == 2


def test_threshold_and_invalid_annotations():
    with pytest.raises(ValueError): match_issues([], [], threshold=0)
    with pytest.raises(ValueError): validate_dataset({"schema_version": 1, "reference_diarization": [{"start": 5, "end": 1, "speaker": "A"}]})
    with pytest.raises(ValueError): validate_analysis({})


def test_rag_ranks_page_identity_and_missing_queries():
    expected = [{"designation": "СП TEST", "page": 7}]
    results = [{"designation": "СП TEST", "page": 8}, {"designation": "OTHER", "page": 7}, {"designation": "СП TEST", "page": 7}]
    result = retrieval_metrics([{"reference_id": "R1", "available": True, "expected": expected, "sources": results}])
    assert (result["hit_at_1"], result["hit_at_3"], result["hit_at_5"]) == (0, 1, 1)
    assert result["mrr"] == pytest.approx(1/3)


def test_live_rag_reuses_existing_prepare_without_llm(monkeypatch):
    from app.test_api_budget import FakeRetriever
    import rag.retriever
    monkeypatch.setattr(rag.retriever, "SemanticRetriever", lambda config: FakeRetriever())
    analysis = minimal_analysis()
    dataset = {"schema_version": 1, "reference_issues": [{"id": "R1", "start": 0, "end": 1,
               "question": "Какие требования?", "expected_sources": [{"designation": "СП TEST"}]}]}
    result = evaluate(dataset, analysis, live_retrieval=True)
    assert result["retrieval"]["hit_at_1"] == 1


def minimal_analysis():
    return {"transcript": [{"start": 0, "end": 1, "speaker": "S", "text": "Текст"}],
            "detected_issues": [], "statistics": {"timings": {"asr_seconds": 2, "diarization_seconds": 1, "total_seconds": 6},
               "llm_usage": {"llm_calls": 1, "run_llm_calls": 1}}, "audio_duration_sec": 10}


def test_default_evaluation_skips_ground_truth_and_exports(tmp_path):
    result = evaluate({"schema_version": 1}, minimal_analysis())
    assert result["asr"]["wer"] is None
    assert result["performance"]["asr_real_time_factor"] == .2
    assert result["performance"]["full_real_time_factor"] == .6
    save_results(result, tmp_path)
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "asr.csv").read_text(encoding="utf-8-sig").startswith("wer,cer,reason")


def test_resume_latency_not_claimed_as_full_rtf():
    analysis = minimal_analysis(); analysis["statistics"]["llm_usage"]["llm_calls"] = 3
    assert performance_metrics(analysis, 10, "fixture")["full_real_time_factor"] is None


def test_unexecuted_zero_stage_times_are_unavailable():
    analysis = minimal_analysis()
    analysis["statistics"]["timings"].update(asr_seconds=0, diarization_seconds=0)
    result = performance_metrics(analysis, 10, "fixture")
    assert result["asr_sec"] is None and result["diarization_sec"] is None
    assert result["full_real_time_factor"] is None
