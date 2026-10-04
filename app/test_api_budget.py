"""Offline regression tests: no network, audio models or embedding models."""

from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import httpx

from app.issue_detection import NormativeIssueDetector, StructuredDetectionError, IssueDetectionConfig
from app.meeting_analysis import MeetingAnalysisPipeline
from app.progress import read_input
from app.reporting import save_meeting_analysis
from app.transcript import TranscriptSegment
from llm.base import APIUnavailableError, HTTPDiagnostic, LLMClient, LLMCompletion, RateLimitError
from llm.config import QAConfig
from llm.groq_client import GroqClient
from llm.models import CitationValidationError, ContextValidationError
from llm.runtime import LLMBudgetExceeded, RunLLMClient, generate, pause_message
from llm.service import QuestionAnsweringService
from rag.models import SearchResult


class FakeClient(LLMClient):
    def __init__(self, responses: list[object]) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []

    @property
    def model(self) -> str:
        return "offline-test"

    def complete(self, **kwargs: object) -> LLMCompletion:
        self.prompts.append(str(kwargs["user_prompt"]))
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return LLMCompletion(str(value), self.model, 10, 5, 15)


class FakeRetriever:
    def __init__(self, text: str = "Нормативное требование к проектированию объекта.") -> None:
        self.calls = 0
        self.text = text

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        self.calls += 1
        return [SearchResult(.01, self.text, {
            "source_file": "test.pdf", "title": "Тестовый норматив", "designation": "СП TEST",
            "page": 1, "chunk_index": 0,
        })]


def fixture(count: int = 3) -> tuple[list[TranscriptSegment], str]:
    words = ["альфа", "бета", "гамма", "дельта", "эпсилон", "дзета", "эта", "тета"]
    transcript = [TranscriptSegment(i * 300, i * 300 + 2, "SPEAKER_00", f"Проверить объект {word}.")
                  for i, word in enumerate(words[:count])]
    payload = {"issues": [{
        "segment_ids": [f"SEGMENT_{i+1:04d}"],
        "normalized_question": f"Какие нормативные требования к объекту {word}?",
        "search_terms": [f"объект {word}"], "category": "other",
        "reason": "Необходима нормативная проверка объекта.",
    } for i, word in enumerate(words[:count])]}
    return transcript, json.dumps(payload, ensure_ascii=False)


def pipeline(client: LLMClient, retriever: FakeRetriever, **kwargs: object) -> MeetingAnalysisPipeline:
    return MeetingAnalysisPipeline(
        issue_detector=NormativeIssueDetector(llm_client=client),
        qa_service=QuestionAnsweringService(retriever=retriever, llm_client=client, config=QAConfig()),
        **kwargs,
    )


def limited(seconds: float = 1602) -> RateLimitError:
    return RateLimitError("rate limited", retry_after_seconds=seconds, diagnostic=HTTPDiagnostic(
        429, "Rate limit reached on tokens per day (TPD)", "tokens", "rate_limit_exceeded",
        "offline-test", "https://api.groq.com/openai/v1/chat/completions",
    ))


class APIBudgetTests(unittest.TestCase):
    def test_cli_transcript_then_resume_without_constructing_audio_pipeline(self) -> None:
        from app import analyze_meeting as cli
        transcript, payload = fixture(1)
        with tempfile.TemporaryDirectory() as directory:
            input_file = Path(directory) / "transcript.json"
            checkpoint = Path(directory) / "progress.json"
            input_file.write_text(json.dumps([asdict(s) for s in transcript], ensure_ascii=False), encoding="utf-8")
            stopped = FakeClient([payload, limited()])
            output = io.StringIO()
            with patch.object(cli, "load_dotenv"), patch.object(cli, "configure_utf8_console"), \
                 patch.object(cli.GroqClient, "from_env", return_value=stopped), \
                 patch.object(cli, "SemanticRetriever", return_value=FakeRetriever()), \
                 patch("app.pipeline.MeetingTranscriptionPipeline", side_effect=AssertionError("No audio initialization")), \
                 patch("sys.argv", ["analyze", "--transcript", str(input_file), "--checkpoint", str(checkpoint),
                                    "--results-dir", directory, "--debug"]), \
                 redirect_stdout(output), redirect_stderr(output):
                code = cli.main()
            self.assertEqual(code, 3)
            self.assertIn("26 мин 42 с", output.getvalue())
            self.assertIn("successful_llm_calls", output.getvalue())
            resumed = FakeClient(["Ответ [SOURCE_1]."])
            with patch.object(cli, "load_dotenv"), patch.object(cli, "configure_utf8_console"), \
                 patch.object(cli.GroqClient, "from_env", return_value=resumed), \
                 patch.object(cli, "SemanticRetriever", return_value=FakeRetriever()), \
                 patch("app.pipeline.MeetingTranscriptionPipeline", side_effect=AssertionError("No audio initialization")), \
                 patch("sys.argv", ["analyze", "--resume", str(checkpoint), "--results-dir", directory]), \
                 redirect_stdout(output), redirect_stderr(output):
                code = cli.main()
            self.assertEqual(code, 0)
            self.assertEqual(len(resumed.prompts), 1)

    def test_cli_existing_issues_reanswer_skips_detection(self) -> None:
        from app import analyze_meeting as cli
        transcript, payload = fixture(1)
        previous = pipeline(FakeClient([payload, "Ответ [SOURCE_1]."]), FakeRetriever()).analyze_transcript(transcript)
        with tempfile.TemporaryDirectory() as directory:
            input_file = Path(directory) / "analysis.json"
            input_file.write_text(json.dumps(previous.to_dict(), ensure_ascii=False), encoding="utf-8")
            client = FakeClient(["Обновлённый ответ [SOURCE_1]."])
            with patch.object(cli, "load_dotenv"), patch.object(cli, "configure_utf8_console"), \
                 patch.object(cli.GroqClient, "from_env", return_value=client), \
                 patch.object(cli, "SemanticRetriever", return_value=FakeRetriever()), \
                 patch("sys.argv", ["analyze", "--issues", str(input_file), "--reanswer", "--results-dir", directory]), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = cli.main()
            self.assertEqual(code, 0)
            self.assertEqual(len(client.prompts), 1)
            self.assertNotIn("Стенограмма для анализа", client.prompts[0])

    def test_budget_one_prevents_corrective_call_but_can_resume_it(self) -> None:
        transcript, payload = fixture(1)
        client = FakeClient(["{broken", payload])
        runner = pipeline(client, FakeRetriever(), max_llm_calls=1)
        first = runner.analyze_transcript(transcript)
        self.assertEqual(len(client.prompts), 1)
        self.assertEqual(first.run_status, "paused")
        resumed = FakeClient([payload, "Ответ [SOURCE_1]."])
        final = pipeline(resumed, FakeRetriever(), state=runner.state).analyze_transcript(transcript)
        self.assertEqual(len(resumed.prompts), 2)
        self.assertEqual(final.statistics.llm_usage["detection_tokens"]["total_tokens"], 30)
        self.assertEqual(final.statistics.llm_usage["corrective_retries"], 1)

    def test_repeated_resume_after_terminal_validation_does_not_generate(self) -> None:
        transcript, payload = fixture(1)
        runner = pipeline(FakeClient(["{broken", "{broken"]), FakeRetriever())
        result = runner.analyze_transcript(transcript)
        self.assertEqual(result.run_status, "validation_error")
        client = FakeClient([])
        result = pipeline(client, FakeRetriever(), state=runner.state).analyze_transcript(transcript)
        self.assertEqual(result.run_status, "validation_error")
        self.assertEqual(client.prompts, [])

    def test_normal_eight_issues_use_nine_calls(self) -> None:
        transcript, payload = fixture(8)
        client = FakeClient([payload, *["Требование найдено [SOURCE_1]."] * 8])
        result = pipeline(client, FakeRetriever()).analyze_transcript(transcript)
        self.assertEqual(result.statistics.detected_issues, 8)
        self.assertEqual(len(client.prompts), 9)
        self.assertEqual(result.statistics.llm_usage["max_llm_calls_per_run"], 18)
        self.assertEqual(result.statistics.llm_usage["corrective_retries"], 0)

    def test_combined_detection_errors_have_one_corrective_retry(self) -> None:
        transcript, valid = fixture(1)
        invalid = json.loads(valid)
        invalid["issues"][0]["normalized_question"] = "What requirements apply to object alpha?"
        invalid["issues"][0]["segment_ids"] = ["SEGMENT_9999"]
        client = FakeClient([json.dumps(invalid), valid])
        detector = NormativeIssueDetector(llm_client=client)
        detector.detect(transcript)
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("not in Russian", client.prompts[1])
        self.assertIn("segment_ids", client.prompts[1])

    def test_json_then_language_error_cannot_trigger_third_call(self) -> None:
        transcript, valid = fixture(1)
        invalid = json.loads(valid)
        invalid["issues"][0]["normalized_question"] = "What are the requirements for this object?"
        client = FakeClient(["{broken", json.dumps(invalid), valid])
        with self.assertRaises(StructuredDetectionError):
            NormativeIssueDetector(llm_client=client).detect(transcript)
        self.assertEqual(len(client.prompts), 2)

    def test_two_invalid_answers_do_not_restart_answer_pipeline(self) -> None:
        transcript, payload = fixture(1)
        client = FakeClient([payload, "Ответ [SOURCE_99]", "Ответ [SOURCE_99]", "unused"])
        runner = pipeline(client, FakeRetriever())
        result = runner.analyze_transcript(transcript)
        self.assertEqual(len(client.prompts), 3)
        self.assertEqual(result.detected_issues[0].status, "processing_error")
        self.assertEqual(result.statistics.total_tokens, 45)
        self.assertEqual(result.statistics.llm_usage["answer_tokens"]["total_tokens"], 30)
        self.assertEqual(result.statistics.llm_usage["retry_tokens"]["total_tokens"], 15)

    def test_missing_normative_value_does_not_call_again(self) -> None:
        transcript, payload = fixture(1)
        client = FakeClient([payload, "Минимум 0,9 м [SOURCE_1].", "unused"])
        result = pipeline(client, FakeRetriever()).analyze_transcript(transcript)
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("ContextValidationError", result.detected_issues[0].error)
        self.assertEqual(result.statistics.total_tokens, 30)

    def test_429_stops_all_calls_and_resume_reuses_work(self) -> None:
        transcript, payload = fixture(3)
        client = FakeClient([payload, "Ответ [SOURCE_1].", limited(), "unused"])
        retriever = FakeRetriever()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            runner = pipeline(client, retriever, checkpoint_path=path)
            with patch("time.sleep", side_effect=AssertionError("CLI must not sleep")):
                partial = runner.analyze_transcript(transcript)
            self.assertEqual(len(client.prompts), 3)
            self.assertEqual(partial.statistics.pending_issues, 2)
            self.assertIn("26 мин 42 с", partial.stop_reason)
            self.assertIn("TPD", partial.stop_reason)
            self.assertEqual(partial.statistics.total_tokens, 30)
            state = read_input(path, mode="resume")
            self.assertIn("ISSUE_002", state["prepared"])
            self.assertNotIn("ISSUE_003", state["prepared"])
            resumed_client = FakeClient(["Ответ [SOURCE_1].", "Ответ [SOURCE_1]."])
            resumed_retriever = FakeRetriever()
            result = pipeline(resumed_client, resumed_retriever, checkpoint_path=path, state=state).analyze_transcript(transcript)
            self.assertEqual(result.run_status, "completed")
            self.assertEqual(len(resumed_client.prompts), 2)
            # Only the last issue needs retrieval; issue 2 reuses saved chunks.
            expected = FakeRetriever()
            probe = QuestionAnsweringService(retriever=expected, llm_client=FakeClient([]), config=QAConfig())
            issue3 = state["detected_issues"][2]
            probe.prepare(issue3["normalized_question"], search_terms=issue3["search_terms"])
            self.assertEqual(resumed_retriever.calls, expected.calls)
            self.assertEqual(result.statistics.total_tokens, 60)
            self.assertEqual(result.statistics.llm_usage["successful_llm_calls"], 4)
            self.assertEqual(result.statistics.llm_usage["llm_calls"], 5)
            paths = save_meeting_analysis(result, directory)
            self.assertTrue(Path(paths.report).is_file())

    def test_short_429_also_pauses_without_generation_retry(self) -> None:
        transcript, payload = fixture(1)
        client = FakeClient([payload, limited(2), "unused"])
        result = pipeline(client, FakeRetriever()).analyze_transcript(transcript)
        self.assertEqual(len(client.prompts), 2)
        self.assertEqual(result.run_status, "paused")

    def test_429_during_detection_preserves_transcript(self) -> None:
        transcript, payload = fixture(1)
        client = FakeClient([limited(), "unused"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            result = pipeline(client, FakeRetriever(), checkpoint_path=path).analyze_transcript(transcript)
            self.assertEqual(result.run_status, "paused")
            state = read_input(path, mode="resume")
            self.assertEqual(state["transcript"], [asdict(s) for s in transcript])
            paths = save_meeting_analysis(result, directory)
            self.assertIn("ещё не завершён", Path(paths.report).read_text(encoding="utf-8"))
            resumed = FakeClient([payload, "Ответ [SOURCE_1]."])
            final = pipeline(resumed, FakeRetriever(), state=state).analyze_transcript(transcript)
            self.assertEqual(final.run_status, "completed")

    def test_network_failure_pauses_instead_of_retrying_all_issues(self) -> None:
        transcript, payload = fixture(3)
        client = FakeClient([payload, APIUnavailableError("offline"), "unused"])
        result = pipeline(client, FakeRetriever()).analyze_transcript(transcript)
        self.assertEqual(len(client.prompts), 2)
        self.assertEqual(result.statistics.pending_issues, 3)

    def test_budget_blocks_request_and_resume_keeps_first_answer(self) -> None:
        transcript, payload = fixture(3)
        client = FakeClient([payload, "Ответ [SOURCE_1].", "unused"])
        runner = pipeline(client, FakeRetriever(), max_llm_calls=2)
        result = runner.analyze_transcript(transcript)
        self.assertEqual(len(client.prompts), 2)
        self.assertEqual(result.statistics.pending_issues, 2)
        self.assertIn("Лимит LLM-вызовов", result.stop_reason)
        resumed = FakeClient(["Ответ [SOURCE_1].", "Ответ [SOURCE_1]."])
        final = pipeline(resumed, FakeRetriever(), state=runner.state, max_llm_calls=4).analyze_transcript(transcript)
        self.assertEqual(len(resumed.prompts), 2)
        self.assertEqual(final.statistics.total_tokens, 60)

    def test_paid_invalid_response_replayed_before_corrective_resume(self) -> None:
        transcript, payload = fixture(1)
        client = FakeClient([payload, "Ответ [SOURCE_99].", limited(), "unused"])
        runner = pipeline(client, FakeRetriever())
        partial = runner.analyze_transcript(transcript)
        self.assertEqual(partial.statistics.total_tokens, 30)
        resumed = FakeClient(["Исправленный ответ [SOURCE_1]."])
        final = pipeline(resumed, FakeRetriever(), state=runner.state).analyze_transcript(transcript)
        self.assertEqual(len(resumed.prompts), 1)
        self.assertEqual(final.statistics.total_tokens, 45)
        self.assertEqual(final.run_status, "completed")

    def test_completed_detection_window_not_regenerated_on_resume(self) -> None:
        transcript, _ = fixture(3)
        first = FakeClient(['{"issues":[]}', limited()])
        detector = NormativeIssueDetector(llm_client=first, config=IssueDetectionConfig(window_segments=2, overlap_segments=0))
        qa = QuestionAnsweringService(retriever=FakeRetriever(), llm_client=first, config=QAConfig())
        runner = MeetingAnalysisPipeline(issue_detector=detector, qa_service=qa)
        runner.analyze_transcript(transcript)
        self.assertIn("0", runner.state["detection_progress"])
        resumed = FakeClient(['{"issues":[]}'])
        next_detector = NormativeIssueDetector(llm_client=resumed, config=detector.config)
        qa = QuestionAnsweringService(retriever=FakeRetriever(), llm_client=resumed, config=QAConfig())
        final = MeetingAnalysisPipeline(issue_detector=next_detector, qa_service=qa, state=runner.state).analyze_transcript(transcript)
        self.assertEqual(len(resumed.prompts), 1)
        self.assertEqual(final.run_status, "completed")

    def test_http_200_empty_response_tokens_are_counted(self) -> None:
        posts = []
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": "offline-test"}]})
            posts.append(request)
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "" if len(posts) == 1 else "Ответ [SOURCE_1]."}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            })
        provider = GroqClient(api_key="offline-dummy", model="offline-test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        tracked = RunLLMClient(provider, max_calls=2)
        service = QuestionAnsweringService(retriever=FakeRetriever(), llm_client=tracked, config=QAConfig())
        answer = service.answer("Какие требования к объекту?")
        self.assertEqual(answer.total_tokens, 240)
        self.assertEqual(tracked.metrics()["total_tokens"], 240)
        self.assertEqual(tracked.metrics()["successful_llm_calls"], 2)
        self.assertEqual(len(posts), 2)

    def test_import_legacy_analysis_skips_completed_answers(self) -> None:
        transcript, payload = fixture(1)
        client = FakeClient([payload, "Ответ [SOURCE_1]."])
        completed = pipeline(client, FakeRetriever()).analyze_transcript(transcript)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.json"
            path.write_text(json.dumps(completed.to_dict(), ensure_ascii=False), encoding="utf-8")
            state = read_input(path, mode="issues")
            provider = FakeClient([])
            result = pipeline(provider, FakeRetriever(), state=state).analyze_transcript(transcript)
            self.assertEqual(provider.prompts, [])
            self.assertEqual(result.run_status, "completed")


if __name__ == "__main__":
    unittest.main()
