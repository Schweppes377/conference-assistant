"""Offline provider and checkpoint migration tests; fixtures contain no real keys."""

import json
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import httpx
from gigachat.exceptions import ResponseError

from app.progress import check_llm_identity
from llm.base import (AccessDeniedError, AuthenticationError, APIUnavailableError,
                      EmptyResponseError, MissingAPIKeyError, ModelUnavailableError, RateLimitError)
from llm.config import QAConfig
from llm.gigachat_client import GigaChatClient
from llm.runtime import RunLLMClient, generate


def response(text="работает", usage=True):
    return NS(model="GigaChat-3-Ultra", choices=[NS(message=NS(content=text))],
              usage=NS(prompt_tokens=10, completion_tokens=5, total_tokens=15) if usage else None)


class GigaChatClientTests(unittest.TestCase):
    def client(self, sdk):
        return GigaChatClient(credentials="dummy-test-credential", model="GigaChat-3-Ultra", sdk_client=sdk)

    def kwargs(self):
        return dict(system_prompt="Система", user_prompt="Вопрос", temperature=.1, max_tokens=100)

    def test_roles_model_and_usage(self):
        sdk = Mock(); sdk.chat.return_value = response()
        result = self.client(sdk).complete(**self.kwargs())
        payload = sdk.chat.call_args.args[0]
        self.assertEqual([m.role for m in payload.messages], ["system", "user"])
        self.assertEqual([m.content for m in payload.messages], ["Система", "Вопрос"])
        self.assertEqual(payload.model, "GigaChat-3-Ultra")
        self.assertEqual(payload.max_tokens, 100)
        self.assertEqual(result.total_tokens, 15)
        self.assertEqual(sdk.chat.call_count, 1)

    def test_structured_detection_both_attempts(self):
        sdk = Mock(); sdk.chat.return_value = response('{"issues":[]}')
        tracked = RunLLMClient(self.client(sdk), max_calls=2)
        for attempt in range(2):
            completion = generate(tracked, stage="detection", operation="0", attempt=attempt, **self.kwargs())
            self.assertEqual(json.loads(completion.content), {"issues": []})
            schema = sdk.chat.call_args.args[0].response_format
            self.assertTrue(schema.strict)
            self.assertEqual(schema.schema_["required"], ["issues"])
        self.assertEqual(tracked.metrics()["corrective_retries"], 1)
        self.assertEqual(tracked.metrics()["total_tokens"], 30)

    def test_missing_usage_marked_incomplete(self):
        sdk = Mock(); sdk.chat.return_value = response(usage=False)
        tracked = RunLLMClient(self.client(sdk))
        self.assertIsNone(tracked.complete(**self.kwargs()).total_tokens)
        self.assertTrue(tracked.metrics()["usage_incomplete"])

    def test_empty_response_retains_usage(self):
        sdk = Mock(); sdk.chat.return_value = response("")
        with self.assertRaises(EmptyResponseError) as error:
            self.client(sdk).complete(**self.kwargs())
        self.assertEqual(error.exception.completion.total_tokens, 15)

    def test_env_only_and_no_sdk_retries(self):
        with patch.dict(os.environ, {"SBER_ULTRA": "dummy-env-credential"}), \
             patch("llm.gigachat_client.GigaChat") as factory:
            GigaChatClient.from_env(model=QAConfig().model)
            kwargs = factory.call_args.kwargs
            self.assertEqual(kwargs["credentials"], "dummy-env-credential")
            self.assertEqual(kwargs["scope"], "GIGACHAT_API_PERS")
            self.assertEqual(kwargs["base_url"], "https://api.giga.chat/v1")
            self.assertEqual(kwargs["max_retries"], 0)
        with patch.dict(os.environ, {"SBER_ULTRA": ""}):
            with self.assertRaises(MissingAPIKeyError):
                GigaChatClient.from_env(model=QAConfig().model)

    def test_http_mapping_redaction_and_no_retry(self):
        for status, expected in [(400, ModelUnavailableError), (401, AuthenticationError),
                                 (403, AccessDeniedError), (429, RateLimitError), (503, APIUnavailableError)]:
            with self.subTest(status=status):
                sdk = Mock()
                sdk.chat.side_effect = ResponseError("https://api.giga.chat/v1/chat/completions", status,
                    json.dumps({"message": "model error dummy-test-credential Bearer private-token"}).encode(),
                    httpx.Headers({"retry-after": "1602", "Authorization": "private-header"}))
                with self.assertRaises(expected) as caught:
                    self.client(sdk).complete(**self.kwargs())
                diagnostic = caught.exception.diagnostic
                self.assertEqual(diagnostic.status, status)
                self.assertNotIn("dummy-test-credential", diagnostic.message)
                self.assertNotIn("private-token", diagnostic.message)
                self.assertNotIn("private-header", str(caught.exception))
                self.assertEqual(sdk.chat.call_count, 1)
                if status == 429:
                    self.assertEqual(caught.exception.retry_after_seconds, 1602)

    def test_network_exception_does_not_expose_secrets(self):
        sdk = Mock(); sdk.chat.side_effect = httpx.ConnectError("dummy-test-credential")
        with self.assertRaises(APIUnavailableError) as caught:
            self.client(sdk).complete(**self.kwargs())
        self.assertNotIn("dummy-test-credential", str(caught.exception))

    def test_official_sdk_wire_v1_schema_and_no_429_retry(self):
        from gigachat import GigaChat
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(429, json={"message": "temporary restriction"},
                                  headers={"retry-after": "1602"})
        sdk = GigaChat(base_url=GigaChatClient.BASE_URL, access_token="offline-fake-token",
                       credentials=None, max_retries=0, model="GigaChat-3-Ultra")
        sdk._client_instance = httpx.Client(base_url=GigaChatClient.BASE_URL,
                                          transport=httpx.MockTransport(handler))
        try:
            with self.assertRaises(RateLimitError):
                self.client(sdk).complete_detection(**self.kwargs())
            self.assertEqual(len(requests), 1)
            self.assertEqual(str(requests[0].url), "https://api.giga.chat/v1/chat/completions")
            payload = json.loads(requests[0].content)
            self.assertTrue(payload["response_format"]["strict"])
            self.assertEqual(payload["response_format"]["type"], "json_schema")
        finally:
            sdk.close()

    def test_foreign_and_legacy_checkpoint_rejected_without_mutation(self):
        for identity in ({}, {"llm_provider": "previous-provider", "llm_model": "previous-model"}):
            state = {**identity, "answers": {"ISSUE_001": {"answer": "saved"}}}
            original = json.dumps(state)
            with self.assertRaisesRegex(ValueError, "--transcript"):
                check_llm_identity(state, provider="gigachat", model=QAConfig().model)
            self.assertEqual(json.dumps(state), original)

    def test_foreign_successful_detection_cache_rejected(self):
        with self.assertRaises(ValueError):
            check_llm_identity({"llm_runtime": {"responses": {"detection:0:0": {}}}},
                               provider="gigachat", model=QAConfig().model)

    def test_same_identity_resumes_and_empty_checkpoint_adopts(self):
        for state in ({}, {"llm_provider": "gigachat", "llm_model": QAConfig().model, "answers": {"x": {}}}):
            check_llm_identity(state, provider="gigachat", model=QAConfig().model)
            self.assertEqual(state["llm_model"], "GigaChat-3-Ultra")


if __name__ == "__main__":
    unittest.main()
