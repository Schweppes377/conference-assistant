from __future__ import annotations

import os
import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import httpx

from llm.base import (
    APIUnavailableError,
    AccessDeniedError,
    EmptyResponseError,
    MissingAPIKeyError,
    ModelUnavailableError,
    RateLimitError,
)
from llm.ask import _print_http_diagnostic
from llm.groq_client import GroqClient


class GroqClientTests(unittest.TestCase):
    def test_completion_and_model_validation(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(
                    200,
                    json={"data": [{"id": "openai/gpt-oss-120b"}]},
                )
            return httpx.Response(
                200,
                json={
                    "model": "openai/gpt-oss-120b",
                    "choices": [{"message": {"content": "Ответ [SOURCE_1]"}}],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "total_tokens": 110,
                    },
                },
            )

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        client = GroqClient(
            api_key="test-key",
            model="openai/gpt-oss-120b",
            http_client=http_client,
        )

        completion = client.complete(
            system_prompt="system",
            user_prompt="question",
            temperature=0.1,
            max_tokens=100,
        )

        self.assertEqual(completion.content, "Ответ [SOURCE_1]")
        self.assertEqual(completion.total_tokens, 110)

    def test_missing_key(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with self.assertRaises(MissingAPIKeyError):
                GroqClient.from_env(model="openai/gpt-oss-120b")

    def test_unavailable_model_is_not_replaced(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": [{"id": "another-model"}]})
        )
        client = GroqClient(
            api_key="test-key",
            model="missing-model",
            http_client=httpx.Client(transport=transport),
        )

        with self.assertRaises(ModelUnavailableError):
            client.validate_model()

    def test_rate_limit_has_clear_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "test-model"}]})
            return httpx.Response(429, headers={"retry-after": "30"})

        client = GroqClient(
            api_key="test-key",
            model="test-model",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        with self.assertRaisesRegex(RateLimitError, "30") as caught:
            client.complete(
                system_prompt="system",
                user_prompt="question",
                temperature=0.1,
                max_tokens=100,
            )
        self.assertEqual(caught.exception.retry_after_seconds, 30.0)

    def test_empty_answer_is_rejected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "test-model"}]})
            return httpx.Response(
                200,
                json={"model": "test-model", "choices": [{"message": {"content": ""}}]},
            )

        client = GroqClient(
            api_key="test-key",
            model="test-model",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with self.assertRaises(EmptyResponseError):
            client.complete(
                system_prompt="system",
                user_prompt="question",
                temperature=0.1,
                max_tokens=100,
            )

    def test_proxy_initialization_error_is_normalized(self) -> None:
        client = GroqClient(api_key="test-key", model="test-model")
        with patch("llm.groq_client.httpx.Client", side_effect=ImportError("socksio missing")):
            with self.assertRaisesRegex(APIUnavailableError, "proxy"):
                client.list_available_models()

    def test_http_403_exposes_only_safe_debug_details(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                403,
                json={
                    "error": {
                        "message": "Access denied for this request",
                        "type": "access_denied_error",
                        "code": "region_blocked",
                    }
                },
            )
        )
        client = GroqClient(
            api_key="secret-test-key",
            model="openai/gpt-oss-120b",
            http_client=httpx.Client(transport=transport),
        )

        with self.assertRaises(AccessDeniedError) as caught:
            client.list_available_models()

        diagnostic = caught.exception.diagnostic
        self.assertIsNotNone(diagnostic)
        assert diagnostic is not None
        self.assertEqual(diagnostic.status, 403)
        self.assertEqual(diagnostic.message, "Access denied for this request")
        self.assertEqual(diagnostic.error_type, "access_denied_error")
        self.assertEqual(diagnostic.error_code, "region_blocked")
        self.assertEqual(diagnostic.model, "openai/gpt-oss-120b")
        self.assertEqual(diagnostic.endpoint, "https://api.groq.com/openai/v1/models")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            _print_http_diagnostic(diagnostic)
        output = stderr.getvalue()
        self.assertIn("HTTP status: 403", output)
        self.assertIn("Error message: Access denied for this request", output)
        self.assertIn("Error type: access_denied_error", output)
        self.assertIn("Error code: region_blocked", output)
        self.assertIn("Model: openai/gpt-oss-120b", output)
        self.assertIn("Endpoint: https://api.groq.com/openai/v1/models", output)
        self.assertNotIn("secret-test-key", output)
        self.assertNotIn("Authorization", output)

    def test_transient_connect_error_is_retried(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("temporary handshake failure", request=request)
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})

        client = GroqClient(
            api_key="test-key",
            model="test-model",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            max_retries=1,
            retry_delay_seconds=0,
        )

        self.assertEqual(client.list_available_models(), ["test-model"])
        self.assertEqual(attempts, 2)


if __name__ == "__main__":
    unittest.main()
