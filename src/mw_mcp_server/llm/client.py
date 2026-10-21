"""
LLM Client

This module provides a thin, testable, and production-safe client wrapper
around the OpenAI Chat Completions API (or compatible providers).

Design Goals
------------
- Async-first
- Fully testable (no hard-coded globals in logic paths)
- No silent network failures
- Clear separation between:
    - Payload construction
    - Transport
    - Response validation
- Easy future migration to:
    - Azure OpenAI
    - OpenAI Responses API
    - Self-hosted LLMs
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import List, Dict, Any, Optional, NamedTuple
import httpx

from ..config import settings

logger = logging.getLogger("mcp.llm")

# HTTP statuses worth retrying: rate limiting and provider-side failures.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


# ---------------------------------------------------------------------
# Response Types
# ---------------------------------------------------------------------

class TokenUsage(NamedTuple):
    """Token usage from an LLM response."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResult(NamedTuple):
    """Result from an LLM chat completion."""
    message: Dict[str, Any]
    usage: TokenUsage
    # "stop", "tool_calls", "length" (output truncated), "content_filter", ...
    finish_reason: Optional[str] = None


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------

class LLMClientError(RuntimeError):
    """Base class for all LLM client errors."""


class LLMTransportError(LLMClientError):
    """Raised when the HTTP request fails."""


class LLMResponseError(LLMClientError):
    """Raised when the LLM returns an invalid or unexpected response."""


class LLMConfigurationError(LLMClientError):
    """Raised when API configuration is invalid."""


class LLMContextLengthError(LLMTransportError):
    """Raised when the provider rejects the request as exceeding the context window."""


# ---------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------

class LLMClient:
    """
    Async client wrapper for LLM chat completions.

    This client is intentionally stateless aside from the API key,
    which allows safe reuse across requests and clean test mocking.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: Optional[int] = None,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 20.0,
    ):
        self.api_key = api_key or settings.openai_api_key.get_secret_value()
        self.model = model or settings.openai_model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.timeout = timeout
        self.max_retries = (
            max_retries if max_retries is not None
            else int(getattr(settings, "llm_max_retries", 3))
        )
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self._client: Optional[httpx.AsyncClient] = None

        self._validate_config()

    def _get_http_client(self) -> httpx.AsyncClient:
        """Lazily create a long-lived AsyncClient for connection pooling."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=30.0,
                ),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _validate_config(self) -> None:
        """Validate required configuration before any network calls."""
        if not self.api_key:
            raise LLMConfigurationError("Missing OpenAI API key.")
        if not self.model:
            raise LLMConfigurationError("Missing OpenAI model configuration.")
        if not self.base_url:
            raise LLMConfigurationError("Missing OpenAI base URL.")

    def _build_payload(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Construct a chat completion request payload."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "temperature": temperature,
        }

        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        if max_tokens:
            payload["max_completion_tokens"] = max_tokens

        return payload

    def _retry_delay(self, attempt: int, response: Optional[httpx.Response]) -> float:
        """Exponential backoff with jitter, honouring a short Retry-After."""
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return min(self.retry_max_delay, max(0.0, float(retry_after)))
                except ValueError:
                    pass
        delay = min(self.retry_max_delay, self.retry_base_delay * (2 ** attempt))
        return delay * random.uniform(0.5, 1.0)

    @staticmethod
    def _is_context_length_error(response: httpx.Response) -> bool:
        if response.status_code != 400:
            return False
        try:
            error = response.json().get("error") or {}
        except Exception:
            return False
        return error.get("code") == "context_length_exceeded" or (
            "context length" in str(error.get("message", "")).lower()
        )

    async def _post_with_retries(
        self, url: str, payload: Dict[str, Any], headers: Dict[str, str]
    ) -> httpx.Response:
        """POST, retrying timeouts/connection errors and 429/5xx (bounded)."""
        attempt = 0
        while True:
            response: Optional[httpx.Response] = None
            try:
                response = await self._get_http_client().post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if self._is_context_length_error(exc.response):
                    raise LLMContextLengthError(
                        "LLM request exceeds the model's context window."
                    ) from exc
                retryable = status in _RETRYABLE_STATUS
                failure = exc
            except httpx.TransportError as exc:
                # Timeouts, connection resets, DNS failures.
                retryable = True
                failure = exc
            except httpx.HTTPError as exc:
                retryable = False
                failure = exc

            if not retryable or attempt >= self.max_retries:
                raise LLMTransportError(
                    f"LLM transport failure: {type(failure).__name__}"
                    + (f" (HTTP {response.status_code})" if response is not None else "")
                ) from failure

            delay = self._retry_delay(attempt, response)
            attempt += 1
            logger.warning(
                "LLM request failed (%s%s); retry %d/%d in %.1fs",
                type(failure).__name__,
                f" HTTP {response.status_code}" if response is not None else "",
                attempt,
                self.max_retries,
                delay,
            )
            await asyncio.sleep(delay)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    async def chat(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResult:
        """
        Execute a chat completion request.

        Parameters
        ----------
        system_prompt : str
            The system-level instruction prompt.

        messages : List[Dict[str, Any]]
            Prior conversation messages in OpenAI-compatible format.

        tools : Optional[List[Dict[str, Any]]]
            Tool/function definitions for the LLM.

        temperature : float
            Sampling temperature.

        tool_choice : Optional[str]
            OpenAI ``tool_choice`` ("auto", "none", "required"); only sent
            when tools are provided.

        max_tokens : Optional[int]
            Cap on generated tokens (sent as ``max_completion_tokens``).

        Retries timeouts, connection errors and HTTP 408/409/429/5xx with
        bounded exponential backoff (``llm_max_retries``, default 3).

        Returns
        -------
        ChatResult
            Named tuple containing:
            - message: The raw assistant message object
            - usage: TokenUsage with prompt/completion/total tokens

        Raises
        ------
        LLMConfigurationError
        LLMTransportError
        LLMContextLengthError
        LLMResponseError
        """
        payload = self._build_payload(
            system_prompt, messages, tools, temperature,
            tool_choice=tool_choice, max_tokens=max_tokens,
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"

        response = await self._post_with_retries(url, payload, headers)

        try:
            data = response.json()
        except Exception as exc:
            raise LLMResponseError("Failed to parse LLM JSON response.") from exc

        # -------------------------------------------------------------
        # Defensive response validation
        # -------------------------------------------------------------
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except Exception as exc:
            keys = sorted(data) if isinstance(data, dict) else type(data).__name__
            raise LLMResponseError(
                f"Unexpected LLM response structure (keys: {keys})"
            ) from exc

        if not isinstance(message, dict):
            raise LLMResponseError("LLM response message is not a dict.")

        if "role" not in message:
            raise LLMResponseError("LLM response missing 'role' field.")

        # Extract token usage from response
        usage_data = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        if finish_reason == "length":
            logger.warning(
                "LLM output truncated (finish_reason=length, completion_tokens=%d)",
                usage.completion_tokens,
            )

        return ChatResult(message=message, usage=usage, finish_reason=finish_reason)

