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

from typing import List, Dict, Any, Optional, NamedTuple
import httpx

from ..config import settings


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
    ):
        self.api_key = api_key or settings.openai_api_key.get_secret_value()
        self.model = model or settings.openai_model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.timeout = timeout

        self._validate_config()

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
    ) -> Dict[str, Any]:
        """Construct a chat completion request payload."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "temperature": temperature,
        }

        if tools:
            payload["tools"] = tools

        return payload

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    async def chat(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
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
        LLMResponseError
        """
        payload = self._build_payload(system_prompt, messages, tools, temperature)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMTransportError(
                f"LLM transport failure: {type(exc).__name__}"
            ) from exc

        try:
            data = response.json()
        except Exception as exc:
            raise LLMResponseError("Failed to parse LLM JSON response.") from exc

        # -------------------------------------------------------------
        # Defensive response validation
        # -------------------------------------------------------------
        try:
            message = data["choices"][0]["message"]
        except Exception as exc:
            raise LLMResponseError(
                f"Unexpected LLM response structure: {data}"
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

        return ChatResult(message=message, usage=usage)

