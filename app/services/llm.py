"""Async Azure OpenAI client with structured JSON output and retry logic."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncAzureOpenAI,
    RateLimitError,
)

logger = logging.getLogger(__name__)

_RETRYABLE = (APIConnectionError, RateLimitError)
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


class LLMService:
    """
    Thin async wrapper around Azure OpenAI chat completions.

    Uses the deployment name as the model identifier. Provides structured JSON
    output via response_format and exponential backoff on transient failures.
    """

    def __init__(
        self,
        *,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        deployment: str,
    ) -> None:
        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint.rstrip("/"),
            api_version=api_version,
        )
        self._deployment = deployment

    async def complete_json(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Send a chat completion request and parse the JSON response.

        Uses response_format=json_object to guarantee valid JSON output.
        Retries on transient network / rate-limit errors with exponential backoff.
        """
        import asyncio

        all_messages = [{"role": "system", "content": system_prompt}] + messages

        last_exception: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.chat.completions.create(
                    model=self._deployment,
                    messages=all_messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content or "{}"
                return json.loads(content)

            except _RETRYABLE as exc:
                last_exception = exc
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    "LLM request failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    _MAX_RETRIES,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)

            except APIStatusError as exc:
                logger.error("LLM API error (non-retryable): %s", exc)
                raise

        raise RuntimeError(
            f"LLM request failed after {_MAX_RETRIES} retries: {last_exception}"
        )

    async def close(self) -> None:
        await self._client.close()
