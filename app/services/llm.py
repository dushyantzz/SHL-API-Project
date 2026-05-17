"""Async Google Gemini client with structured JSON output and retry logic."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


class LLMService:
    """
    Thin async wrapper around the Gemini generateContent API.

    Uses JSON response mode and exponential backoff on transient failures.
    """

    def __init__(self, *, api_key: str, model: str = "gemini-3.1-flash-lite") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def complete_json(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """
        Send a chat completion request and parse the JSON response.

        Uses response_mime_type=application/json for structured output.
        Retries on transient errors with exponential backoff.
        """
        contents = [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages
        ]

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )

        last_exception: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                content = response.text or "{}"
                return json.loads(content)

            except (ConnectionError, TimeoutError, OSError) as exc:
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

            except json.JSONDecodeError as exc:
                logger.error("LLM returned invalid JSON: %s", exc)
                raise

            except Exception as exc:
                if _is_retryable(exc):
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
                else:
                    logger.error("LLM API error (non-retryable): %s", exc)
                    raise

        raise RuntimeError(
            f"LLM request failed after {_MAX_RETRIES} retries: {last_exception}"
        )

    async def close(self) -> None:
        aclose = getattr(self._client, "aclose", None)
        if callable(aclose):
            await aclose()


def _is_retryable(exc: Exception) -> bool:
    """Treat rate limits and 5xx-style API errors as retryable."""
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return (
        "rate" in name
        or "quota" in message
        or "429" in message
        or "503" in message
        or "unavailable" in message
    )
