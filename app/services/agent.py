"""Agent orchestrator: two-phase retrieval + generation pipeline."""

from __future__ import annotations

import logging
from typing import Any

from app.models.request import ChatRequest
from app.models.response import ChatResponse, Recommendation
from app.prompts.templates import build_system_prompt, build_query_extraction_prompt
from app.services.catalog import Assessment, CatalogStore
from app.services.llm import LLMService

logger = logging.getLogger(__name__)

_TOP_K_RETRIEVAL = 20
_FALLBACK_REPLY = (
    "I'm sorry, I encountered an issue processing your request. "
    "Could you please rephrase or provide more details about the "
    "SHL assessments you're looking for?"
)


class AgentOrchestrator:
    """
    Two-phase pipeline for each /chat request:

    Phase 1 - Retrieval: extract search intent from conversation history,
    query the TF-IDF catalog index, and collect relevant assessments.

    Phase 2 - Generation: inject retrieved assessments into the system prompt
    and call the LLM for a grounded, structured response.
    """

    def __init__(
        self,
        catalog: CatalogStore,
        llm: LLMService,
        max_turns: int = 8,
        max_recommendations: int = 10,
    ) -> None:
        self._catalog = catalog
        self._llm = llm
        self._max_turns = max_turns
        self._max_recommendations = max_recommendations

    async def process(self, request: ChatRequest) -> ChatResponse:
        """Execute the full retrieval-augmented generation pipeline."""
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        try:
            relevant = await self._retrieve(messages)
            return await self._generate(messages, relevant)
        except Exception:
            logger.exception("Agent pipeline failed")
            return ChatResponse(
                reply=_FALLBACK_REPLY,
                recommendations=None,
                end_of_conversation=False,
            )

    # ------------------------------------------------------------------
    # Phase 1: Retrieval
    # ------------------------------------------------------------------

    async def _retrieve(self, messages: list[dict[str, str]]) -> list[Assessment]:
        """
        Extract search intent from conversation and query the catalog.

        Uses a lightweight LLM call to distil the conversation into search
        terms, then runs TF-IDF + name lookup against the catalog.
        """
        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )

        try:
            extraction = await self._llm.complete_json(
                system_prompt="You are a search-query extractor. Return JSON only.",
                messages=[
                    {
                        "role": "user",
                        "content": build_query_extraction_prompt(conversation_text),
                    }
                ],
                temperature=0.0,
                max_tokens=256,
            )
            query = extraction.get("query", "")
            product_names: list[str] = extraction.get("product_names", [])
        except Exception:
            logger.warning("Query extraction failed; falling back to last user message")
            query = messages[-1]["content"] if messages else ""
            product_names = []

        results: list[Assessment] = []
        seen_ids: set[str] = set()

        for name in product_names:
            for match in self._catalog.find_by_name_substring(name):
                if match.entity_id not in seen_ids:
                    results.append(match)
                    seen_ids.add(match.entity_id)

        if query:
            for a in self._catalog.search_by_text(query, top_k=_TOP_K_RETRIEVAL):
                if a.entity_id not in seen_ids:
                    results.append(a)
                    seen_ids.add(a.entity_id)

        if len(results) < 5:
            for a in self._catalog.assessments:
                if a.entity_id not in seen_ids:
                    results.append(a)
                    seen_ids.add(a.entity_id)
                if len(results) >= _TOP_K_RETRIEVAL:
                    break

        logger.info("Retrieved %d candidate assessments", len(results))
        return results

    # ------------------------------------------------------------------
    # Phase 2: Generation
    # ------------------------------------------------------------------

    async def _generate(
        self,
        messages: list[dict[str, str]],
        candidates: list[Assessment],
    ) -> ChatResponse:
        """Send conversation + catalog context to LLM for structured response."""
        catalog_context = self._catalog.get_assessments_context(candidates)
        system_prompt = build_system_prompt(catalog_context, self._max_turns)

        raw: dict[str, Any] = await self._llm.complete_json(
            system_prompt=system_prompt,
            messages=messages,
            temperature=0.3,
            max_tokens=2048,
        )

        return self._parse_response(raw)

    def _parse_response(self, raw: dict[str, Any]) -> ChatResponse:
        """Validate and sanitize the LLM's JSON output."""
        reply = raw.get("reply", _FALLBACK_REPLY)
        end_of_conversation = bool(raw.get("end_of_conversation", False))

        raw_recs = raw.get("recommendations")
        recommendations: list[Recommendation] | None = None

        if raw_recs and isinstance(raw_recs, list):
            validated: list[Recommendation] = []
            for item in raw_recs[: self._max_recommendations]:
                rec = self._validate_recommendation(item)
                if rec is not None:
                    validated.append(rec)
            recommendations = validated if validated else None

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=end_of_conversation,
        )

    def _validate_recommendation(self, item: dict[str, Any]) -> Recommendation | None:
        """
        Ensure every recommendation references a real catalog product.

        Cross-checks name against the catalog and substitutes the
        canonical URL and test_type to prevent hallucination.
        """
        name = item.get("name", "")
        if not name:
            return None

        catalog_match = self._catalog.get_by_name(name)
        if catalog_match is None:
            matches = self._catalog.find_by_name_substring(name)
            catalog_match = matches[0] if matches else None

        if catalog_match is None:
            logger.warning("LLM recommended unknown product: %s — skipping", name)
            return None

        return Recommendation(
            name=catalog_match.name,
            url=catalog_match.url,  # type: ignore[arg-type]
            test_type=catalog_match.test_type,
        )
