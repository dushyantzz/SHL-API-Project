"""FastAPI application for the SHL Assessment Recommendation Agent."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.config import get_settings
from app.models.request import ChatRequest
from app.models.response import ChatResponse
from app.services.agent import AgentOrchestrator
from app.services.catalog import CatalogStore
from app.services.llm import LLMService

logger = logging.getLogger(__name__)

catalog_store = CatalogStore()
llm_service: LLMService | None = None
agent: AgentOrchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Pre-load catalog and initialise services on startup; clean up on shutdown."""
    global llm_service, agent  # noqa: PLW0603

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    catalog_store.load(settings.catalog_path)
    logger.info(
        "Catalog loaded: %d assessments", len(catalog_store.assessments)
    )

    llm_service = LLMService(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
    )

    agent = AgentOrchestrator(
        catalog=catalog_store,
        llm=llm_service,
        max_turns=settings.max_turns,
        max_recommendations=settings.max_recommendations,
    )

    logger.info("Agent ready (model=%s)", settings.gemini_model)
    yield

    if llm_service:
        await llm_service.close()
    logger.info("Shutdown complete")


app = FastAPI(
    title="SHL Assessment Recommendation Agent",
    description=(
        "A conversational agent that recommends SHL assessments based on "
        "multi-turn dialogue. Supports clarification, recommendation, "
        "refinement, and comparison of assessment products."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception handlers ───────────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect browser visits to interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Readiness probe — returns 200 when the service is ready to serve."""
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a stateless conversation and return the agent's next reply.

    The full conversation history must be provided in every request.
    When the agent has enough context, the response includes a structured
    list of SHL assessment recommendations.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")

    return await agent.process(request)
