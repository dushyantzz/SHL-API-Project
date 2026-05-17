# Interview Preparation Guide — SHL Assessment Agent

This document covers questions a recruiter or technical interviewer may ask about this project, organized by category.

---

## 1. System Design (HLD)

### Q: Walk me through the high-level architecture.

The system is a stateless REST API built with FastAPI. Every `POST /chat` request carries the full conversation history. The request flows through three stages:

1. **Validation** — Pydantic models enforce schema rules (alternating roles, non-empty content, turn cap).
2. **Retrieval** — A lightweight LLM call extracts search intent from the conversation. The extracted query feeds into a TF-IDF index over 377 SHL products, returning the top-20 most relevant candidates. Any specific product names mentioned are also matched via substring lookup.
3. **Generation** — The system prompt, retrieved catalog subset, and full conversation history are sent to OpenAI. The LLM returns a structured JSON response with a reply, optional recommendations, and an end-of-conversation flag. Every recommended product is cross-validated against the catalog to prevent hallucinated names or URLs.

### Q: Why did you choose a two-phase (retrieval + generation) architecture?

Sending all 377 products to the LLM on every request would consume ~20K tokens of context per call. The retrieval phase narrows this to ~20 relevant products (~4K tokens), reducing cost by ~80% per request. It also improves response quality by surfacing only contextually relevant assessments.

### Q: How would this scale to 10,000 products?

Three changes:
1. Replace TF-IDF with a vector database (Pinecone, Qdrant, or pgvector) using OpenAI embeddings for semantic search.
2. Add metadata pre-filtering at the database level (keys, job_levels, languages) before semantic ranking.
3. Increase `top_k` and consider a re-ranking step using a cross-encoder model.

### Q: Why is the API stateless?

The spec requires it — every request carries the full conversation history. This eliminates server-side session management, simplifies horizontal scaling (any replica can handle any request), and avoids stale-state bugs. The trade-off is slightly larger request payloads, which is negligible for text conversations.

### Q: How would you add authentication?

Add an API key or JWT bearer token middleware in FastAPI. For multi-tenant deployments, tie the token to an organization ID and log it alongside request metrics for usage billing.

---

## 2. Low-Level Design (LLD)

### Q: Explain the CatalogStore design.

`CatalogStore` is a dataclass that loads the SHL product catalog JSON at application startup and builds three indexes:

1. **ID index** — `dict[str, Assessment]` for O(1) lookup by `entity_id`.
2. **Name index** — `dict[str, Assessment]` with lowercased keys for case-insensitive name lookup.
3. **TF-IDF matrix** — `TfidfVectorizer` fit on `name + description + keys` text. Query-time cosine similarity returns ranked results.

It exposes `search_by_text()`, `filter_assessments()`, and a hybrid `search()` that intersects TF-IDF ranking with structured filters (keys, job_levels, languages, duration, adaptive, remote).

### Q: Why TF-IDF instead of embeddings?

For 377 products, TF-IDF offers several advantages:
- **Zero external API cost** — no embedding API calls.
- **Zero latency overhead** — computation is local, sub-millisecond.
- **High precision on keyword queries** — "Java developer" directly matches "Java" in product names/descriptions.
- **No cold-start** — no need to pre-compute and store embedding vectors.

Embeddings would add value at scale (10K+ products) or for nuanced semantic queries, but for this catalog size, TF-IDF is the pragmatic choice.

### Q: How does the recommendation validation work?

After the LLM returns recommendations, `_validate_recommendation()` in `AgentOrchestrator`:
1. Looks up the product name in the catalog (exact match, then substring match).
2. If found, substitutes the canonical `url` and `test_type` from the catalog — even if the LLM got them slightly wrong.
3. If not found, the recommendation is silently dropped.

This prevents hallucinated product names or URLs from reaching the client.

### Q: How do you derive `test_type` from the catalog?

The catalog has a `keys` field (array of strings like `"Knowledge & Skills"`, `"Personality & Behavior"`). A static mapping converts each key to a single-letter code: K, P, A, C, D, S, B. Products with multiple keys get comma-separated codes (e.g., `"A,S"` for Ability + Situational Judgment).

### Q: Explain the retry logic in LLMService.

The service retries on two specific exceptions:
- `APIConnectionError` — network failures.
- `RateLimitError` — OpenAI rate limit (429).

It uses exponential backoff: 1s, 2s, 4s. Non-retryable errors (400, 401, 403) are raised immediately. After 3 failures, a `RuntimeError` is raised, caught by the agent orchestrator's fallback logic.

### Q: Why structured output (`response_format=json_object`)?

It guarantees the LLM returns valid JSON, eliminating regex-based parsing of free-text responses. Combined with the Pydantic `ChatResponse` model, it creates a two-layer validation: the LLM guarantees JSON syntax, and Pydantic guarantees schema compliance.

---

## 3. Prompt Engineering

### Q: How does the system prompt enforce scope?

The prompt has explicit "SCOPE RULES (non-negotiable)" that:
1. Restrict the agent to SHL assessment products only.
2. Instruct it to refuse legal, salary, and general hiring advice with a standard redirect phrase.
3. Instruct it to ignore prompt-injection attempts with a standard deflection.
4. Mandate that every URL comes from the provided catalog.

### Q: How does the prompt handle the four conversational behaviors?

Each behavior (Clarify, Recommend, Refine, Compare) is documented as a named section with clear trigger conditions and expected output. The prompt specifies:
- **Clarify**: Ask one focused question; set `recommendations: null`.
- **Recommend**: Return 1-10 products with exact catalog data.
- **Refine**: Narrate the delta; don't start over.
- **Compare**: Use only catalog data; don't invent differentiators.

### Q: What is the query extraction prompt for?

It's a lightweight LLM call (Phase 1) that distils a multi-turn conversation into concise search terms. For example, a 4-turn conversation about "graduate finance analysts who need numerical reasoning" gets distilled to `"graduate financial analyst numerical reasoning knowledge test"`. This becomes the TF-IDF search query.

---

## 4. API Design & FastAPI

### Q: Why FastAPI over Flask or Django?

- **Async-native** — `await` on OpenAI API calls without blocking the event loop.
- **Pydantic integration** — request/response validation with zero boilerplate.
- **Auto-generated OpenAPI docs** — `/docs` endpoint for free.
- **Lifespan management** — clean startup (load catalog, init services) and shutdown (close HTTP clients).
- **Performance** — FastAPI on uvicorn handles significantly more concurrent requests than Flask.

### Q: How does the lifespan handler work?

The `lifespan` async context manager runs once at startup and shutdown:
- **Startup**: Loads the catalog JSON, builds TF-IDF index, initializes the OpenAI client and agent orchestrator.
- **Shutdown**: Closes the async OpenAI HTTP client cleanly.

This ensures the catalog is loaded once (not per-request) and services are properly cleaned up.

### Q: How do you handle errors?

Three layers:
1. **Pydantic validation** — returns 422 for malformed requests (wrong role order, empty messages, etc.).
2. **Agent fallback** — if the LLM call fails, the agent returns a graceful fallback message instead of crashing.
3. **Global exception handlers** — FastAPI catches `ValueError` (422) and unhandled `Exception` (500) with structured error responses.

### Q: Why allow CORS from all origins?

This is configured for development convenience. In production, you would restrict `allow_origins` to specific frontend domains. The middleware is there to demonstrate awareness of CORS as a requirement for browser-based clients.

---

## 5. Data Engineering

### Q: How is the catalog structured?

The catalog is a JSON array of 377 product objects. Each has 14 fields: `entity_id`, `name`, `link`, `description`, `keys` (taxonomy tags), `job_levels`, `languages`, `duration`, `duration_raw`, `remote`, `adaptive`, `status`, `scraped_at`, and raw variants of list fields.

### Q: What data quality issues did you handle?

- **Duration parsing**: `duration_raw` contains strings like `"Approximate Completion Time in minutes = 30"`. A regex extracts the numeric value. `"Untimed"` and empty strings map to `None`.
- **Boolean-as-string**: `remote` and `adaptive` are `"yes"`/`"no"` strings, not booleans. The `_parse_product()` method normalizes them.
- **Empty arrays**: `languages` and `job_levels` can be empty `[]`. Filters treat empty as "not specified" rather than "no match".
- **Multi-key products**: Products can have multiple `keys` (e.g., `["Ability & Aptitude", "Competencies"]`). The test_type derives from all keys.

### Q: How would you keep the catalog up to date?

Add a scraping pipeline that periodically fetches the SHL product catalog pages, compares with the existing JSON, and updates the file. The lifespan handler could watch for file changes, or you could expose an admin endpoint that triggers a catalog reload.

---

## 6. Testing Strategy

### Q: How do you test without making real API calls?

The `conftest.py` fixture patches `LLMService.complete_json` with an `AsyncMock` that returns a controlled JSON response. This lets all chat endpoint tests run without an OpenAI API key, making them fast, deterministic, and CI-friendly.

### Q: What test categories do you have?

1. **Unit tests** — `test_catalog.py`: Duration parsing, test_type derivation, catalog loading, ID/name lookup, TF-IDF search, structured filtering, hybrid search.
2. **Integration tests** — `test_chat.py`: Valid single/multi-turn requests, response schema compliance, validation rejection for empty messages, wrong role order, consecutive roles, empty content, invalid roles.
3. **Smoke tests** — `test_health.py`: Health endpoint returns 200 with correct body.

### Q: What would you add with more time?

- End-to-end tests with real LLM calls (marked as slow, skipped in CI).
- Conversation replay tests using the 10 sample conversations as golden fixtures.
- Load/stress testing with locust to verify concurrent request handling.
- Contract tests to ensure the response schema matches the evaluation harness.

---

## 7. DevOps & Deployment

### Q: Walk me through the Dockerfile.

Multi-step build on `python:3.12-slim`:
1. Create a non-root `appuser` for security.
2. Copy and install `requirements.txt` first (leveraging Docker layer caching).
3. Copy application code and catalog.
4. Switch to `appuser`.
5. Built-in `HEALTHCHECK` pings `GET /health` every 30 seconds.
6. Entry point: `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

### Q: How would you deploy this to production?

1. **Container orchestration** — Kubernetes or AWS ECS with the Docker image.
2. **Load balancer** — ALB/Nginx in front with health check on `GET /health`.
3. **Secrets** — OpenAI API key via Kubernetes Secrets or AWS Secrets Manager, not environment files.
4. **Monitoring** — Prometheus metrics endpoint + Grafana dashboards for latency, error rate, token usage.
5. **Rate limiting** — API gateway or middleware to prevent abuse.
6. **Auto-scaling** — Scale on CPU/request-count metrics since the service is stateless.

---

## 8. Trade-offs & Alternatives

### Q: What are the main trade-offs in your design?

| Trade-off | Chose | Alternative | Why |
|-----------|-------|-------------|-----|
| Search | TF-IDF | Embeddings | Zero cost, sufficient for 377 products |
| LLM calls per request | 2 (extract + generate) | 1 (all-in-one) | Better retrieval quality; each call is focused |
| Catalog loading | Startup / in-memory | Database | Fast reads, no infra dependency for 377 items |
| Validation | Post-hoc (validate after LLM) | Constrained decoding | Simpler, works with any LLM provider |
| State | Stateless (client sends history) | Server-side sessions | Spec requirement; simpler scaling |

### Q: What would break if the catalog grew to 100K products?

1. In-memory TF-IDF wouldn't fit comfortably — switch to a vector database.
2. The full catalog summary wouldn't fit in the system prompt — use strict retrieval.
3. Name lookups would need indexing (already O(1) via dict, but substring search is O(n)).
4. Startup time would increase — consider lazy loading or background indexing.

### Q: Why not use LangChain or similar frameworks?

For this scope, LangChain adds abstraction without proportionate value. The two-phase pipeline is ~150 lines of code. Adding LangChain would introduce:
- Dependency complexity (LangChain has 50+ transitive deps).
- Abstraction layers that obscure the retrieval and generation logic.
- Harder debugging when the chain misbehaves.

For a production system at this scale, explicit code is easier to maintain and reason about.

---

## 9. Behavioral / Situational Questions

### Q: How did you ensure the agent doesn't hallucinate product names?

Three layers:
1. The system prompt explicitly forbids fabricating products.
2. Only catalog-sourced products are injected into the LLM context.
3. Post-generation validation cross-checks every recommended name against the catalog and drops unrecognized products.

### Q: How does the agent handle prompt injection?

The system prompt includes explicit instructions to ignore "ignore previous instructions" and similar patterns. The scope guard redirects to a standard response. Additionally, the structured output format (`response_format=json_object`) limits the LLM's ability to produce arbitrary output.

### Q: What was the hardest part of this project?

Designing the prompt to reliably produce all four behaviors (clarify, recommend, refine, compare) while staying grounded in catalog data. The challenge is balancing specificity (clear rules for each behavior) with flexibility (handling diverse user inputs). The post-validation layer was added as a safety net after observing that even well-prompted models occasionally hallucinate product names.

### Q: If you had two more weeks, what would you add?

1. **Conversation replay testing** — Replay all 10 sample conversations against the live agent and measure Recall@10.
2. **Streaming responses** — Use SSE for real-time token streaming to reduce perceived latency.
3. **Observability** — Structured logging with correlation IDs, OpenTelemetry tracing, token usage tracking.
4. **Caching** — Cache LLM responses for identical conversation histories (deterministic temperature=0 calls).
5. **Frontend** — A simple React chat UI to demonstrate the full experience.
