"""Shared fixtures for the test suite."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.services.catalog import CatalogStore


@pytest.fixture(scope="session")
def catalog_path() -> Path:
    return Path(__file__).resolve().parent.parent / "shl_product_catalog.json"


@pytest.fixture(scope="session")
def catalog(catalog_path: Path) -> CatalogStore:
    store = CatalogStore()
    store.load(catalog_path)
    return store


@pytest.fixture()
def client():
    """TestClient with mocked LLM to avoid real API calls in CI."""
    os.environ["AZURE_OPENAI_API_KEY"] = "test-azure-key"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com/"
    os.environ["AZURE_OPENAI_API_VERSION"] = "2024-08-01-preview"
    os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4o-mini"

    mock_response = {
        "reply": "Here are some assessments for Java developers.",
        "recommendations": [
            {
                "name": ".NET Framework 4.5",
                "url": "https://www.shl.com/products/product-catalog/view/net-framework-4-5/",
                "test_type": "K",
            }
        ],
        "end_of_conversation": False,
    }

    with patch(
        "app.services.llm.LLMService.complete_json",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        from app.config import get_settings
        get_settings.cache_clear()

        from app.main import app
        with TestClient(app) as tc:
            yield tc

    get_settings.cache_clear()
