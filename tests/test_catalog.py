"""Tests for the CatalogStore service."""

from __future__ import annotations

from app.services.catalog import CatalogStore, _derive_test_type, _parse_duration_minutes


class TestDurationParsing:
    def test_numeric_minutes(self) -> None:
        assert _parse_duration_minutes("Approximate Completion Time in minutes = 30") == 30

    def test_plain_minutes(self) -> None:
        assert _parse_duration_minutes("25 minutes") == 25

    def test_empty_string(self) -> None:
        assert _parse_duration_minutes("") is None

    def test_untimed(self) -> None:
        assert _parse_duration_minutes("Untimed") is None


class TestTestTypeDerivation:
    def test_knowledge_skills(self) -> None:
        assert _derive_test_type(["Knowledge & Skills"]) == "K"

    def test_multiple_keys(self) -> None:
        codes = _derive_test_type(["Personality & Behavior", "Competencies"])
        assert set(codes.split(",")) == {"C", "P"}

    def test_empty_keys(self) -> None:
        assert _derive_test_type([]) == "K"

    def test_unknown_key(self) -> None:
        assert _derive_test_type(["Some Unknown Key"]) == "K"


class TestCatalogLoading:
    def test_loads_all_assessments(self, catalog: CatalogStore) -> None:
        assert len(catalog.assessments) > 0

    def test_all_have_names(self, catalog: CatalogStore) -> None:
        for a in catalog.assessments:
            assert a.name, f"Assessment {a.entity_id} has no name"

    def test_all_have_urls(self, catalog: CatalogStore) -> None:
        for a in catalog.assessments:
            assert a.url.startswith("https://www.shl.com/")

    def test_all_have_test_type(self, catalog: CatalogStore) -> None:
        for a in catalog.assessments:
            assert a.test_type, f"Assessment {a.name} has no test_type"


class TestCatalogLookup:
    def test_get_by_id(self, catalog: CatalogStore) -> None:
        result = catalog.get_by_id("3827")
        assert result is not None
        assert result.name == ".NET Framework 4.5"

    def test_get_by_id_missing(self, catalog: CatalogStore) -> None:
        assert catalog.get_by_id("99999") is None

    def test_get_by_name(self, catalog: CatalogStore) -> None:
        result = catalog.get_by_name(".NET Framework 4.5")
        assert result is not None
        assert result.entity_id == "3827"

    def test_get_by_name_case_insensitive(self, catalog: CatalogStore) -> None:
        result = catalog.get_by_name(".net framework 4.5")
        assert result is not None

    def test_find_by_name_substring(self, catalog: CatalogStore) -> None:
        results = catalog.find_by_name_substring("OPQ")
        assert len(results) > 0
        assert all("opq" in a.name.lower() for a in results)


class TestCatalogSearch:
    def test_text_search_returns_results(self, catalog: CatalogStore) -> None:
        results = catalog.search_by_text("Java programming")
        assert len(results) > 0

    def test_text_search_relevance(self, catalog: CatalogStore) -> None:
        results = catalog.search_by_text("OPQ personality questionnaire")
        assert len(results) > 0
        found = any("opq" in a.name.lower() for a in results[:5])
        assert found, "Expected an OPQ product in the top-5 results"

    def test_filter_by_adaptive(self, catalog: CatalogStore) -> None:
        results = catalog.filter_assessments(adaptive_only=True)
        assert all(a.adaptive for a in results)

    def test_filter_by_keys(self, catalog: CatalogStore) -> None:
        results = catalog.filter_assessments(keys=["Knowledge & Skills"])
        assert len(results) > 0
        for a in results:
            assert "Knowledge & Skills" in a.keys

    def test_combined_search(self, catalog: CatalogStore) -> None:
        results = catalog.search(
            "programming developer",
            keys=["Knowledge & Skills"],
            top_k=5,
        )
        assert len(results) <= 5
        for a in results:
            assert "Knowledge & Skills" in a.keys

    def test_search_empty_query(self, catalog: CatalogStore) -> None:
        results = catalog.search_by_text("")
        assert isinstance(results, list)
