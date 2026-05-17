"""In-memory SHL product catalog with TF-IDF search and structured filtering."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

KEYS_TO_TYPE_CODE: dict[str, str] = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Competencies": "C",
    "Development & 360": "D",
    "Biodata & Situational Judgment": "B",
    "Assessment Exercises": "E",
    "Simulations": "S",
}

_DURATION_RE = re.compile(r"(\d+)\s*(?:minutes?)?", re.IGNORECASE)


def _parse_duration_minutes(raw: str) -> int | None:
    """Extract numeric minutes from duration_raw or duration strings."""
    if not raw:
        return None
    match = _DURATION_RE.search(raw)
    return int(match.group(1)) if match else None


def _derive_test_type(keys: list[str]) -> str:
    """Map catalog 'keys' list to comma-separated type codes."""
    codes = [KEYS_TO_TYPE_CODE[k] for k in keys if k in KEYS_TO_TYPE_CODE]
    return ",".join(sorted(set(codes))) if codes else "K"


@dataclass(frozen=True)
class Assessment:
    """Immutable representation of a single SHL catalog product."""

    entity_id: str
    name: str
    url: str
    description: str
    keys: tuple[str, ...]
    test_type: str
    job_levels: tuple[str, ...]
    languages: tuple[str, ...]
    duration_text: str
    duration_minutes: int | None
    remote: bool
    adaptive: bool

    def to_context_str(self) -> str:
        """Compact string representation for LLM context injection."""
        langs = ", ".join(self.languages[:5])
        if len(self.languages) > 5:
            langs += f" (+{len(self.languages) - 5} more)"
        return (
            f"[{self.entity_id}] {self.name}\n"
            f"  Type: {self.test_type} | Keys: {', '.join(self.keys)}\n"
            f"  Duration: {self.duration_text or 'N/A'} | "
            f"Adaptive: {'Yes' if self.adaptive else 'No'} | "
            f"Remote: {'Yes' if self.remote else 'No'}\n"
            f"  Levels: {', '.join(self.job_levels) or 'N/A'}\n"
            f"  Languages: {langs or 'N/A'}\n"
            f"  URL: {self.url}\n"
            f"  Description: {self.description[:300]}"
        )

    def to_summary_str(self) -> str:
        """One-line summary for catalog listing."""
        return (
            f"{self.name} | Type={self.test_type} | "
            f"Duration={self.duration_text or 'N/A'} | "
            f"Adaptive={'Yes' if self.adaptive else 'No'}"
        )


@dataclass
class CatalogStore:
    """
    Singleton catalog index providing text search and structured filtering.

    Loads the SHL product catalog JSON once, builds a TF-IDF index over
    concatenated name+description text, and provides multiple query methods.
    """

    assessments: list[Assessment] = field(default_factory=list)
    _id_index: dict[str, Assessment] = field(default_factory=dict)
    _name_index: dict[str, Assessment] = field(default_factory=dict)
    _tfidf_vectorizer: TfidfVectorizer | None = field(default=None, repr=False)
    _tfidf_matrix: np.ndarray | None = field(default=None, repr=False)

    # --- Loading ----------------------------------------------------------

    def load(self, catalog_path: Path) -> None:
        """Load products from JSON and build all indexes."""
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.assessments = [self._parse_product(p) for p in raw]
        self._build_indexes()
        logger.info("Loaded %d assessments from %s", len(self.assessments), catalog_path)

    @staticmethod
    def _parse_product(raw: dict) -> Assessment:
        return Assessment(
            entity_id=raw["entity_id"],
            name=raw["name"],
            url=raw["link"],
            description=raw.get("description", ""),
            keys=tuple(raw.get("keys", [])),
            test_type=_derive_test_type(raw.get("keys", [])),
            job_levels=tuple(raw.get("job_levels", [])),
            languages=tuple(raw.get("languages", [])),
            duration_text=raw.get("duration", ""),
            duration_minutes=_parse_duration_minutes(
                raw.get("duration_raw", "") or raw.get("duration", "")
            ),
            remote=raw.get("remote", "").lower() == "yes",
            adaptive=raw.get("adaptive", "").lower() == "yes",
        )

    def _build_indexes(self) -> None:
        """Build ID/name lookup dicts and TF-IDF matrix."""
        self._id_index = {a.entity_id: a for a in self.assessments}
        self._name_index = {a.name.lower(): a for a in self.assessments}

        corpus = [
            f"{a.name} {a.description} {' '.join(a.keys)}"
            for a in self.assessments
        ]
        self._tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)
        logger.info("TF-IDF index built: %s", self._tfidf_matrix.shape)

    # --- Lookup -----------------------------------------------------------

    def get_by_id(self, entity_id: str) -> Assessment | None:
        return self._id_index.get(entity_id)

    def get_by_name(self, name: str) -> Assessment | None:
        return self._name_index.get(name.lower())

    def find_by_name_substring(self, substring: str) -> list[Assessment]:
        """Case-insensitive substring search on product names."""
        lower = substring.lower()
        return [a for a in self.assessments if lower in a.name.lower()]

    # --- Text Search (TF-IDF) --------------------------------------------

    def search_by_text(self, query: str, top_k: int = 20) -> list[Assessment]:
        """Return top-k assessments ranked by TF-IDF cosine similarity."""
        if not self._tfidf_vectorizer or self._tfidf_matrix is None:
            return []
        query_vec = self._tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        ranked_indices = scores.argsort()[::-1][:top_k]
        return [self.assessments[i] for i in ranked_indices if scores[i] > 0.0]

    # --- Structured Filtering ---------------------------------------------

    def filter_assessments(
        self,
        *,
        keys: Sequence[str] | None = None,
        job_levels: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
        max_duration_minutes: int | None = None,
        adaptive_only: bool = False,
        remote_only: bool = False,
    ) -> list[Assessment]:
        """Filter assessments by structured criteria (AND logic)."""
        results = list(self.assessments)

        if keys:
            keys_lower = {k.lower() for k in keys}
            results = [
                a for a in results
                if keys_lower & {k.lower() for k in a.keys}
            ]

        if job_levels:
            levels_lower = {j.lower() for j in job_levels}
            results = [
                a for a in results
                if levels_lower & {j.lower() for j in a.job_levels}
            ]

        if languages:
            langs_lower = {la.lower() for la in languages}
            results = [
                a for a in results
                if langs_lower & {la.lower() for la in a.languages}
            ]

        if max_duration_minutes is not None:
            results = [
                a for a in results
                if a.duration_minutes is not None
                and a.duration_minutes <= max_duration_minutes
            ]

        if adaptive_only:
            results = [a for a in results if a.adaptive]

        if remote_only:
            results = [a for a in results if a.remote]

        return results

    # --- Combined Search --------------------------------------------------

    def search(
        self,
        query: str,
        *,
        keys: Sequence[str] | None = None,
        job_levels: Sequence[str] | None = None,
        languages: Sequence[str] | None = None,
        max_duration_minutes: int | None = None,
        adaptive_only: bool = False,
        remote_only: bool = False,
        top_k: int = 20,
    ) -> list[Assessment]:
        """
        Hybrid search: TF-IDF text relevance intersected with structured filters.

        Returns up to top_k results, preferring TF-IDF order for items that
        pass all structured filters.
        """
        text_ranked = self.search_by_text(query, top_k=len(self.assessments))
        if not any([keys, job_levels, languages, max_duration_minutes, adaptive_only, remote_only]):
            return text_ranked[:top_k]

        filtered_ids = {
            a.entity_id
            for a in self.filter_assessments(
                keys=keys,
                job_levels=job_levels,
                languages=languages,
                max_duration_minutes=max_duration_minutes,
                adaptive_only=adaptive_only,
                remote_only=remote_only,
            )
        }

        return [a for a in text_ranked if a.entity_id in filtered_ids][:top_k]

    # --- Catalog Summary for Prompt Injection -----------------------------

    def get_full_catalog_summary(self) -> str:
        """Compact catalog listing for system prompt context."""
        lines = [a.to_summary_str() for a in self.assessments]
        return "\n".join(lines)

    def get_assessments_context(self, assessments: list[Assessment]) -> str:
        """Detailed context block for a subset of assessments."""
        return "\n\n".join(a.to_context_str() for a in assessments)
