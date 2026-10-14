"""Structured evidence, answer, and experiment metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rag.models import SearchResult


class SourceMetadataError(ValueError):
    """Raised when a retrieved chunk has incomplete source metadata."""


class CitationValidationError(ValueError):
    """Raised when the LLM cites a source ID absent from supplied context."""


class ContextValidationError(CitationValidationError):
    """Required evidence is absent; identical context cannot fix this locally."""


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """Auditable provenance, not a certificate of semantic/legal correctness."""

    version: int
    mode: str
    conclusion: str
    project_facts: list[dict[str, Any]]
    requirements: list[dict[str, Any]]
    comparisons: list[dict[str, Any]]
    unknowns: list[str]
    limitations: list[str]
    compliance_confirmed: bool = False
    semantic_validation: str = "not_performed"
    proposed_conclusion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    """A retrieved chunk tied to trustworthy ChromaDB metadata."""

    source_id: str
    document: str
    designation: str
    page: int
    chunk_id: str
    text: str
    score: float
    distance: float
    rerank_score: float | None = None
    matched_terms: list[str] | None = None
    provenance: list[dict[str, Any]] | None = None
    related_to: str | None = None

    @classmethod
    def from_search_result(
        cls,
        index: int,
        result: SearchResult,
        *,
        rerank_score: float | None = None,
        matched_terms: list[str] | None = None,
    ) -> "SourceEvidence":
        metadata = result.metadata
        document = metadata.get("title")
        page = metadata.get("page")
        chunk_index = metadata.get("chunk_index")
        source_file = metadata.get("source_file")

        if not isinstance(document, str) or not document.strip():
            raise SourceMetadataError("Retrieved chunk has no valid document title")
        try:
            page_number = int(page)
            chunk_number = int(chunk_index)
        except (TypeError, ValueError) as exc:
            raise SourceMetadataError(
                f"Retrieved chunk from {document!r} has invalid page or chunk index"
            ) from exc
        if page_number < 1 or chunk_number < 0:
            raise SourceMetadataError(
                f"Retrieved chunk from {document!r} has invalid page or chunk index"
            )
        if not isinstance(source_file, str) or not source_file.strip():
            raise SourceMetadataError(f"Retrieved chunk from {document!r} has no source file")
        if not result.text.strip():
            raise SourceMetadataError(f"Retrieved chunk from {document!r} has empty text")

        designation = metadata.get("designation")
        return cls(
            source_id=f"SOURCE_{index}",
            document=document.strip(),
            designation=designation.strip() if isinstance(designation, str) else "",
            page=page_number,
            chunk_id=f"{source_file}:p{page_number}:c{chunk_number}",
            text=result.text.strip(),
            score=result.score,
            distance=result.distance,
            rerank_score=rerank_score,
            matched_terms=matched_terms or [],
            provenance=metadata.get('_provenance'),
            related_to=metadata.get('_related_to'),
        )


@dataclass(frozen=True, slots=True)
class QATimings:
    """Latency values suitable for later diploma experiments."""

    retrieval_seconds: float
    llm_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class PreparedQuestion:
    """Exact retrieved context saved before generation for cheap resume."""

    question: str
    validation_input: str
    retrieved_candidates: list[SourceEvidence]
    context_sources: list[SourceEvidence]
    retrieval_seconds: float
    retrieval_trace: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreparedQuestion":
        for raw in [*data["retrieved_candidates"], *data["context_sources"]]:
            if (not isinstance(raw.get("page"), int) or raw["page"] < 1
                    or not all(isinstance(raw.get(name), str) and raw[name].strip()
                               for name in ("source_id", "document", "chunk_id", "text"))):
                raise SourceMetadataError("Saved retrieval has invalid source metadata")
        available = {s["source_id"]: s for s in data["retrieved_candidates"]}
        if len(available) != len(data["retrieved_candidates"]):
            raise SourceMetadataError("Saved retrieval contains duplicate source IDs")
        if any(available.get(s["source_id"]) != s for s in data["context_sources"]):
            raise SourceMetadataError("Saved context does not match retrieved candidates")
        return cls(
            question=data["question"], validation_input=data["validation_input"],
            retrieved_candidates=[SourceEvidence(**s) for s in data["retrieved_candidates"]],
            context_sources=[SourceEvidence(**s) for s in data["context_sources"]],
            retrieval_seconds=data["retrieval_seconds"],
            retrieval_trace=data.get('retrieval_trace'),
        )


@dataclass(frozen=True, slots=True)
class QAResult:
    """Structured grounded answer plus evidence and diagnostics."""

    question: str
    answer: str
    cited_sources: list[SourceEvidence]
    retrieved_candidates: list[SourceEvidence]
    model: str
    top_k: int
    candidate_k: int
    max_context_chars: int
    context_characters: int
    temperature: float
    insufficient_context: bool
    partial_context: bool
    timings: QATimings
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    validation: dict[str, Any] | None = None
    retrieval_trace: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable experiment record."""

        return asdict(self)

    @property
    def sources(self) -> list[SourceEvidence]:
        """Backward-compatible alias for verified cited sources."""

        return self.cited_sources

    @property
    def retrieval_results(self) -> list[SourceEvidence]:
        """Backward-compatible alias for reranked retrieval candidates."""

        return self.retrieved_candidates
