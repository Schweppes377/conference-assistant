"""Structured records produced by offline meeting analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.transcript import TranscriptSegment
from llm.models import SourceEvidence


STATUS_REQUIREMENT_FOUND = "requirement_found"
STATUS_PARTIAL_INFORMATION = "partial_information"
STATUS_INSUFFICIENT_INFORMATION = "insufficient_information"
STATUS_POTENTIAL_NONCOMPLIANCE = "potential_noncompliance"
STATUS_PROCESSING_ERROR = "processing_error"


@dataclass(frozen=True, slots=True)
class DetectedIssue:
    """One normative issue tied to verified transcript segments."""

    id: str
    start: float
    end: float
    speakers: list[str]
    segment_ids: list[str]
    original_text: str
    normalized_question: str
    search_terms: list[str]
    category: str
    reason: str


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    """Aggregate diagnostics for structured issue detection."""

    windows: int
    llm_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Validated, deduplicated issues and detector diagnostics."""

    issues: list[DetectedIssue]
    metrics: DetectionMetrics


@dataclass(frozen=True, slots=True)
class AnalyzedIssue:
    """A detected issue enriched with grounded RAG evidence and an answer."""

    id: str
    start: float
    end: float
    speakers: list[str]
    segment_ids: list[str]
    original_text: str
    normalized_question: str
    search_terms: list[str]
    category: str
    reason: str
    answer: str
    status: str
    cited_sources: list[SourceEvidence] = field(default_factory=list)
    retrieved_candidates: list[SourceEvidence] = field(default_factory=list)
    error: str | None = None
    retrieval_seconds: float = 0.0
    llm_seconds: float = 0.0
    total_seconds: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def sources(self) -> list[SourceEvidence]:
        """Backward-compatible alias for cited sources."""

        return self.cited_sources

    @property
    def retrieval_results(self) -> list[SourceEvidence]:
        """Backward-compatible alias for reranked candidates."""

        return self.retrieved_candidates


@dataclass(frozen=True, slots=True)
class PipelineTimings:
    """Latency breakdown for one complete meeting run."""

    audio_decode_seconds: float
    asr_seconds: float
    diarization_seconds: float
    detection_llm_seconds: float
    retrieval_seconds: float
    answer_llm_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class MeetingStatistics:
    """Counts and metrics suitable for later evaluation."""

    transcript_segments: int
    speakers: list[str]
    detected_issues: int
    requirement_found: int
    partial_information: int
    insufficient_information: int
    potential_noncompliance: int
    processing_errors: int
    detection_windows: int
    timings: PipelineTimings
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class MeetingAnalysis:
    """Complete machine-readable analysis of one audio meeting."""

    audio_file: str
    created_at: str
    transcript: list[TranscriptSegment]
    detected_issues: list[AnalyzedIssue]
    statistics: MeetingStatistics

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnalysisOutputPaths:
    """Files created for one analysis run."""

    transcript: str
    analysis: str
    report: str
