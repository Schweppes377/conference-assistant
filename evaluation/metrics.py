"""Reproducible offline metrics; missing references are never zero scores."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Lowercase and separate punctuation; keep digits, ё, and word differences."""
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text.lower())
    return " ".join(text.split())


def edit_counts(reference: list, hypothesis: list) -> dict:
    """Levenshtein alignment; deterministic ties: substitution, deletion, insertion."""
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, word in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, actual in enumerate(hypothesis, 1):
            cost, s, d, ins = previous[j - 1]
            sub = (cost + (word != actual), s + (word != actual), d, ins)
            cost, s, d, ins = previous[j]
            delete = (cost + 1, s, d + 1, ins)
            cost, s, d, ins = current[-1]
            insert = (cost + 1, s, d, ins + 1)
            current.append(min((sub, delete, insert), key=lambda x: x[0]))
        previous = current
    _, s, d, ins = previous[-1]
    return {"substitutions": s, "deletions": d, "insertions": ins}


def asr_metrics(reference: str | None, hypothesis: str) -> dict:
    if reference is None:
        return {"wer": None, "cer": None, "reason": "reference data not provided",
                "reference_word_count": None, "hypothesis_word_count": len(normalize_text(hypothesis).split()),
                "substitutions": None, "deletions": None, "insertions": None}
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    counts = edit_counts(ref.split(), hyp.split())
    # CER excludes whitespace; punctuation policy is identical to WER.
    ref_chars, hyp_chars = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    chars = edit_counts(ref_chars, hyp_chars)
    return {"wer": sum(counts.values()) / len(ref.split()) if ref.split() else None,
            "cer": sum(chars.values()) / len(ref_chars) if ref_chars else None,
            "reference_word_count": len(ref.split()), "hypothesis_word_count": len(hyp.split()),
            **counts, "character_errors": chars,
            "reason": None if ref.split() else "empty reference: denominator is zero"}


def diarization_metrics(reference: list | None, hypothesis: list | None,
                        *, collar: float = .25, skip_overlap: bool = False,
                        evaluation_regions: list | None = None) -> dict:
    if reference is None or hypothesis is None:
        return {"der": None, "reason": "reference data not provided" if reference is None
                else "raw diarization hypothesis not saved (transcript labels are not DER annotations)"}
    if not reference:
        return {"der": None, "reason": "empty reference: denominator is zero"}
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate
    def annotation(turns):
        result = Annotation()
        for index, turn in enumerate(turns):
            result[Segment(turn["start"], turn["end"]), index] = turn["speaker"]
        return result
    uem = Timeline([Segment(r["start"], r["end"]) for r in evaluation_regions]) if evaluation_regions else None
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    details = metric(annotation(reference), annotation(hypothesis), uem=uem, detailed=True)
    der = details.get("diarization error rate") if details["total"] > 0 else None
    return {"der": der, "collar_sec": collar, "skip_overlap": skip_overlap,
            "details": {k: float(v) for k, v in details.items()},
            "reason": None if der is not None else "no scored reference speech after collar/UEM"}


def match_issues(reference: list, predicted: list, *, mode: str = "time", threshold: float = .5) -> list[tuple[int, int]]:
    """Maximum-cardinality 1:1 matching using time IoU or segment-ID Jaccard.

    This measures localization, not semantic correctness of the generated question.
    No LLM or embeddings are involved. Threshold is fixed before evaluation.
    """
    if mode not in ("time", "segments") or not 0 < threshold <= 1:
        raise ValueError("Invalid matching mode/threshold")
    edges = []
    for ref in reference:
        candidates = []
        for index, pred in enumerate(predicted):
            if mode == "segments":
                a, b = set(ref.get("segment_ids", [])), set(pred.get("segment_ids", []))
                score = len(a & b) / len(a | b) if a | b else 0
            else:
                intersection = max(0, min(ref["end"], pred["end"]) - max(ref["start"], pred["start"]))
                union = (ref["end"] - ref["start"]) + (pred["end"] - pred["start"]) - intersection
                score = intersection / union if union > 0 else 0
            if score >= threshold:
                candidates.append((score, index))
        edges.append([index for _, index in sorted(candidates, key=lambda x: (-x[0], x[1]))])
    assigned = {}
    def augment(ref_index, seen):
        for pred_index in edges[ref_index]:
            if pred_index in seen:
                continue
            seen.add(pred_index)
            if pred_index not in assigned or augment(assigned[pred_index], seen):
                assigned[pred_index] = ref_index
                return True
        return False
    for index in range(len(reference)):
        augment(index, set())
    return sorted((ref, pred) for pred, ref in assigned.items())


def detection_metrics(reference: list | None, predicted: list, **matching) -> dict:
    if reference is None:
        return {"tp": None, "fp": None, "fn": None, "precision": None,
                "recall": None, "f1": None, "reason": "reference data not provided"}
    pairs = match_issues(reference, predicted, **matching)
    tp, fp, fn = len(pairs), len(predicted) - len(pairs), len(reference) - len(pairs)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "matches": pairs, "reason": None if reference or predicted else "no annotated positive examples"}


def is_relevant(source: dict, expected: dict) -> bool:
    """Exact source identity; optional page/chunk criteria make relevance narrower."""
    if "designation" in expected and source.get("designation") != expected["designation"]:
        return False
    if "document" in expected and source.get("document") != expected["document"]:
        return False
    return all(source.get(key) == expected[key] for key in ("page", "chunk_id") if key in expected)


def retrieval_metrics(queries: list[dict]) -> dict:
    evaluated = [q for q in queries if q.get("available") and q.get("expected")]
    if not evaluated:
        return {"hit_at_1": None, "hit_at_3": None, "hit_at_5": None, "mrr": None,
                "reason": "reference relevance or saved retrieval data not provided", "queries": queries}
    rows = []
    for query in evaluated:
        relevant = [any(is_relevant(s, e) for e in query["expected"]) for s in query["sources"]]
        ranks = [i + 1 for i, hit in enumerate(relevant) if hit]
        rows.append({"reference_id": query["reference_id"],
                     "ranking_depth": len(query["sources"]),
                     **{f"hit_at_{k}": any(relevant[:k]) for k in (1, 3, 5)},
                     "reciprocal_rank": 1 / min(ranks) if ranks else 0})
    return {**{f"hit_at_{k}": sum(r[f"hit_at_{k}"] for r in rows) / len(rows) for k in (1, 3, 5)},
            "mrr": sum(r["reciprocal_rank"] for r in rows) / len(rows),
            "evaluated_queries": len(rows), "provided_queries": len(queries),
            "mrr_scope": "provided/truncated ranking, not the whole corpus",
            "per_query": rows, "queries": queries, "reason": None}
