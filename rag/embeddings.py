"""Replaceable sentence-transformers embedding backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    """Minimal interface used by indexing and search."""

    @property
    def device(self) -> str: ...

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def encode_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Lazy-loaded sentence-transformers model with optional E5 prefixes."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: SentenceTransformer | None = None

    @property
    def device(self) -> str:
        return self._device

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages as normalized vectors."""

        prepared = [self._with_prefix(text, "passage") for text in texts]
        return self._encode(prepared, show_progress=len(prepared) >= self.batch_size * 2)

    def encode_query(self, text: str) -> list[float]:
        """Embed one search query as a normalized vector."""

        prepared = self._with_prefix(text.strip(), "query")
        return self._encode([prepared], show_progress=False)[0]

    def _encode(self, texts: Sequence[str], *, show_progress: bool) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return vectors.tolist()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self._device,
                    local_files_only=True,
                )
            except OSError:
                self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    def _with_prefix(self, text: str, kind: str) -> str:
        if "e5" in self.model_name.casefold():
            return f"{kind}: {text}"
        return text
