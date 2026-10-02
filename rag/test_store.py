from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag.models import TextChunk
from rag.store import VectorStore


class VectorStoreTests(unittest.TestCase):
    def test_reindexing_replaces_document_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            store = VectorStore(Path(temp_dir), "test_norms", "test-model")
            chunks = [
                TextChunk("doc.pdf", "Document", "SP TEST", 1, 0, "эвакуационный выход"),
                TextChunk("doc.pdf", "Document", "SP TEST", 2, 1, "тепловая защита"),
            ]
            embeddings = [[1.0, 0.0], [0.0, 1.0]]

            store.replace_document("doc.pdf", chunks, embeddings)
            self.assertEqual(store.count, 2)
            store.replace_document("doc.pdf", chunks, embeddings)
            self.assertEqual(store.count, 2)

            store.replace_document("doc.pdf", chunks[:1], embeddings[:1])
            self.assertEqual(store.count, 1)
            result = store.search([1.0, 0.0], top_k=1)[0]
            self.assertEqual(result.metadata["page"], 1)


if __name__ == "__main__":
    unittest.main()
