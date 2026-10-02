from __future__ import annotations

import unittest

from rag.chunking import split_text


class ChunkingTests(unittest.TestCase):
    def test_chunks_respect_limit_and_word_boundaries(self) -> None:
        text = (
            "Первый абзац содержит требования к эвакуационным выходам и путям. "
            "Он должен остаться читаемым.\n\n"
            "Второй абзац описывает минимальную ширину прохода и направление двери. "
            "Третий фрагмент добавляет условия для многоквартирного дома."
        )

        chunks = split_text(text, chunk_size=110, chunk_overlap=25)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 110 for chunk in chunks))
        self.assertTrue(all(not chunk.startswith(" ") and not chunk.endswith(" ") for chunk in chunks))
        self.assertTrue(all("  " not in chunk for chunk in chunks))

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            split_text("text", chunk_size=100, chunk_overlap=100)


if __name__ == "__main__":
    unittest.main()
