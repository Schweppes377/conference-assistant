from __future__ import annotations

import unittest

import pymupdf

from rag.config import RAGConfig
from rag.pdf_loader import load_source_catalog


class CorpusTests(unittest.TestCase):
    def test_catalog_matches_readable_pdf_files(self) -> None:
        documents_dir = RAGConfig().documents_dir
        catalog = load_source_catalog(documents_dir)

        self.assertGreaterEqual(len(catalog), 10)
        for record in catalog.values():
            path = documents_dir / record.file
            self.assertTrue(path.is_file(), record.file)
            with path.open("rb") as stream:
                self.assertEqual(stream.read(5), b"%PDF-")
            with pymupdf.open(path) as document:
                self.assertGreater(len(document), 0, record.file)
                extracted = sum(len(page.get_text("text").strip()) for page in document)
            self.assertEqual(extracted > 0, record.has_extractable_text, record.file)


if __name__ == "__main__":
    unittest.main()
