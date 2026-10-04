"""Numeric evidence regression tests: no LLM or retrieval calls."""

from decimal import Decimal
import unittest

from llm.models import CitationValidationError, SourceEvidence
from llm.service import (_measurements, _measurement_diagnostics,
                         _validate_measured_values, _resolve_citations)


def source(text, source_id="SOURCE_1"):
    return SourceEvidence(source_id=source_id, document="Test", designation="СП TEST",
                          page=1, chunk_id=source_id, text=text, score=.9, distance=.1)


class NumericValidationTests(unittest.TestCase):
    def valid(self, question, answer, text=""):
        cited = [source(text)]
        _validate_measured_values(answer, question, cited, cited)

    def invalid(self, question, answer, text="", other=""):
        cited = [source(text)]
        with self.assertRaises(CitationValidationError):
            _validate_measured_values(answer, question, cited, cited + [source(other, "SOURCE_2")])

    def test_01_question_80_centimeters(self):
        self.valid("коридор шириной 80 сантиметров", "ширина 0.8 м")

    def test_02_question_520_square_meters(self):
        self.valid("площадь квартирного этажа около 520 квадратных метров", "площадь 520 м²")

    def test_03_cited_decimal_comma(self):
        self.valid("ширина?", "1.05 м", "минимальная ширина лестничного марша составляет 1,05 м")

    def test_04_cited_centimeters(self):
        self.valid("ширина?", "1.05 м", "минимальная ширина составляет 105 см")

    def test_05_cited_millimeters(self):
        self.valid("ширина?", "1.05 м", "минимальная ширина составляет 1050 мм")

    def test_06_invented_value(self):
        self.invalid("ширина около 1 м", "минимум 1.05 м", "требования к ширине")

    def test_07_uncited_value(self):
        self.invalid("ширина около 1 м", "минимум 1.05 м", "требования к ширине", "1.05 м")

    def test_08_cited_area(self):
        self.valid("площадь?", "520 м²", "площадь 520 м²")

    def test_09_length_not_area(self):
        self.invalid("площадь?", "520 м²", "длина 520 м")

    def test_10_question_length_not_area(self):
        self.invalid("ширина 80 см", "0.8 м²")

    def test_11_question_decimal_comma(self):
        self.valid("ширина 1,05 м", "1.05 м")

    def test_12_question_area_abbreviation(self):
        self.valid("площадь 520 кв. м", "520 м²")

    def test_all_russian_units(self):
        for text in ("0.8 м", "80 см", "800 мм", "0,8 метр", "0,8 метра", "0,8 метров",
                     "80 сантиметр", "80 сантиметра", "80 сантиметров", "800 миллиметр",
                     "800 миллиметра", "800 миллиметров"):
            with self.subTest(text=text):
                self.assertEqual(_measurements(text), {("length_m", Decimal('.8'))})
        for unit in ("м²", "м2", "кв. м", "кв.м", "квадратный метр", "квадратных метров", "квадратного метра"):
            with self.subTest(unit=unit):
                self.assertEqual(_measurements("520 " + unit), {("area_m2", Decimal(520))})

    def test_other_dimensions_preserved(self):
        self.assertEqual(_measurements("50 дБА 20% 30 процентов 15 человек"),
                         {("noise_db", Decimal(50)), ("percent", Decimal(20)),
                          ("percent", Decimal(30)), ("people", Decimal(15))})
        self.assertEqual(_measurements("520 м3 520 м³"), set())

    def test_exact_decimal_conversion_not_approximate(self):
        self.valid("1050 мм", "1.05 м")
        self.invalid("1.00 м", "1.05 м")

    def test_width_slope_table_shared_unit(self):
        table = "Таблица 6.1 Максимальный Наименование марша Минимальная ширина, м уклон\n" \
                "- секционных двухэтажных 1,05 1:1,5 - коридорных 1,2 1:1,75\n" \
                "Марши в подвалы 0,9\nПо СП 118.13330 По СП 118.13330\n1:1,25\n" \
                "П р и м е ч а н и я 1 Указания 12 1:4"
        self.assertEqual(_measurements(table), {("length_m", Decimal('1.05')),
                                              ("length_m", Decimal('1.2')), ("length_m", Decimal('.9'))})
        self.valid("ширина?", "1.05 м 1.2 м 0.9 м", table)

    def test_table_requires_dimension_and_structure(self):
        for table in ("Таблица 1 Площадь, м² уклон 520 1:2", "Таблица 1 Ширина 1,05 1:2",
                      "Таблица 1 ширина, м уклон значения 1,05", "Таблица 1 ширина, м уклон По СП 118.13330 1:2"):
            self.assertNotIn(("length_m", Decimal('1.05')), _measurements(table))
            self.assertNotIn(("length_m", Decimal('118.13330')), _measurements(table))

    def test_source_ids_not_list_position(self):
        sources = [source("9 м", "SOURCE_8"), source("105 см", "SOURCE_2")]
        cited = _resolve_citations("1.05 м [SOURCE_2]", sources)
        self.assertEqual(cited[0].text, "105 см")
        _validate_measured_values("1.05 м [SOURCE_2]", "ширина?", cited, sources)
        self.invalid("ширина?", "9 м", "105 см", "9 м")

    def test_evidence_not_joined_across_chunks(self):
        cited = [source("длина 105"), source("см норматив", "SOURCE_2")]
        with self.assertRaises(CitationValidationError):
            _validate_measured_values("1.05 м", "ширина?", cited, cited)

    def test_debug_contains_matched_original_text(self):
        diagnostic = _measurement_diagnostics("0.8 м 1.05 м", "80 сантиметров", [source("105 см")])
        self.assertFalse(any(item["rejected"] for item in diagnostic))
        self.assertIn("80 сантиметров", str(diagnostic))
        self.assertIn("SOURCE_1", str(diagnostic))
        rejected = _measurement_diagnostics("1.2 м", "80 см", [source("105 см")])[0]
        self.assertTrue(rejected["rejected"])
        self.assertEqual(rejected["question_matches"], [])


if __name__ == "__main__":
    unittest.main()
