from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "support"))

from bci_core import (  # noqa: E402
    archive_comparison_passed,
    assert_safe_paths,
    atomic_write_csv,
    clone_tree_apfs,
    compare_archive,
    is_administrative_artifact,
    normalize_frequency_csv_bytes,
    normalize_performances_csv_bytes,
    parse_csv_text,
    validate_gdf_signature,
    validate_python,
    validate_xml,
    validate_zip_container,
)


class CsvCleaningTests(unittest.TestCase):
    def test_frequency_crcrlf_normalization_preserves_nonempty_cells(self) -> None:
        raw = b"FREQUENCY,SCORE\r\r\n0.0,1.25\r\r\n0.5,-2.0\r\r\n"
        cleaned, rule = normalize_frequency_csv_bytes(raw)
        self.assertEqual(rule, "nonempty-cell-matrix-equal")
        self.assertNotIn(b"\r", cleaned)
        self.assertEqual(
            parse_csv_text(cleaned.decode("utf-8"), ","),
            [["FREQUENCY", "SCORE"], ["0.0", "1.25"], ["0.5", "-2.0"]],
        )

    def test_frequency_unknown_line_endings_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_frequency_csv_bytes(b"a,b\n1,2\n")

    def test_performance_transcode_preserves_multiline_and_repeated_headers(self) -> None:
        text = "DATA A;\r\nname;comment\r\nA1;\"v\xe9lo\r\nquoted\"\r\n\r\nname;comment\r\n"
        raw = text.encode("cp1252")
        original = parse_csv_text(raw.decode("cp1252"), ";")
        cleaned, rule = normalize_performances_csv_bytes(raw)
        self.assertEqual(rule, "parsed-cell-matrix-equal")
        self.assertEqual(parse_csv_text(cleaned.decode("utf-8"), ";"), original)
        self.assertEqual(sum(row == ["name", "comment"] for row in original), 2)


class ArchiveComparisonTests(unittest.TestCase):
    def test_archive_comparison_matches_prefixed_member(self) -> None:
        inventory = [{"relative_path": "Signals/A1.gdf", "size_bytes": "4", "crc32": "1234abcd"}]
        archive = [{"archive_path": "BCI Database/Signals/A1.gdf", "size_bytes": 4, "crc32": "1234abcd"}]
        result = compare_archive("BCI Database", inventory, archive)
        self.assertEqual(result[0]["status"], "match")

    def test_archive_comparison_blocks_crc_mismatch(self) -> None:
        inventory = [{"relative_path": "file", "size_bytes": "4", "crc32": "00000000"}]
        archive = [{"archive_path": "BCI Database/file", "size_bytes": 4, "crc32": "ffffffff"}]
        self.assertEqual(compare_archive("BCI Database", inventory, archive)[0]["status"], "crc_mismatch")

    def test_archive_gate_accepts_only_explicit_administrative_extra(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "archive_comparison.csv"
            fields = ["relative_path", "status"]
            atomic_write_csv(
                report,
                [{"relative_path": ".DS_Store", "status": "administrative_extra"}],
                fields,
            )
            self.assertTrue(archive_comparison_passed(report))
            atomic_write_csv(
                report,
                [{"relative_path": "unknown.bin", "status": "extra_local"}],
                fields,
            )
            self.assertFalse(archive_comparison_passed(report))


class SafetyAndFormatTests(unittest.TestCase):
    def test_raw_symlink_and_apfs_clone(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "data.txt").write_text("unchanged", encoding="utf-8")
            raw_link = root / "raw"
            raw_link.symlink_to(source.name)
            cleaned = root / "cleaned"
            raw, destination = assert_safe_paths(raw_link, cleaned)
            clone_tree_apfs(raw, destination)
            self.assertEqual((cleaned / "data.txt").read_text(encoding="utf-8"), "unchanged")
            self.assertEqual((source / "data.txt").read_text(encoding="utf-8"), "unchanged")

    def test_cleaned_inside_raw_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "raw"
            raw.mkdir()
            with self.assertRaises(RuntimeError):
                assert_safe_paths(raw, raw / "cleaned")

    def test_embedded_clone_skips_project_subtree(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "BCI Database"
            project = raw / "bci_cleaning"
            project.mkdir(parents=True)
            (raw / "publisher.txt").write_text("publisher data", encoding="utf-8")
            (project / "notebook.ipynb").write_text("project file", encoding="utf-8")
            cleaned = project / "cleaned"
            clone_tree_apfs(raw, cleaned)
            self.assertEqual(
                (cleaned / "publisher.txt").read_text(encoding="utf-8"),
                "publisher data",
            )
            self.assertFalse((cleaned / "bci_cleaning").exists())

    def test_administrative_artifacts_require_exact_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ds = root / ".DS_Store"
            ds.write_bytes(b"metadata")
            lock = root / "~$book.xlsx"
            lock.write_bytes(b"not a zip")
            workbook = root / "book.xlsx"
            workbook.write_bytes(b"not a zip")
            self.assertTrue(is_administrative_artifact(".DS_Store", ds))
            self.assertTrue(is_administrative_artifact("~$book.xlsx", lock))
            self.assertFalse(is_administrative_artifact("book.xlsx", workbook))

    def test_corrupt_xml_python_zip_and_gdf_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            xml = root / "bad.xml"
            xml.write_text("<root>", encoding="utf-8")
            py = root / "bad.py"
            py.write_text("if:", encoding="utf-8")
            office = root / "bad.docx"
            office.write_bytes(b"not a zip")
            gdf = root / "bad.gdf"
            gdf.write_bytes(b"not gdf")
            with self.assertRaises(Exception):
                validate_xml(xml)
            with self.assertRaises(SyntaxError):
                validate_python(py)
            with self.assertRaises(Exception):
                validate_zip_container(office)
            with self.assertRaises(ValueError):
                validate_gdf_signature(gdf)

    def test_invalid_pdf_is_rejected_when_pypdf_is_available(self) -> None:
        try:
            from pypdf import PdfReader
        except ImportError:
            self.skipTest("pypdf not installed")
        with self.assertRaises(Exception):
            PdfReader(io.BytesIO(b"not a pdf"), strict=True)


if __name__ == "__main__":
    unittest.main()
