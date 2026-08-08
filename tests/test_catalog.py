from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from llm_eval.catalog import load_catalogs, load_canonical_dataset, load_legacy_benchmark_datasets


ROOT = Path(__file__).resolve().parents[1]


class CatalogTests(unittest.TestCase):
    def test_versioned_catalogs_load_without_code_changes(self) -> None:
        catalogs = load_catalogs(ROOT / "catalog")

        self.assertEqual(len(catalogs.models), 2)
        self.assertIn("summarization", catalogs.tasks.keys())
        self.assertIn("tabular_cleaning", catalogs.tasks.keys())
        self.assertIn("basic-zero-shot-el@1", catalogs.prompt_strategies.keys())
        self.assertIn("few-shot-el@1", catalogs.prompt_strategies.keys())
        self.assertIn("role-expert-el@1", catalogs.prompt_strategies.keys())
        self.assertIn("critique-revise-el@1", catalogs.prompt_strategies.keys())

    def test_legacy_dataset_provenance_is_not_misrepresented(self) -> None:
        datasets = load_legacy_benchmark_datasets(ROOT / "benchmark_prompts.json")

        self.assertEqual(len(datasets), 10)
        dataset = datasets["summarization"]
        self.assertEqual(dataset.version, "1.1")
        self.assertEqual(dataset.provenance.record_origin, "newly_authored")
        self.assertEqual(dataset.metadata["data_status"], "newly_authored_benchmark_items")
        self.assertTrue(all(record.provenance.record_origin == "newly_authored" for record in dataset.records))

    def test_canonical_dataset_loads_record_level_provenance_and_variables(self) -> None:
        payload = {
            "id": "classification-demo",
            "name": "Classification demo",
            "task_id": "classification",
            "version": "1",
            "language": "el",
            "provenance": {
                "source_title": "Internal labeled sample",
                "record_origin": "sourced",
            },
            "records": [
                {
                    "id": "one",
                    "input": "Example",
                    "reference": "positive",
                    "difficulty": "easy",
                    "variables": {"labels": "positive, negative"},
                    "provenance": {
                        "source_title": "Record source",
                        "url": "https://example.test/record/one",
                        "record_origin": "sourced",
                    },
                    "metadata": {"labels": ["positive", "negative"]},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            dataset = load_canonical_dataset(path)

        self.assertEqual(dataset.task_id, "classification")
        self.assertEqual(dataset.records[0].variables["labels"], "positive, negative")
        self.assertEqual(dataset.records[0].provenance.url, "https://example.test/record/one")


if __name__ == "__main__":
    unittest.main()
