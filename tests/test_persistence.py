from __future__ import annotations

import sqlite3
import shutil
import tempfile
import unittest
from pathlib import Path

from database import EXTENSIBLE_SCHEMA_MIGRATION_ID, initialize_database
from llm_eval.domain import Dataset, DatasetRecord, Model, PromptStrategy, RunSpec, Task, Usage
from llm_eval.evaluators import default_evaluator_registry
from llm_eval.persistence import SqliteRunRepository
from llm_eval.pipeline import EvaluationPipeline
from llm_eval.providers import ProviderResponse
from llm_eval.registry import ProviderRegistry


ROOT = Path(__file__).resolve().parents[1]


class FakeProvider:
    name = "fake"

    def generate(self, request):
        return ProviderResponse(
            status="success",
            output="A",
            provider="fake",
            resolved_model_id=request.model.id,
            usage=Usage(4, 1, 5, 0.002),
        )


def make_spec(run_id: str, system_prompt: str = "") -> RunSpec:
    return RunSpec(
        run_id,
        (Model("model", "Model", "fake"),),
        Task("classification", "Classification", "", "classification", "classification"),
        Dataset(
            "labels",
            "Labels",
            "classification",
            "1",
            (DatasetRecord("one", "Choose A", "A", metadata={"labels": ["A", "B"]}),),
        ),
        (PromptStrategy("basic", "Basic", "zero_shot", "1", system_prompt, "{input}"),),
    )


class MigrationAndPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "evaluation.db"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_migration_is_additive_and_idempotent(self) -> None:
        initialize_database(self.db_path)
        initialize_database(self.db_path)

        with sqlite3.connect(self.db_path) as con:
            tables = {
                row[0]
                for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            self.assertIn("runs", tables)
            self.assertIn("model_outputs", tables)
            self.assertIn("evaluation_runs", tables)
            self.assertIn("evaluation_results", tables)
            self.assertIn("metric_values", tables)
            migrations = con.execute("SELECT migration_id FROM schema_migrations").fetchall()
            self.assertEqual(migrations, [(EXTENSIBLE_SCHEMA_MIGRATION_ID,)])

    def test_migration_preserves_the_completed_legacy_seed_run(self) -> None:
        shutil.copy2(ROOT / "seed" / "llm_eval_seed.db", self.db_path)

        initialize_database(self.db_path)

        with sqlite3.connect(self.db_path) as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 1)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM prompts").fetchone()[0], 120)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM model_outputs").fetchone()[0], 240)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM evaluation_runs").fetchone()[0], 0)

    def test_pipeline_persists_snapshots_results_metrics_and_completion(self) -> None:
        providers = ProviderRegistry()
        providers.register("fake", FakeProvider())
        pipeline = EvaluationPipeline(
            providers=providers,
            evaluators=default_evaluator_registry(),
            sink=SqliteRunRepository(self.db_path),
        )

        pipeline.run(make_spec("run-persisted"))

        with sqlite3.connect(self.db_path) as con:
            run = con.execute(
                "SELECT status,evaluator_version FROM evaluation_runs WHERE run_id='run-persisted'"
            ).fetchone()
            result = con.execute(
                """
                SELECT prompt_strategy_version,evaluator_version,system_prompt_snapshot
                FROM evaluation_results WHERE run_id='run-persisted'
                """
            ).fetchone()
            metric_count = con.execute("SELECT COUNT(*) FROM metric_values").fetchone()[0]
            self.assertEqual(run, ("completed", "1.0"))
            self.assertEqual(result, ("1", "1.0", ""))
            self.assertGreater(metric_count, 10)

    def test_prompt_version_cannot_be_silently_rewritten(self) -> None:
        repository = SqliteRunRepository(self.db_path)
        repository.begin_run(make_spec("run-one"), evaluator_version="1.0")

        with self.assertRaisesRegex(ValueError, "Prompt strategy basic@1 is immutable"):
            repository.begin_run(
                make_spec("run-two", system_prompt="Changed without a new version"),
                evaluator_version="1.0",
            )


if __name__ == "__main__":
    unittest.main()
