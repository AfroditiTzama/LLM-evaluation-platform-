from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from database import connect, initialize_database

from .domain import EvaluationResult, RunSpec, to_jsonable, utc_now_iso


def _json(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


class SqliteRunRepository:
    """Persistence adapter for the additive framework schema.

    Catalog definitions are mutable, but versioned datasets, records and prompt
    strategies are immutable. A caller must create a new version instead of
    silently rewriting an existing benchmark or prompt definition.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path
        initialize_database(db_path)

    @staticmethod
    def _assert_snapshot(existing: Any, snapshot: str, label: str) -> None:
        if existing is not None and str(existing[0]) != snapshot:
            raise ValueError(f"{label} is immutable; create a new version")

    def begin_run(self, spec: RunSpec, *, evaluator_version: str) -> None:
        with connect(self.db_path) as con:
            for model in spec.models:
                snapshot = model.snapshot()
                con.execute(
                    """
                    INSERT INTO models(
                        model_key,model_id,name,provider,context_window,input_price_per_million,
                        output_price_per_million,reasoning_support,metadata_json,snapshot_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(model_key) DO UPDATE SET
                        name=excluded.name,context_window=excluded.context_window,
                        input_price_per_million=excluded.input_price_per_million,
                        output_price_per_million=excluded.output_price_per_million,
                        reasoning_support=excluded.reasoning_support,
                        metadata_json=excluded.metadata_json,snapshot_json=excluded.snapshot_json
                    """,
                    (
                        model.key,
                        model.id,
                        model.name,
                        model.provider,
                        model.context_window,
                        model.input_price_per_million,
                        model.output_price_per_million,
                        int(model.reasoning_support),
                        _json(snapshot.get("metadata", {})),
                        _json(snapshot),
                    ),
                )

            task_snapshot = spec.task.snapshot()
            con.execute(
                """
                INSERT INTO tasks(
                    task_id,name,description,task_type,evaluator_type,evaluator_version,
                    supported_metrics_json,metadata_json,snapshot_json
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(task_id) DO UPDATE SET
                    name=excluded.name,description=excluded.description,task_type=excluded.task_type,
                    evaluator_type=excluded.evaluator_type,evaluator_version=excluded.evaluator_version,
                    supported_metrics_json=excluded.supported_metrics_json,
                    metadata_json=excluded.metadata_json,snapshot_json=excluded.snapshot_json
                """,
                (
                    spec.task.id,
                    spec.task.name,
                    spec.task.description,
                    spec.task.task_type,
                    spec.task.evaluator_type,
                    spec.task.evaluator_version,
                    _json(spec.task.supported_metrics),
                    _json(task_snapshot.get("metadata", {})),
                    _json(task_snapshot),
                ),
            )

            dataset_snapshot = spec.dataset.snapshot()
            dataset_snapshot_json = _json(dataset_snapshot)
            existing_dataset = con.execute(
                "SELECT snapshot_json FROM datasets WHERE dataset_id=? AND version=?",
                (spec.dataset.id, spec.dataset.version),
            ).fetchone()
            self._assert_snapshot(
                existing_dataset,
                dataset_snapshot_json,
                f"Dataset {spec.dataset.id}@{spec.dataset.version}",
            )
            con.execute(
                """
                INSERT INTO datasets(
                    dataset_id,version,task_id,name,language,domain,record_count,
                    provenance_json,metadata_json,snapshot_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(dataset_id,version) DO NOTHING
                """,
                (
                    spec.dataset.id,
                    spec.dataset.version,
                    spec.dataset.task_id,
                    spec.dataset.name,
                    spec.dataset.language,
                    spec.dataset.domain,
                    len(spec.dataset.records),
                    _json(dataset_snapshot.get("provenance", {})),
                    _json(dataset_snapshot.get("metadata", {})),
                    dataset_snapshot_json,
                ),
            )
            for record in spec.dataset.records:
                snapshot = record.snapshot()
                snapshot_json = _json(snapshot)
                existing_record = con.execute(
                    """
                    SELECT snapshot_json FROM dataset_records
                    WHERE dataset_id=? AND dataset_version=? AND record_id=?
                    """,
                    (spec.dataset.id, spec.dataset.version, record.id),
                ).fetchone()
                self._assert_snapshot(
                    existing_record,
                    snapshot_json,
                    f"Dataset record {spec.dataset.id}@{spec.dataset.version}/{record.id}",
                )
                con.execute(
                    """
                    INSERT INTO dataset_records(
                        dataset_id,dataset_version,record_id,input_json,reference_json,language,
                        difficulty,domain,variables_json,provenance_json,metadata_json,snapshot_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(dataset_id,dataset_version,record_id) DO NOTHING
                    """,
                    (
                        spec.dataset.id,
                        spec.dataset.version,
                        record.id,
                        _json(record.input),
                        _json(record.reference),
                        record.language,
                        record.difficulty,
                        record.domain,
                        _json(dict(record.variables)),
                        _json(snapshot.get("provenance", {})),
                        _json(dict(record.metadata)),
                        snapshot_json,
                    ),
                )

            for strategy in spec.prompt_strategies:
                snapshot = strategy.snapshot()
                snapshot_json = _json(snapshot)
                existing_prompt = con.execute(
                    "SELECT snapshot_json FROM prompt_strategies WHERE strategy_id=? AND version=?",
                    (strategy.id, strategy.version),
                ).fetchone()
                self._assert_snapshot(existing_prompt, snapshot_json, f"Prompt strategy {strategy.key}")
                con.execute(
                    """
                    INSERT INTO prompt_strategies(
                        strategy_id,version,name,strategy_type,system_prompt,user_prompt_template,
                        language,variables_json,metadata_json,snapshot_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(strategy_id,version) DO NOTHING
                    """,
                    (
                        strategy.id,
                        strategy.version,
                        strategy.name,
                        strategy.strategy_type,
                        strategy.system_prompt,
                        strategy.user_prompt_template,
                        strategy.language,
                        _json(strategy.variables),
                        _json(dict(strategy.metadata)),
                        snapshot_json,
                    ),
                )

            spec_snapshot = {
                "run_id": spec.run_id,
                "created_at": spec.created_at,
                "models": [model.snapshot() for model in spec.models],
                "task": task_snapshot,
                "dataset": dataset_snapshot,
                "prompt_strategies": [item.snapshot() for item in spec.prompt_strategies],
                "generation_settings": spec.generation_settings.snapshot(),
                "metric_configuration": dict(spec.metric_configuration),
                "metadata": dict(spec.metadata),
            }
            con.execute(
                """
                INSERT INTO evaluation_runs(
                    run_id,task_id,dataset_id,dataset_version,status,created_at,started_at,
                    generation_settings_json,evaluator_type,evaluator_version,
                    metric_configuration_json,spec_snapshot_json,metadata_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    spec.run_id,
                    spec.task.id,
                    spec.dataset.id,
                    spec.dataset.version,
                    "running",
                    spec.created_at,
                    utc_now_iso(),
                    _json(spec.generation_settings.snapshot()),
                    spec.task.evaluator_type,
                    evaluator_version,
                    _json(dict(spec.metric_configuration)),
                    _json(spec_snapshot),
                    _json(dict(spec.metadata)),
                ),
            )
            for model in spec.models:
                con.execute(
                    "INSERT INTO evaluation_run_models(run_id,model_key,model_snapshot_json) VALUES (?,?,?)",
                    (spec.run_id, model.key, _json(model.snapshot())),
                )
            for strategy in spec.prompt_strategies:
                con.execute(
                    """
                    INSERT INTO evaluation_run_prompt_strategies(
                        run_id,strategy_id,strategy_version,prompt_strategy_snapshot_json
                    ) VALUES (?,?,?,?)
                    """,
                    (spec.run_id, strategy.id, strategy.version, _json(strategy.snapshot())),
                )
            con.commit()

    def save_result(self, result: EvaluationResult) -> None:
        with connect(self.db_path) as con:
            con.execute(
                """
                INSERT INTO evaluation_results(
                    result_id,run_id,record_id,model_key,task_id,dataset_id,dataset_version,
                    prompt_strategy_id,prompt_strategy_version,status,provider,resolved_model_id,
                    raw_input_json,raw_output,reference_json,system_prompt_snapshot,user_prompt_snapshot,
                    model_snapshot_json,task_snapshot_json,dataset_snapshot_json,
                    prompt_strategy_snapshot_json,generation_settings_snapshot_json,evaluator_version,
                    metric_configuration_snapshot_json,input_tokens,output_tokens,total_tokens,cost_usd,
                    end_to_end_seconds,provider_request_seconds,time_to_first_token_seconds,
                    generation_seconds,inter_token_latency_seconds,error,created_at,metadata_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    result.result_id,
                    result.run_id,
                    result.record_id,
                    result.model_key,
                    result.task_id,
                    result.dataset_id,
                    result.dataset_version,
                    result.prompt_strategy_id,
                    result.prompt_strategy_version,
                    result.status,
                    result.provider,
                    result.resolved_model_id,
                    _json(result.raw_input),
                    result.raw_output,
                    _json(result.reference),
                    result.system_prompt_snapshot,
                    result.user_prompt_snapshot,
                    _json(dict(result.model_snapshot)),
                    _json(dict(result.task_snapshot)),
                    _json(dict(result.dataset_snapshot)),
                    _json(dict(result.prompt_strategy_snapshot)),
                    _json(dict(result.generation_settings_snapshot)),
                    result.evaluator_version,
                    _json(dict(result.metric_configuration_snapshot)),
                    result.usage.input_tokens,
                    result.usage.output_tokens,
                    result.usage.total_tokens,
                    result.usage.cost_usd,
                    result.timings.end_to_end_seconds,
                    result.timings.provider_request_seconds,
                    result.timings.time_to_first_token_seconds,
                    result.timings.generation_seconds,
                    result.timings.inter_token_latency_seconds,
                    result.error,
                    result.created_at,
                    _json(dict(result.metadata)),
                ),
            )
            for metric in result.metrics:
                value = metric.value
                if value is None:
                    value_type, value_real, value_text = "null", None, None
                elif isinstance(value, bool):
                    value_type, value_real, value_text = "boolean", float(value), None
                elif isinstance(value, (int, float)):
                    value_type, value_real, value_text = "number", float(value), None
                else:
                    value_type, value_real, value_text = "text", None, str(value)
                con.execute(
                    """
                    INSERT INTO metric_values(
                        result_id,metric_name,value_real,value_text,value_type,unit,metadata_json
                    ) VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        result.result_id,
                        metric.name,
                        value_real,
                        value_text,
                        value_type,
                        metric.unit,
                        _json(dict(metric.metadata)),
                    ),
                )
            con.commit()

    def finish_run(self, run_id: str, *, status: str) -> None:
        with connect(self.db_path) as con:
            con.execute(
                "UPDATE evaluation_runs SET status=?, ended_at=? WHERE run_id=?",
                (status, utc_now_iso(), run_id),
            )
            con.commit()
