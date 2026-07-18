from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Iterable

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "llm_eval.db"
SEED_DB_PATH = BASE_DIR / "seed" / "llm_eval_seed.db"


def turso_enabled() -> bool:
    return bool(os.getenv("TURSO_DATABASE_URL", "").strip() and os.getenv("TURSO_AUTH_TOKEN", "").strip())


def get_database_path() -> Path:
    raw = os.getenv("DATABASE_PATH", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_DB_PATH


def database_label(path: Path | None = None) -> str:
    if turso_enabled():
        return os.getenv("TURSO_DATABASE_URL", "Turso Cloud")
    return str(path or get_database_path())


def ensure_database_file(path: Path | None = None) -> Path | None:
    if turso_enabled():
        initialize_database(None)
        return None
    target = path or get_database_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() and SEED_DB_PATH.exists() and target.resolve() != SEED_DB_PATH.resolve():
        shutil.copy2(SEED_DB_PATH, target)
    initialize_database(target)
    return target


def _open_connection(path: Path | None = None, *, readonly: bool = False):
    if turso_enabled():
        import libsql

        connection = libsql.connect(
            database=os.environ["TURSO_DATABASE_URL"],
            auth_token=os.environ["TURSO_AUTH_TOKEN"],
            timeout=30.0,
        )
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    target = path or get_database_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() and SEED_DB_PATH.exists() and target.resolve() != SEED_DB_PATH.resolve():
        shutil.copy2(SEED_DB_PATH, target)
    if readonly:
        connection = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=30)
    else:
        connection = sqlite3.connect(target, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    if not readonly:
        connection.execute("PRAGMA journal_mode = WAL")
    return connection


def connect(path: Path | None = None, *, readonly: bool = False):
    if not turso_enabled():
        ensure_database_file(path)
    return _open_connection(path, readonly=readonly)


def initialize_database(path: Path | None = None) -> None:
    if not turso_enabled():
        target = path or get_database_path()
        target.parent.mkdir(parents=True, exist_ok=True)
    with _open_connection(path, readonly=False) as con:
        con.execute("PRAGMA foreign_keys = ON")
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT,
                dataset_name TEXT,
                dataset_version TEXT,
                judge_model TEXT,
                execution_mode TEXT,
                notes TEXT,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS prompts (
                dataset_version TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                category TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                source_benchmark TEXT,
                evaluation_type TEXT,
                expected_format TEXT,
                prompt_text TEXT NOT NULL,
                prompt_json TEXT NOT NULL,
                PRIMARY KEY (dataset_version, prompt_id)
            );

            CREATE TABLE IF NOT EXISTS model_outputs (
                run_id TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_id TEXT,
                category TEXT,
                difficulty TEXT,
                provider TEXT,
                status TEXT,
                response TEXT,
                latency_seconds REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                cost_usd REAL,
                retry_count INTEGER,
                strict_exact_match REAL,
                normalized_exact_match REAL,
                contains_expected_answer REAL,
                numeric_match REAL,
                deterministic_pass REAL,
                deterministic_method TEXT,
                structured_required INTEGER,
                structured_format TEXT,
                syntax_valid REAL,
                schema_valid REAL,
                no_markdown_fence REAL,
                format_compliance REAL,
                timestamp TEXT,
                row_json TEXT NOT NULL,
                PRIMARY KEY (run_id, prompt_id, model_name),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_outputs_run_model ON model_outputs(run_id, model_name);
            CREATE INDEX IF NOT EXISTS idx_outputs_category ON model_outputs(run_id, category, difficulty);

            CREATE TABLE IF NOT EXISTS pairwise_judgments (
                run_id TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                category TEXT,
                difficulty TEXT,
                judge_model TEXT,
                answer_a_model TEXT,
                answer_b_model TEXT,
                winner_label_raw TEXT,
                winner_label_effective TEXT,
                winner_model_effective TEXT,
                normalization_reason TEXT,
                confidence REAL,
                pairwise_reason TEXT,
                judge_latency_seconds REAL,
                judge_cost_usd REAL,
                judge_provider TEXT,
                judgment_json TEXT NOT NULL,
                PRIMARY KEY (run_id, prompt_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS judge_scores (
                run_id TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_id TEXT,
                category TEXT,
                difficulty TEXT,
                blind_answer_label TEXT,
                winner_label_raw TEXT,
                winner_label_effective TEXT,
                winner_model_name_effective TEXT,
                is_winner REAL,
                is_tie REAL,
                is_cannot_assess REAL,
                judge_confidence REAL,
                correctness REAL,
                instruction_following REAL,
                factuality_grounding REAL,
                greek_quality REAL,
                overall_quality REAL,
                hallucination_detected INTEGER,
                task_success INTEGER,
                critical_error TEXT,
                rationale TEXT,
                pairwise_reason TEXT,
                judge_model TEXT,
                judge_latency_seconds REAL,
                judge_cost_usd REAL,
                row_json TEXT NOT NULL,
                PRIMARY KEY (run_id, prompt_id, model_name),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_judge_run_model ON judge_scores(run_id, model_name);
            CREATE INDEX IF NOT EXISTS idx_judge_category ON judge_scores(run_id, category, difficulty);

            CREATE TABLE IF NOT EXISTS model_summary (
                run_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                PRIMARY KEY (run_id, model_name),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS category_summary (
                run_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                category TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                PRIMARY KEY (run_id, model_name, category),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS difficulty_summary (
                run_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                PRIMARY KEY (run_id, model_name, difficulty),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS provider_summary (
                run_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                provider TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                PRIMARY KEY (run_id, model_name, provider),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS human_review_sample (
                run_id TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                category TEXT,
                difficulty TEXT,
                prompt_text TEXT,
                answer_a TEXT,
                answer_b TEXT,
                answer_a_model TEXT,
                answer_b_model TEXT,
                PRIMARY KEY (run_id, prompt_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS human_reviews (
                run_id TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                reviewer TEXT NOT NULL DEFAULT 'default',
                winner TEXT,
                correctness_a REAL,
                correctness_b REAL,
                instruction_following_a REAL,
                instruction_following_b REAL,
                factuality_a REAL,
                factuality_b REAL,
                greek_quality_a REAL,
                greek_quality_b REAL,
                comments TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, prompt_id, reviewer),
                FOREIGN KEY (run_id, prompt_id) REFERENCES human_review_sample(run_id, prompt_id) ON DELETE CASCADE
            );
            """
        )
        con.commit()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _bool_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(bool(value))


def import_run_bundle(
    *,
    db_path: Path | None,
    metadata: dict[str, Any],
    prompts: list[dict[str, Any]],
    results: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
    model_summary: list[dict[str, Any]],
    category_summary: list[dict[str, Any]],
    difficulty_summary: list[dict[str, Any]],
    provider_summary: list[dict[str, Any]],
    human_rows: list[dict[str, Any]],
    human_key: list[dict[str, Any]],
    notes: str = "",
) -> None:
    initialize_database(db_path)
    run_id = str(metadata.get("run_id") or (results[0].get("run_id") if results else ""))
    if not run_id:
        raise ValueError("Missing run_id")
    dataset_meta = metadata.get("dataset_metadata") or {}
    dataset_version = str(dataset_meta.get("version") or "1.1")

    key_by_prompt = {str(item["prompt_id"]): item for item in human_key}
    flat_by_prompt_model = {(str(row.get("prompt_id")), str(row.get("model_name"))): row for row in judge_rows}

    with connect(db_path) as con:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute(
            """
            INSERT INTO runs(run_id, created_at, dataset_name, dataset_version, judge_model, execution_mode, notes, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                created_at=excluded.created_at,
                dataset_name=excluded.dataset_name,
                dataset_version=excluded.dataset_version,
                judge_model=excluded.judge_model,
                execution_mode=excluded.execution_mode,
                notes=excluded.notes,
                metadata_json=excluded.metadata_json
            """,
            (
                run_id,
                metadata.get("created_at"),
                dataset_meta.get("dataset_name", "Greek Research-Inspired LLM Benchmark"),
                dataset_version,
                metadata.get("judge_model"),
                metadata.get("execution_mode"),
                notes,
                _json(metadata),
            ),
        )

        for prompt in prompts:
            con.execute(
                """
                INSERT INTO prompts(dataset_version, prompt_id, category, difficulty, source_benchmark,
                                    evaluation_type, expected_format, prompt_text, prompt_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_version, prompt_id) DO UPDATE SET
                    category=excluded.category,
                    difficulty=excluded.difficulty,
                    source_benchmark=excluded.source_benchmark,
                    evaluation_type=excluded.evaluation_type,
                    expected_format=excluded.expected_format,
                    prompt_text=excluded.prompt_text,
                    prompt_json=excluded.prompt_json
                """,
                (
                    dataset_version,
                    prompt.get("prompt_id"),
                    prompt.get("category"),
                    prompt.get("difficulty"),
                    prompt.get("source_benchmark"),
                    prompt.get("evaluation_type"),
                    prompt.get("expected_format"),
                    prompt.get("prompt"),
                    _json(prompt),
                ),
            )

        for row in results:
            con.execute(
                """
                INSERT INTO model_outputs(
                    run_id,prompt_id,model_name,model_id,category,difficulty,provider,status,response,
                    latency_seconds,input_tokens,output_tokens,total_tokens,cost_usd,retry_count,
                    strict_exact_match,normalized_exact_match,contains_expected_answer,numeric_match,
                    deterministic_pass,deterministic_method,structured_required,structured_format,
                    syntax_valid,schema_valid,no_markdown_fence,format_compliance,timestamp,row_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id,prompt_id,model_name) DO UPDATE SET
                    model_id=excluded.model_id,category=excluded.category,difficulty=excluded.difficulty,
                    provider=excluded.provider,status=excluded.status,response=excluded.response,
                    latency_seconds=excluded.latency_seconds,input_tokens=excluded.input_tokens,
                    output_tokens=excluded.output_tokens,total_tokens=excluded.total_tokens,
                    cost_usd=excluded.cost_usd,retry_count=excluded.retry_count,
                    strict_exact_match=excluded.strict_exact_match,
                    normalized_exact_match=excluded.normalized_exact_match,
                    contains_expected_answer=excluded.contains_expected_answer,
                    numeric_match=excluded.numeric_match,deterministic_pass=excluded.deterministic_pass,
                    deterministic_method=excluded.deterministic_method,
                    structured_required=excluded.structured_required,structured_format=excluded.structured_format,
                    syntax_valid=excluded.syntax_valid,schema_valid=excluded.schema_valid,
                    no_markdown_fence=excluded.no_markdown_fence,format_compliance=excluded.format_compliance,
                    timestamp=excluded.timestamp,row_json=excluded.row_json
                """,
                (
                    run_id,row.get("prompt_id"),row.get("model_name"),row.get("model_id"),
                    row.get("category"),row.get("difficulty"),row.get("provider"),row.get("status"),
                    row.get("response"),row.get("latency_seconds"),row.get("input_tokens"),
                    row.get("output_tokens"),row.get("total_tokens"),row.get("cost_usd"),row.get("retry_count"),
                    row.get("strict_exact_match"),row.get("normalized_exact_match"),row.get("contains_expected_answer"),
                    row.get("numeric_match"),row.get("deterministic_pass"),row.get("deterministic_method"),
                    _bool_int(row.get("structured_required")),row.get("structured_format"),row.get("syntax_valid"),
                    row.get("schema_valid"),row.get("no_markdown_fence"),row.get("format_compliance"),
                    row.get("timestamp"),_json(row),
                ),
            )

        for judgment in judgments:
            if judgment.get("status") != "success":
                continue
            prompt_id = str(judgment.get("prompt_id"))
            first = next((row for row in judge_rows if str(row.get("prompt_id")) == prompt_id), {})
            effective = first.get("winner_label_effective")
            effective_model = first.get("winner_model_name_effective")
            con.execute(
                """
                INSERT INTO pairwise_judgments(
                    run_id,prompt_id,category,difficulty,judge_model,answer_a_model,answer_b_model,
                    winner_label_raw,winner_label_effective,winner_model_effective,normalization_reason,
                    confidence,pairwise_reason,judge_latency_seconds,judge_cost_usd,judge_provider,judgment_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id,prompt_id) DO UPDATE SET
                    category=excluded.category,difficulty=excluded.difficulty,judge_model=excluded.judge_model,
                    answer_a_model=excluded.answer_a_model,answer_b_model=excluded.answer_b_model,
                    winner_label_raw=excluded.winner_label_raw,winner_label_effective=excluded.winner_label_effective,
                    winner_model_effective=excluded.winner_model_effective,normalization_reason=excluded.normalization_reason,
                    confidence=excluded.confidence,pairwise_reason=excluded.pairwise_reason,
                    judge_latency_seconds=excluded.judge_latency_seconds,judge_cost_usd=excluded.judge_cost_usd,
                    judge_provider=excluded.judge_provider,judgment_json=excluded.judgment_json
                """,
                (
                    run_id,prompt_id,judgment.get("category"),judgment.get("difficulty"),judgment.get("judge_model"),
                    judgment.get("answer_a_model_name"),judgment.get("answer_b_model_name"),
                    judgment.get("winner_label") or judgment.get("winner"),effective,effective_model,
                    first.get("winner_normalization_reason"),judgment.get("confidence"),judgment.get("pairwise_reason"),
                    judgment.get("judge_latency_seconds"),judgment.get("judge_cost_usd"),judgment.get("judge_provider"),
                    _json(judgment),
                ),
            )

        for row in judge_rows:
            if row.get("status") != "success":
                continue
            con.execute(
                """
                INSERT INTO judge_scores(
                    run_id,prompt_id,model_name,model_id,category,difficulty,blind_answer_label,
                    winner_label_raw,winner_label_effective,winner_model_name_effective,is_winner,is_tie,
                    is_cannot_assess,judge_confidence,correctness,instruction_following,factuality_grounding,
                    greek_quality,overall_quality,hallucination_detected,task_success,critical_error,rationale,
                    pairwise_reason,judge_model,judge_latency_seconds,judge_cost_usd,row_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id,prompt_id,model_name) DO UPDATE SET
                    model_id=excluded.model_id,category=excluded.category,difficulty=excluded.difficulty,
                    blind_answer_label=excluded.blind_answer_label,winner_label_raw=excluded.winner_label_raw,
                    winner_label_effective=excluded.winner_label_effective,
                    winner_model_name_effective=excluded.winner_model_name_effective,
                    is_winner=excluded.is_winner,is_tie=excluded.is_tie,is_cannot_assess=excluded.is_cannot_assess,
                    judge_confidence=excluded.judge_confidence,correctness=excluded.correctness,
                    instruction_following=excluded.instruction_following,factuality_grounding=excluded.factuality_grounding,
                    greek_quality=excluded.greek_quality,overall_quality=excluded.overall_quality,
                    hallucination_detected=excluded.hallucination_detected,task_success=excluded.task_success,
                    critical_error=excluded.critical_error,rationale=excluded.rationale,
                    pairwise_reason=excluded.pairwise_reason,judge_model=excluded.judge_model,
                    judge_latency_seconds=excluded.judge_latency_seconds,judge_cost_usd=excluded.judge_cost_usd,
                    row_json=excluded.row_json
                """,
                (
                    run_id,row.get("prompt_id"),row.get("model_name"),row.get("model_id"),row.get("category"),
                    row.get("difficulty"),row.get("blind_answer_label"),row.get("winner_label_raw"),
                    row.get("winner_label_effective"),row.get("winner_model_name_effective"),row.get("is_winner"),
                    row.get("is_tie"),row.get("is_cannot_assess"),row.get("judge_confidence"),row.get("correctness"),
                    row.get("instruction_following"),row.get("factuality_grounding"),row.get("greek_quality"),
                    row.get("overall_quality"),_bool_int(row.get("hallucination_detected")),
                    _bool_int(row.get("task_success")),row.get("critical_error"),row.get("rationale"),
                    row.get("pairwise_reason"),row.get("judge_model"),row.get("judge_latency_seconds"),
                    row.get("judge_cost_usd"),_json(row),
                ),
            )

        for table, key_columns, rows in (
            ("model_summary", ("model_name",), model_summary),
            ("category_summary", ("model_name", "category"), category_summary),
            ("difficulty_summary", ("model_name", "difficulty"), difficulty_summary),
            ("provider_summary", ("model_name", "provider"), provider_summary),
        ):
            for row in rows:
                columns = ["run_id", *key_columns, "summary_json"]
                placeholders = ",".join("?" for _ in columns)
                conflict = ",".join(["run_id", *key_columns])
                values = [run_id, *[row.get(key) for key in key_columns], _json(row)]
                con.execute(
                    f"INSERT INTO {table}({','.join(columns)}) VALUES ({placeholders}) "
                    f"ON CONFLICT({conflict}) DO UPDATE SET summary_json=excluded.summary_json",
                    values,
                )

        for human_row in human_rows:
            prompt_id = str(human_row.get("prompt_id"))
            key = key_by_prompt.get(prompt_id, {})
            con.execute(
                """
                INSERT INTO human_review_sample(
                    run_id,prompt_id,category,difficulty,prompt_text,answer_a,answer_b,answer_a_model,answer_b_model
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id,prompt_id) DO UPDATE SET
                    category=excluded.category,difficulty=excluded.difficulty,prompt_text=excluded.prompt_text,
                    answer_a=excluded.answer_a,answer_b=excluded.answer_b,
                    answer_a_model=excluded.answer_a_model,answer_b_model=excluded.answer_b_model
                """,
                (
                    run_id,prompt_id,human_row.get("category"),human_row.get("difficulty"),human_row.get("prompt"),
                    human_row.get("answer_a"),human_row.get("answer_b"),key.get("answer_a_model"),key.get("answer_b_model"),
                ),
            )

        con.commit()


def table_rows(table: str, run_id: str | None = None, db_path: Path | None = None) -> list[dict[str, Any]]:
    allowed = {
        "runs", "prompts", "model_outputs", "pairwise_judgments", "judge_scores",
        "model_summary", "category_summary", "difficulty_summary", "provider_summary",
        "human_review_sample", "human_reviews",
    }
    if table not in allowed:
        raise ValueError(f"Unsupported table: {table}")
    query = f"SELECT * FROM {table}"
    params: tuple[Any, ...] = ()
    if run_id and table != "prompts":
        query += " WHERE run_id = ?"
        params = (run_id,)
    with connect(db_path, readonly=True) as con:
        cursor = con.execute(query, params)
        columns = [item[0] for item in cursor.description or []]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
