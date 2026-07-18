from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import random
import re
import statistics
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from database import get_database_path, import_run_bundle
from evaluation import deterministic_evaluation as evaluate_deterministically
from reporting import (
    create_group_summary as create_group_summary_v2,
    create_model_summary as create_model_summary_v2,
    create_provider_summary,
    create_stratified_human_sample,
    flatten_judgments as flatten_judgments_v2,
    recalculate_results,
)


# ============================================================
# PATHS AND ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PROMPTS_FILE = BASE_DIR / "benchmark_prompts.json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-4.1-mini").strip()

API_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "180"))
MAX_API_RETRIES = int(os.getenv("MAX_API_RETRIES", "2"))
RETRY_BACKOFF_SECONDS = float(os.getenv("RETRY_BACKOFF_SECONDS", "1.0"))
PAUSE_BETWEEN_REQUESTS = float(os.getenv("PAUSE_BETWEEN_REQUESTS", "0.5"))
BENCHMARK_MAX_WORKERS = int(os.getenv("BENCHMARK_MAX_WORKERS", "4"))
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}

_thread_local = threading.local()


# ============================================================
# MODELS AND SETTINGS
# ============================================================

MODELS = [
    {
        "name": "Qwen 3.6 27B",
        "id": "qwen/qwen3.6-27b",
    },
    {
        "name": "Gemma 4 31B Instruct",
        "id": "google/gemma-4-31b-it",
    },
]

DEFAULT_SYSTEM_PROMPT = (
    "Απάντησε αποκλειστικά στα ελληνικά, εκτός αν το prompt ζητά ρητά "
    "κώδικα, JSON, SQL, YAML ή άλλη συγκεκριμένη μορφή. "
    "Ακολούθησε πιστά όλες τις οδηγίες του χρήστη. "
    "Μην προσθέτεις πληροφορίες που δεν προκύπτουν από τα παρεχόμενα "
    "δεδομένα. Όταν δεν υπάρχουν αρκετές πληροφορίες, δήλωσέ το καθαρά. "
    "Χρησιμοποίησε φυσική, σαφή και επαγγελματική ελληνική γλώσσα."
)

BENCHMARK_SETTINGS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 3000,
    "reasoning_enabled": False,
}

JUDGE_SETTINGS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1600,
    "reasoning_enabled": False,
}

JUDGE_SCORE_FIELDS = [
    "correctness",
    "instruction_following",
    "factuality_grounding",
    "greek_quality",
    "overall_quality",
]


# ============================================================
# GENERAL HELPERS
# ============================================================


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_http_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def number_or_zero(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def mean_or_none(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    result = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return round(result, 3)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json|yaml|sql|python|html)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def normalized_answer(value: Any) -> str:
    text = strip_code_fence(str(value)).casefold().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.,;:!?\"'`")


def parse_json_response(text: str) -> Any | None:
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start_candidates = [index for index in (cleaned.find("{"), cleaned.find("[")) if index >= 0]
        if not start_candidates:
            return None
        start = min(start_candidates)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end <= start:
            return None
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None


# ============================================================
# OPENROUTER CLIENT
# ============================================================


def call_openrouter(
    model_id: str,
    messages: list[dict[str, str]],
    settings: dict[str, Any],
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not OPENROUTER_API_KEY:
        return {
            "status": "error",
            "error": "Δεν βρέθηκε OPENROUTER_API_KEY στο .env.",
            "latency_seconds": 0,
            "attempt_count": 0,
            "retry_count": 0,
        }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Greek LLM Evaluation"),
    }

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "max_tokens": settings["max_tokens"],
    }
    if response_format is not None:
        payload["response_format"] = response_format

    if settings.get("reasoning_enabled"):
        payload["reasoning"] = {"enabled": True, "effort": "medium"}
    else:
        payload["reasoning"] = {"effort": "none", "exclude": True}

    started = time.perf_counter()
    attempt_log: list[dict[str, Any]] = []

    for attempt_index in range(MAX_API_RETRIES + 1):
        attempt_started = time.perf_counter()
        try:
            response = get_http_session().post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            request_latency = time.perf_counter() - attempt_started
        except requests.RequestException as exc:
            request_latency = time.perf_counter() - attempt_started
            attempt_log.append({
                "attempt": attempt_index + 1,
                "status": "network_error",
                "latency_seconds": round(request_latency, 3),
                "error": str(exc),
            })
            if attempt_index < MAX_API_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt_index))
                continue
            return {
                "status": "error",
                "error": f"Σφάλμα δικτύου: {exc}",
                "latency_seconds": round(time.perf_counter() - started, 3),
                "attempt_count": len(attempt_log),
                "retry_count": max(0, len(attempt_log) - 1),
                "attempt_log": attempt_log,
            }

        try:
            data = response.json()
        except ValueError:
            data = {}

        if not response.ok:
            attempt_log.append({
                "attempt": attempt_index + 1,
                "status": f"http_{response.status_code}",
                "latency_seconds": round(request_latency, 3),
            })
            if response.status_code in RETRYABLE_STATUS_CODES and attempt_index < MAX_API_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt_index))
                continue
            return {
                "status": "error",
                "error": f"OpenRouter HTTP {response.status_code}: {json.dumps(data or response.text[:1000], ensure_ascii=False)}",
                "latency_seconds": round(time.perf_counter() - started, 3),
                "attempt_count": len(attempt_log),
                "retry_count": max(0, len(attempt_log) - 1),
                "attempt_log": attempt_log,
            }

        choices = data.get("choices") or []
        if not choices:
            return {
                "status": "error",
                "error": "Το API δεν επέστρεψε choices.",
                "latency_seconds": round(time.perf_counter() - started, 3),
                "attempt_count": attempt_index + 1,
                "retry_count": attempt_index,
            }

        choice = choices[0]
        message = choice.get("message") or {}
        content = normalize_content(message.get("content"))
        usage = data.get("usage") or {}
        attempt_log.append({
            "attempt": attempt_index + 1,
            "status": "success",
            "latency_seconds": round(request_latency, 3),
        })

        return {
            "status": "success" if content else "empty",
            "response_text": content,
            "finish_reason": choice.get("finish_reason", ""),
            "latency_seconds": round(time.perf_counter() - started, 3),
            "request_latency_seconds": round(request_latency, 3),
            "attempt_count": len(attempt_log),
            "retry_count": max(0, len(attempt_log) - 1),
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "cost_usd": optional_float(usage.get("cost")),
            "generation_id": data.get("id", ""),
            "provider": data.get("provider", ""),
            "attempt_log": attempt_log,
            "error": "" if content else "Κενό τελικό content.",
        }

    return {"status": "error", "error": "Άγνωστο σφάλμα API."}


# ============================================================
# DATASET VALIDATION
# ============================================================


def load_benchmark_dataset() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not PROMPTS_FILE.exists():
        raise FileNotFoundError(f"Δεν βρέθηκε: {PROMPTS_FILE}")

    with PROMPTS_FILE.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        metadata: dict[str, Any] = {}
        prompts = payload
    elif isinstance(payload, dict) and isinstance(payload.get("prompts"), list):
        metadata = {key: value for key, value in payload.items() if key != "prompts"}
        prompts = payload["prompts"]
    else:
        raise ValueError("Το benchmark_prompts.json πρέπει να είναι λίστα ή object με πεδίο prompts.")

    required = {"prompt_id", "category", "difficulty", "prompt"}
    seen_ids: set[str] = set()
    validated: list[dict[str, Any]] = []

    for index, item in enumerate(prompts, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Το prompt #{index} δεν είναι JSON object.")
        missing = required - set(item)
        if missing:
            raise ValueError(f"Το prompt #{index} δεν έχει: {sorted(missing)}")
        prompt_id = str(item["prompt_id"]).strip()
        if prompt_id in seen_ids:
            raise ValueError(f"Διπλό prompt_id: {prompt_id}")
        seen_ids.add(prompt_id)
        difficulty = str(item["difficulty"]).lower().strip()
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(f"Μη έγκυρη difficulty στο {prompt_id}: {difficulty}")
        normalized = dict(item)
        normalized["prompt_id"] = prompt_id
        normalized["difficulty"] = difficulty
        validated.append(normalized)

    return metadata, validated


# ============================================================
# DETERMINISTIC CHECKS
# ============================================================


def deterministic_evaluation(prompt: dict[str, Any], response: str) -> dict[str, Any]:
    """Run corrected deterministic checks from evaluation.py."""
    return evaluate_deterministically(prompt, response)


# ============================================================
# BENCHMARK EXECUTION
# ============================================================


def build_benchmark_row(
    run_id: str,
    prompt: dict[str, Any],
    model: dict[str, str],
) -> dict[str, Any]:
    result = call_openrouter(
        model_id=model["id"],
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": str(prompt["prompt"])},
        ],
        settings=BENCHMARK_SETTINGS,
    )

    response_text = result.get("response_text", "")
    deterministic = (
        deterministic_evaluation(prompt, response_text)
        if result.get("status") == "success"
        else {}
    )

    return {
        "run_id": run_id,
        "prompt_id": prompt["prompt_id"],
        "category": prompt["category"],
        "difficulty": prompt["difficulty"],
        "source_benchmark": prompt.get("source_benchmark", ""),
        "evaluation_type": prompt.get("evaluation_type", ""),
        "expected_format": prompt.get("expected_format", ""),
        "model_name": model["name"],
        "model_id": model["id"],
        "prompt": prompt["prompt"],
        "response": response_text,
        "status": result.get("status", "error"),
        "error": result.get("error", ""),
        "latency_seconds": result.get("latency_seconds", 0),
        "request_latency_seconds": result.get("request_latency_seconds", 0),
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "total_tokens": result.get("total_tokens", 0),
        "cost_usd": result.get("cost_usd"),
        "provider": result.get("provider", ""),
        "generation_id": result.get("generation_id", ""),
        "finish_reason": result.get("finish_reason", ""),
        "attempt_count": result.get("attempt_count", 0),
        "retry_count": result.get("retry_count", 0),
        **deterministic,
        "timestamp": now_iso(),
    }


def save_results(run_id: str, results: list[dict[str, Any]]) -> tuple[Path, Path]:
    sorted_rows = sorted(results, key=lambda row: (str(row["prompt_id"]), str(row["model_name"])))
    json_path = RESULTS_DIR / f"{run_id}_results.json"
    csv_path = RESULTS_DIR / f"{run_id}_results.csv"
    write_json(json_path, sorted_rows)
    write_csv(csv_path, sorted_rows)
    return json_path, csv_path


def run_model_benchmark(
    run_id: str,
    prompts: list[dict[str, Any]],
    execution_mode: str,
) -> list[dict[str, Any]]:
    tasks = [(prompt, model) for prompt in prompts for model in MODELS]
    total = len(tasks)
    results: list[dict[str, Any]] = []

    print(f"\nΘα εκτελεστούν {total} model requests.")

    if execution_mode == "parallel":
        with concurrent.futures.ThreadPoolExecutor(max_workers=BENCHMARK_MAX_WORKERS) as executor:
            futures = {
                executor.submit(build_benchmark_row, run_id, prompt, model): (prompt, model)
                for prompt, model in tasks
            }
            for counter, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                prompt, model = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # defensive worker boundary
                    row = {
                        "run_id": run_id,
                        "prompt_id": prompt["prompt_id"],
                        "category": prompt["category"],
                        "difficulty": prompt["difficulty"],
                        "model_name": model["name"],
                        "model_id": model["id"],
                        "prompt": prompt["prompt"],
                        "response": "",
                        "status": "error",
                        "error": f"Worker error: {exc}",
                        "timestamp": now_iso(),
                    }
                results.append(row)
                save_results(run_id, results)
                print_progress(counter, total, row)
    else:
        for counter, (prompt, model) in enumerate(tasks, start=1):
            row = build_benchmark_row(run_id, prompt, model)
            results.append(row)
            save_results(run_id, results)
            print_progress(counter, total, row)
            if counter < total:
                time.sleep(PAUSE_BETWEEN_REQUESTS)

    return results


def print_progress(counter: int, total: int, row: dict[str, Any]) -> None:
    print(
        f"[{counter}/{total}] {row.get('prompt_id')} | {row.get('model_name')} | "
        f"{row.get('status')} | {row.get('latency_seconds', 0)}s | "
        f"${row.get('cost_usd') if row.get('cost_usd') is not None else 'N/A'}"
    )
    if row.get("error"):
        print(f"  Error: {row['error']}")


# ============================================================
# BLIND PAIRWISE LLM JUDGE
# ============================================================


def judge_response_format() -> dict[str, Any]:
    score_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "correctness": {"type": "integer", "minimum": 1, "maximum": 5},
            "instruction_following": {"type": "integer", "minimum": 1, "maximum": 5},
            "factuality_grounding": {"type": "integer", "minimum": 1, "maximum": 5},
            "greek_quality": {"type": "integer", "minimum": 1, "maximum": 5},
            "overall_quality": {"type": "integer", "minimum": 1, "maximum": 5},
            "hallucination_detected": {"type": "boolean"},
            "task_success": {"type": "boolean"},
            "critical_error": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": [
            "correctness",
            "instruction_following",
            "factuality_grounding",
            "greek_quality",
            "overall_quality",
            "hallucination_detected",
            "task_success",
            "critical_error",
            "rationale",
        ],
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "blind_pairwise_llm_evaluation",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "winner": {
                        "type": "string",
                        "enum": ["A", "B", "tie", "cannot_assess"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "answer_a": score_schema,
                    "answer_b": score_schema,
                    "pairwise_reason": {"type": "string"},
                },
                "required": ["winner", "confidence", "answer_a", "answer_b", "pairwise_reason"],
            },
        },
    }


def build_judge_prompt(
    prompt: dict[str, Any],
    answer_a: str,
    answer_b: str,
) -> str:
    reference = prompt.get("reference_answer")
    criteria = prompt.get("evaluation_criteria")
    accepted = prompt.get("accepted_answers")

    context = {
        "task": prompt["prompt"],
        "category": prompt.get("category"),
        "difficulty": prompt.get("difficulty"),
        "evaluation_type": prompt.get("evaluation_type"),
        "expected_format": prompt.get("expected_format"),
        "reference_answer": reference,
        "accepted_answers": accepted,
        "evaluation_criteria": criteria,
        "answer_a": answer_a,
        "answer_b": answer_b,
    }

    return (
        "Είσαι ανεξάρτητος και αυστηρός evaluator δύο ανώνυμων απαντήσεων LLM. "
        "Δεν γνωρίζεις ούτε πρέπει να υποθέσεις ποιο μοντέλο έδωσε κάθε απάντηση. "
        "Αξιολόγησε μόνο με βάση το task, τα διαθέσιμα reference στοιχεία και τα criteria. "
        "Για exact-match ή structured-output tasks, δώσε ιδιαίτερο βάρος στην ακριβή μορφή. "
        "Για grounded tasks, χαρακτήρισε ως hallucination κάθε μη υποστηριζόμενο ουσιαστικό ισχυρισμό. "
        "Μην επιβραβεύεις την περιττή έκταση. Tie επιτρέπεται όταν οι απαντήσεις είναι ουσιαστικά ισοδύναμες. "
        "Cannot_assess μόνο όταν λείπει κρίσιμη πληροφορία για δίκαιη κρίση. "
        "Οι βαθμοί είναι 1=πολύ κακό έως 5=άριστο. Επίστρεψε μόνο το ζητούμενο JSON.\n\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def ensure_independent_judge() -> None:
    evaluated_ids = {model["id"] for model in MODELS}
    if JUDGE_MODEL in evaluated_ids:
        raise ValueError(
            f"Το JUDGE_MODEL ({JUDGE_MODEL}) είναι evaluated model. "
            "Όρισε ανεξάρτητο judge στο .env."
        )


def evaluate_pair(
    run_id: str,
    prompt: dict[str, Any],
    first_row: dict[str, Any],
    second_row: dict[str, Any],
) -> dict[str, Any]:
    pair = [first_row, second_row]
    random.shuffle(pair)
    answer_a_row, answer_b_row = pair

    result = call_openrouter(
        model_id=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Αξιολόγησε αντικειμενικά και επέστρεψε αποκλειστικά έγκυρο JSON σύμφωνα με το schema.",
            },
            {
                "role": "user",
                "content": build_judge_prompt(prompt, answer_a_row["response"], answer_b_row["response"]),
            },
        ],
        settings=JUDGE_SETTINGS,
        response_format=judge_response_format(),
    )

    parsed = parse_json_response(result.get("response_text", "")) if result.get("status") == "success" else None
    if not isinstance(parsed, dict):
        return {
            "run_id": run_id,
            "prompt_id": prompt["prompt_id"],
            "status": "error",
            "error": result.get("error") or "Ο judge δεν επέστρεψε parseable JSON.",
            "judge_model": JUDGE_MODEL,
            "answer_a_model_name": answer_a_row["model_name"],
            "answer_a_model_id": answer_a_row["model_id"],
            "answer_b_model_name": answer_b_row["model_name"],
            "answer_b_model_id": answer_b_row["model_id"],
            "judge_latency_seconds": result.get("latency_seconds", 0),
            "judge_cost_usd": result.get("cost_usd"),
            "timestamp": now_iso(),
        }

    return {
        "run_id": run_id,
        "prompt_id": prompt["prompt_id"],
        "category": prompt["category"],
        "difficulty": prompt["difficulty"],
        "status": "success",
        "error": "",
        "judge_model": JUDGE_MODEL,
        "answer_a_model_name": answer_a_row["model_name"],
        "answer_a_model_id": answer_a_row["model_id"],
        "answer_b_model_name": answer_b_row["model_name"],
        "answer_b_model_id": answer_b_row["model_id"],
        "winner_label": parsed.get("winner"),
        "winner_model_name": map_winner_model(parsed.get("winner"), answer_a_row, answer_b_row),
        "confidence": parsed.get("confidence"),
        "answer_a": parsed.get("answer_a", {}),
        "answer_b": parsed.get("answer_b", {}),
        "pairwise_reason": parsed.get("pairwise_reason", ""),
        "judge_latency_seconds": result.get("latency_seconds", 0),
        "judge_input_tokens": result.get("input_tokens", 0),
        "judge_output_tokens": result.get("output_tokens", 0),
        "judge_total_tokens": result.get("total_tokens", 0),
        "judge_cost_usd": result.get("cost_usd"),
        "judge_provider": result.get("provider", ""),
        "timestamp": now_iso(),
    }


def map_winner_model(
    winner: Any,
    answer_a_row: dict[str, Any],
    answer_b_row: dict[str, Any],
) -> str:
    if winner == "A":
        return str(answer_a_row["model_name"])
    if winner == "B":
        return str(answer_b_row["model_name"])
    return str(winner or "")


def flatten_judgments(judgments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for judgment in judgments:
        if judgment.get("status") != "success":
            rows.append({
                "run_id": judgment.get("run_id"),
                "prompt_id": judgment.get("prompt_id"),
                "status": "error",
                "error": judgment.get("error"),
                "judge_model": judgment.get("judge_model"),
            })
            continue

        for label in ("a", "b"):
            score_data = judgment.get(f"answer_{label}", {})
            model_name = judgment.get(f"answer_{label}_model_name")
            model_id = judgment.get(f"answer_{label}_model_id")
            row = {
                "run_id": judgment["run_id"],
                "prompt_id": judgment["prompt_id"],
                "category": judgment.get("category"),
                "difficulty": judgment.get("difficulty"),
                "model_name": model_name,
                "model_id": model_id,
                "blind_answer_label": label.upper(),
                "winner_label": judgment.get("winner_label"),
                "winner_model_name": judgment.get("winner_model_name"),
                "is_winner": float(judgment.get("winner_model_name") == model_name),
                "is_tie": float(judgment.get("winner_label") == "tie"),
                "judge_confidence": judgment.get("confidence"),
                "hallucination_detected": score_data.get("hallucination_detected"),
                "task_success": score_data.get("task_success"),
                "critical_error": score_data.get("critical_error", ""),
                "rationale": score_data.get("rationale", ""),
                "pairwise_reason": judgment.get("pairwise_reason", ""),
                "judge_model": judgment.get("judge_model"),
                "judge_latency_seconds": judgment.get("judge_latency_seconds"),
                "judge_cost_usd": judgment.get("judge_cost_usd"),
            }
            for score in JUDGE_SCORE_FIELDS:
                row[score] = score_data.get(score)
            rows.append(row)
    return rows


def run_llm_judge(
    run_id: str,
    prompts: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ensure_independent_judge()
    prompts_by_id = {str(prompt["prompt_id"]): prompt for prompt in prompts}
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        if row.get("status") == "success" and str(row.get("response", "")).strip():
            by_prompt[str(row["prompt_id"])].append(row)

    eligible = [
        prompt_id for prompt_id, rows in by_prompt.items()
        if len(rows) == len(MODELS) and prompt_id in prompts_by_id
    ]
    judgments: list[dict[str, Any]] = []
    json_path = RESULTS_DIR / f"{run_id}_pairwise_judgments.json"
    flat_csv_path = RESULTS_DIR / f"{run_id}_judge_scores.csv"

    print(f"\nΘα γίνουν {len(eligible)} blind judge requests με {JUDGE_MODEL}.")
    for counter, prompt_id in enumerate(sorted(eligible), start=1):
        rows = by_prompt[prompt_id]
        judgment = evaluate_pair(run_id, prompts_by_id[prompt_id], rows[0], rows[1])
        judgments.append(judgment)
        write_json(json_path, judgments)
        write_csv(flat_csv_path, flatten_judgments(judgments))
        print(
            f"[Judge {counter}/{len(eligible)}] {prompt_id} | {judgment.get('status')} | "
            f"winner={judgment.get('winner_model_name', '')}"
        )
        if counter < len(eligible):
            time.sleep(PAUSE_BETWEEN_REQUESTS)

    return judgments


# ============================================================
# HUMAN REVIEW SAMPLE
# ============================================================


def create_blind_human_sample(
    run_id: str,
    prompts: list[dict[str, Any]],
    results: list[dict[str, Any]],
    sample_size: int = 30,
) -> tuple[Path, Path]:
    """Create one easy, medium and hard prompt from each category.

    The sample contains 30 prompts for the current 10-category dataset and
    balances which model appears as Answer A.
    """
    output_rows, key_rows = create_stratified_human_sample(run_id, prompts, results)
    path = RESULTS_DIR / f"{run_id}_blind_human_sample.csv"
    key_path = RESULTS_DIR / f"{run_id}_blind_human_sample_key.json"
    write_csv(path, output_rows)
    write_json(key_path, {
        "run_id": run_id,
        "warning": "Άνοιξε το κλειδί μόνο μετά την ανθρώπινη αξιολόγηση.",
        "design": "30 prompts: 1 easy, 1 medium and 1 hard from each of 10 categories; balanced A/B assignment.",
        "answers": key_rows,
    })
    return path, key_path


# ============================================================
# AGGREGATION
# ============================================================


def create_model_summary(
    run_id: str,
    results: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    judge_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judge_rows:
        if row.get("status") != "error":
            judge_by_model[str(row.get("model_name", ""))].append(row)

    summaries: list[dict[str, Any]] = []
    for model in MODELS:
        model_rows = [row for row in results if row.get("model_name") == model["name"]]
        successful = [row for row in model_rows if row.get("status") == "success"]
        judged = judge_by_model.get(model["name"], [])
        latencies = [number_or_zero(row.get("latency_seconds")) for row in successful]
        costs = [value for row in model_rows if (value := optional_float(row.get("cost_usd"))) is not None]
        exact_values = [
            float(row["deterministic_exact_match"])
            for row in model_rows
            if row.get("deterministic_exact_match") is not None
        ]
        json_values = [
            float(row["json_valid"])
            for row in model_rows
            if row.get("json_valid") is not None
        ]

        summary: dict[str, Any] = {
            "run_id": run_id,
            "model_name": model["name"],
            "model_id": model["id"],
            "request_count": len(model_rows),
            "success_count": len(successful),
            "api_success_rate": round(len(successful) / len(model_rows), 4) if model_rows else None,
            "latency_mean_seconds": mean_or_none(latencies),
            "latency_p50_seconds": percentile(latencies, 0.50),
            "latency_p95_seconds": percentile(latencies, 0.95),
            "total_cost_usd": round(sum(costs), 8) if costs else None,
            "total_input_tokens": sum(int(number_or_zero(row.get("input_tokens"))) for row in model_rows),
            "total_output_tokens": sum(int(number_or_zero(row.get("output_tokens"))) for row in model_rows),
            "total_tokens": sum(int(number_or_zero(row.get("total_tokens"))) for row in model_rows),
            "total_retries": sum(int(number_or_zero(row.get("retry_count"))) for row in model_rows),
            "exact_match_evaluated": len(exact_values),
            "exact_match_accuracy": mean_or_none(exact_values),
            "json_evaluated": len(json_values),
            "json_validity_rate": mean_or_none(json_values),
            "judge_evaluated": len(judged),
            "judge_win_count": sum(float(row.get("is_winner") or 0) for row in judged),
            "judge_tie_count": sum(float(row.get("is_tie") or 0) for row in judged),
            "judge_task_success_rate": mean_or_none([
                float(bool(row.get("task_success"))) for row in judged if row.get("task_success") is not None
            ]),
            "judge_hallucination_rate": mean_or_none([
                float(bool(row.get("hallucination_detected"))) for row in judged if row.get("hallucination_detected") is not None
            ]),
        }
        for score in JUDGE_SCORE_FIELDS:
            summary[f"judge_{score}_mean"] = mean_or_none([
                float(row[score]) for row in judged if optional_float(row.get(score)) is not None
            ])
        summaries.append(summary)

    return summaries


def create_category_summary(
    run_id: str,
    results: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted({(str(row.get("model_name")), str(row.get("category"))) for row in results})
    for model_name, category in keys:
        result_subset = [
            row for row in results
            if row.get("model_name") == model_name and row.get("category") == category
        ]
        judge_subset = [
            row for row in judge_rows
            if row.get("model_name") == model_name and row.get("category") == category
        ]
        latencies = [number_or_zero(row.get("latency_seconds")) for row in result_subset if row.get("status") == "success"]
        costs = [value for row in result_subset if (value := optional_float(row.get("cost_usd"))) is not None]
        row: dict[str, Any] = {
            "run_id": run_id,
            "model_name": model_name,
            "category": category,
            "prompts": len(result_subset),
            "success_rate": round(sum(r.get("status") == "success" for r in result_subset) / len(result_subset), 4) if result_subset else None,
            "latency_mean_seconds": mean_or_none(latencies),
            "total_cost_usd": round(sum(costs), 8) if costs else None,
            "judge_evaluated": len(judge_subset),
            "win_count": sum(float(r.get("is_winner") or 0) for r in judge_subset),
            "tie_count": sum(float(r.get("is_tie") or 0) for r in judge_subset),
        }
        for score in JUDGE_SCORE_FIELDS:
            row[f"{score}_mean"] = mean_or_none([
                float(r[score]) for r in judge_subset if optional_float(r.get(score)) is not None
            ])
        rows.append(row)
    return rows


def save_summaries(
    run_id: str,
    results: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> tuple[Path, Path, Path, Path]:
    _, prompts = load_benchmark_dataset()
    corrected_results = recalculate_results(results, prompts)
    save_results(run_id, corrected_results)
    judge_rows = flatten_judgments_v2(judgments)
    write_csv(RESULTS_DIR / f"{run_id}_judge_scores.csv", judge_rows)
    write_json(RESULTS_DIR / f"{run_id}_judge_scores.json", judge_rows)

    model_summary = create_model_summary_v2(run_id, corrected_results, judge_rows)
    category_summary = create_group_summary_v2(run_id, corrected_results, judge_rows, "category")
    difficulty_summary = create_group_summary_v2(run_id, corrected_results, judge_rows, "difficulty")
    provider_summary = create_provider_summary(run_id, corrected_results)

    model_json = RESULTS_DIR / f"{run_id}_model_summary.json"
    model_csv = RESULTS_DIR / f"{run_id}_model_summary.csv"
    category_json = RESULTS_DIR / f"{run_id}_category_summary.json"
    category_csv = RESULTS_DIR / f"{run_id}_category_summary.csv"
    write_json(model_json, model_summary)
    write_csv(model_csv, model_summary)
    write_json(category_json, category_summary)
    write_csv(category_csv, category_summary)
    write_json(RESULTS_DIR / f"{run_id}_difficulty_summary.json", difficulty_summary)
    write_csv(RESULTS_DIR / f"{run_id}_difficulty_summary.csv", difficulty_summary)
    write_json(RESULTS_DIR / f"{run_id}_provider_summary.json", provider_summary)
    write_csv(RESULTS_DIR / f"{run_id}_provider_summary.csv", provider_summary)
    return model_json, model_csv, category_json, category_csv


def sync_run_to_sqlite(
    run_id: str,
    metadata: dict[str, Any],
    prompts: list[dict[str, Any]],
    results: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> Path:
    corrected_results = recalculate_results(results, prompts)
    judge_rows = flatten_judgments_v2(judgments)
    model_summary = create_model_summary_v2(run_id, corrected_results, judge_rows)
    category_summary = create_group_summary_v2(run_id, corrected_results, judge_rows, "category")
    difficulty_summary = create_group_summary_v2(run_id, corrected_results, judge_rows, "difficulty")
    provider_summary = create_provider_summary(run_id, corrected_results)
    human_rows, human_key = create_stratified_human_sample(run_id, prompts, corrected_results)
    db_path = get_database_path()
    import_run_bundle(
        db_path=db_path,
        metadata=metadata,
        prompts=prompts,
        results=corrected_results,
        judgments=judgments,
        judge_rows=judge_rows,
        model_summary=model_summary,
        category_summary=category_summary,
        difficulty_summary=difficulty_summary,
        provider_summary=provider_summary,
        human_rows=human_rows,
        human_key=human_key,
        notes="Generated by main.py benchmark pipeline.",
    )
    return db_path


# ============================================================
# RUN MANAGEMENT
# ============================================================


def load_results(run_id: str) -> list[dict[str, Any]]:
    path = RESULTS_DIR / f"{run_id}_results.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("Το results JSON δεν περιέχει λίστα.")
    return payload


def list_run_ids() -> list[str]:
    files = sorted(RESULTS_DIR.glob("run_*_results.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [path.name.removesuffix("_results.json") for path in files]


def choose_run_id() -> str | None:
    run_ids = list_run_ids()
    if not run_ids:
        print("Δεν βρέθηκαν προηγούμενα runs.")
        return None
    print("\nΠρόσφατα runs:")
    for index, run_id in enumerate(run_ids[:10], start=1):
        print(f"{index}. {run_id}")
    raw = input("Επιλογή [1]: ").strip() or "1"
    try:
        index = int(raw)
        return run_ids[index - 1]
    except (ValueError, IndexError):
        print("Μη έγκυρη επιλογή.")
        return None


def choose_execution_mode() -> str:
    print("\n1. Sequential — σωστότερο για latency")
    print("2. Parallel — γρηγορότερο, όχι για τελικό latency")
    choice = input("Επιλογή [1]: ").strip() or "1"
    return "parallel" if choice == "2" else "sequential"


def test_connection() -> None:
    print("\nΈλεγχος Qwen, Gemma και judge...")
    test_messages = [
        {"role": "system", "content": "Απάντησε σύντομα."},
        {"role": "user", "content": "Απάντησε μόνο: OK"},
    ]
    for model in [*MODELS, {"name": "Judge", "id": JUDGE_MODEL}]:
        result = call_openrouter(model["id"], test_messages, {**BENCHMARK_SETTINGS, "temperature": 0.0, "max_tokens": 20})
        print(f"{model['name']} ({model['id']}): {result.get('status')} | {result.get('response_text', '')}")
        if result.get("error"):
            print(f"  {result['error']}")


def run_complete_pipeline() -> None:
    metadata, prompts = load_benchmark_dataset()
    ensure_independent_judge()
    print(f"\nDataset: {metadata.get('dataset_name', PROMPTS_FILE.name)}")
    print(f"Prompts: {len(prompts)} | Model responses: {len(prompts) * len(MODELS)} | Judge pairs: {len(prompts)}")
    confirm = input("Να ξεκινήσει benchmark + judge; [y/N]: ").strip().casefold()
    if confirm not in {"y", "yes", "ν", "ναι"}:
        print("Ακυρώθηκε.")
        return

    execution_mode = choose_execution_mode()
    run_id = f"run_{timestamp_id()}"
    metadata_path = RESULTS_DIR / f"{run_id}_metadata.json"
    run_metadata = {
        "run_id": run_id,
        "created_at": now_iso(),
        "dataset_metadata": metadata,
        "prompts_file": str(PROMPTS_FILE),
        "models": MODELS,
        "judge_model": JUDGE_MODEL,
        "benchmark_settings": BENCHMARK_SETTINGS,
        "judge_settings": JUDGE_SETTINGS,
        "execution_mode": execution_mode,
        "expected_model_requests": len(prompts) * len(MODELS),
        "expected_judge_requests": len(prompts),
    }
    write_json(metadata_path, run_metadata)

    try:
        results = run_model_benchmark(run_id, prompts, execution_mode)
    except KeyboardInterrupt:
        print("\nΤο benchmark διακόπηκε. Τα μέχρι τώρα αποτελέσματα έχουν αποθηκευτεί.")
        return

    human_path, human_key_path = create_blind_human_sample(run_id, prompts, results)
    print(f"\nBlind human sample: {human_path}")
    print(f"Blind human key: {human_key_path}")

    run_judge = input("Να ξεκινήσει τώρα ο ανεξάρτητος LLM judge; [Y/n]: ").strip().casefold()
    if run_judge in {"n", "no", "ο", "όχι"}:
        db_path = sync_run_to_sqlite(run_id, run_metadata, prompts, results, [])
        print("Το benchmark ολοκληρώθηκε χωρίς judge. Μπορείς να τον τρέξεις αργότερα από το menu.")
        print(f"SQLite database: {db_path}")
        return

    judgments = run_llm_judge(run_id, prompts, results)
    paths = save_summaries(run_id, results, judgments)
    print("\nΟλοκληρώθηκε το πλήρες pipeline.")
    print(f"Results: {RESULTS_DIR / f'{run_id}_results.csv'}")
    print(f"Judge scores: {RESULTS_DIR / f'{run_id}_judge_scores.csv'}")
    print(f"Model summary: {paths[1]}")
    print(f"Category summary: {paths[3]}")
    db_path = sync_run_to_sqlite(run_id, run_metadata, prompts, results, judgments)
    print(f"SQLite database: {db_path}")


def run_benchmark_only() -> None:
    metadata, prompts = load_benchmark_dataset()
    print(f"\nDataset: {metadata.get('dataset_name', PROMPTS_FILE.name)} | Prompts: {len(prompts)}")
    confirm = input("Να ξεκινήσει μόνο το benchmark; [y/N]: ").strip().casefold()
    if confirm not in {"y", "yes", "ν", "ναι"}:
        return
    run_id = f"run_{timestamp_id()}"
    mode = choose_execution_mode()
    run_metadata = {
        "run_id": run_id,
        "created_at": now_iso(),
        "dataset_metadata": metadata,
        "models": MODELS,
        "judge_model": JUDGE_MODEL,
        "execution_mode": mode,
    }
    write_json(RESULTS_DIR / f"{run_id}_metadata.json", run_metadata)
    results = run_model_benchmark(run_id, prompts, mode)
    human_path, human_key_path = create_blind_human_sample(run_id, prompts, results)
    print(f"Benchmark results: {RESULTS_DIR / f'{run_id}_results.csv'}")
    print(f"Blind human sample: {human_path}")
    print(f"Blind human key: {human_key_path}")
    db_path = sync_run_to_sqlite(run_id, run_metadata, prompts, results, [])
    print(f"SQLite database: {db_path}")


def judge_existing_run() -> None:
    run_id = choose_run_id()
    if not run_id:
        return
    _, prompts = load_benchmark_dataset()
    results = load_results(run_id)
    ensure_independent_judge()
    judgments = run_llm_judge(run_id, prompts, results)
    paths = save_summaries(run_id, results, judgments)
    metadata_path = RESULTS_DIR / f"{run_id}_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {"run_id": run_id}
    db_path = sync_run_to_sqlite(run_id, metadata, prompts, results, judgments)
    print(f"Judge scores: {RESULTS_DIR / f'{run_id}_judge_scores.csv'}")
    print(f"Model summary: {paths[1]}")
    print(f"Category summary: {paths[3]}")
    print(f"SQLite database: {db_path}")


def validate_environment() -> bool:
    if not OPENROUTER_API_KEY:
        print("\nΔεν βρέθηκε OPENROUTER_API_KEY. Αντέγραψε το .env.example σε .env και βάλε το κλειδί σου.")
        return False
    return True


def main() -> None:
    while True:
        print("\n" + "=" * 72)
        print("QWEN vs GEMMA — RESEARCH-INSPIRED LLM EVALUATION")
        print("=" * 72)
        print("1. Πλήρες benchmark + independent LLM judge")
        print("2. Benchmark μόνο")
        print("3. Judge σε υπάρχον run")
        print("4. Έλεγχος σύνδεσης Qwen / Gemma / judge")
        print("5. Έξοδος")

        choice = input("\nΕπιλογή: ").strip()
        try:
            if choice == "1" and validate_environment():
                run_complete_pipeline()
            elif choice == "2" and validate_environment():
                run_benchmark_only()
            elif choice == "3" and validate_environment():
                judge_existing_run()
            elif choice == "4" and validate_environment():
                test_connection()
            elif choice == "5":
                return
            else:
                print("Μη έγκυρη επιλογή.")
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            print(f"\nΣφάλμα: {exc}")
        except KeyboardInterrupt:
            print("\nΗ ενέργεια διακόπηκε από τον χρήστη.")


if __name__ == "__main__":
    main()
