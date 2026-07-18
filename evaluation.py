from __future__ import annotations

import csv
import io
import json
import math
import re
from typing import Any

import yaml
from jsonschema import ValidationError, validate


CODE_FENCE_RE = re.compile(r"^```(?:json|yaml|yml|csv|sql|python|html|markdown|md)?\s*|\s*```$", re.I)
NUMBER_RE = re.compile(r"[-+]?\d+(?:[\.,]\d+)?")


def strip_code_fence(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json|yaml|yml|csv|sql|python|html|markdown|md)?\s*",
            "",
            cleaned,
            flags=re.I,
        )
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def normalize_text(value: Any) -> str:
    text = strip_code_fence(str(value or "")).casefold().strip()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.,;:!?\"'`")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _numeric_values(value: Any) -> list[float]:
    text = strip_code_fence(str(value or ""))
    values: list[float] = []
    for match in NUMBER_RE.findall(text):
        try:
            values.append(float(match.replace(",", ".")))
        except ValueError:
            continue
    return values


def _numeric_match(response: str, accepted: list[Any], tolerance: float) -> float | None:
    response_values = _numeric_values(response)
    target_values: list[float] = []
    for answer in accepted:
        target_values.extend(_numeric_values(answer))
    if not response_values or not target_values:
        return None

    is_percent_target = any("%" in str(answer) for answer in accepted)
    if is_percent_target and "%" not in response:
        return 0.0

    for target in target_values:
        for observed in response_values:
            if math.isclose(observed, target, rel_tol=0.0, abs_tol=tolerance):
                return 1.0
    return 0.0


def _parse_json_strict(response: str) -> tuple[Any | None, bool, bool]:
    raw = str(response or "").strip()
    no_fence = not raw.startswith("```")
    cleaned = strip_code_fence(raw)
    try:
        return json.loads(cleaned), True, no_fence
    except json.JSONDecodeError:
        return None, False, no_fence


def _parse_yaml_strict(response: str) -> tuple[Any | None, bool, bool]:
    raw = str(response or "").strip()
    no_fence = not raw.startswith("```")
    cleaned = strip_code_fence(raw)
    try:
        parsed = yaml.safe_load(cleaned)
        return parsed, parsed is not None, no_fence
    except yaml.YAMLError:
        return None, False, no_fence


def _validate_schema(parsed: Any, schema: dict[str, Any] | None) -> float | None:
    if schema is None:
        return None
    try:
        validate(instance=parsed, schema=schema)
        return 1.0
    except ValidationError:
        return 0.0


def _csv_validation(response: str, spec: dict[str, Any]) -> tuple[float, float | None]:
    cleaned = strip_code_fence(response)
    try:
        rows = list(csv.reader(io.StringIO(cleaned)))
    except csv.Error:
        return 0.0, 0.0

    rows = [row for row in rows if any(str(cell).strip() for cell in row)]
    if not rows:
        return 0.0, 0.0

    expected_rows = spec.get("rows")
    expected_columns = spec.get("columns")
    syntax_valid = 1.0
    schema_ok = 1.0
    if expected_rows is not None and len(rows) != int(expected_rows):
        schema_ok = 0.0
    if expected_columns is not None and any(len(row) != int(expected_columns) for row in rows):
        schema_ok = 0.0
    expected_values = spec.get("expected_values")
    if expected_values is not None:
        normalized_rows = [[normalize_text(cell) for cell in row] for row in rows]
        expected_norm = [[normalize_text(cell) for cell in row] for row in expected_values]
        if normalized_rows != expected_norm:
            schema_ok = 0.0
    return syntax_valid, schema_ok


def _markdown_table_validation(response: str, spec: dict[str, Any]) -> tuple[float, float | None]:
    cleaned = strip_code_fence(response)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 2:
        return 0.0, 0.0

    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip("|").split("|")]

    header = cells(table_lines[0])
    separator = cells(table_lines[1])
    data_rows = [cells(line) for line in table_lines[2:]]
    valid_separator = all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in separator)
    syntax_valid = float(valid_separator and len(header) > 0 and all(len(row) == len(header) for row in data_rows))

    schema_ok = syntax_valid
    expected_headers = spec.get("headers")
    if expected_headers is not None and [normalize_text(x) for x in header] != [normalize_text(x) for x in expected_headers]:
        schema_ok = 0.0
    expected_rows = spec.get("data_rows")
    if expected_rows is not None and len(data_rows) != int(expected_rows):
        schema_ok = 0.0
    max_words = spec.get("max_words_per_cell")
    if max_words is not None:
        for row in [header, *data_rows]:
            if any(len(re.findall(r"\b\w+\b", cell, flags=re.UNICODE)) > int(max_words) for cell in row):
                schema_ok = 0.0
                break
    if spec.get("only_table") and len(table_lines) != len(lines):
        schema_ok = 0.0
    return syntax_valid, schema_ok


def deterministic_evaluation(prompt: dict[str, Any], response: str) -> dict[str, Any]:
    evaluation_type = str(prompt.get("evaluation_type", "")).strip()
    expected_format = str(prompt.get("expected_format", "text")).strip().lower()
    match_mode = str(prompt.get("match_mode", "")).strip().lower()
    accepted = _as_list(prompt.get("accepted_answers"))
    if not accepted and evaluation_type == "exact_match":
        accepted = _as_list(prompt.get("reference_answer"))

    strict_exact: float | None = None
    normalized_exact: float | None = None
    contains_expected: float | None = None
    numeric_match: float | None = None
    deterministic_pass: float | None = None
    deterministic_method = ""

    if accepted:
        strict_exact = float(any(str(response).strip() == str(answer).strip() for answer in accepted))
        normalized_response = normalize_text(response)
        normalized_answers = [normalize_text(answer) for answer in accepted]
        normalized_exact = float(any(normalized_response == answer for answer in normalized_answers))
        contains_expected = float(any(answer and answer in normalized_response for answer in normalized_answers))
        tolerance = float(prompt.get("numeric_tolerance", 1e-9))
        numeric_match = _numeric_match(response, accepted, tolerance)

        if match_mode == "numeric":
            deterministic_pass = numeric_match
            deterministic_method = "numeric"
        elif match_mode == "contains":
            deterministic_pass = contains_expected
            deterministic_method = "contains"
        elif match_mode == "strict":
            deterministic_pass = strict_exact
            deterministic_method = "strict"
        else:
            deterministic_pass = normalized_exact
            deterministic_method = "normalized_exact"

    structured_required = expected_format in {"json", "yaml", "csv", "markdown_table"}
    syntax_valid: float | None = None
    schema_valid: float | None = None
    no_markdown_fence: float | None = None
    format_compliance: float | None = None

    spec = prompt.get("format_spec") if isinstance(prompt.get("format_spec"), dict) else {}
    allow_fence = bool(spec.get("allow_markdown_fence", True))

    if expected_format == "json":
        parsed, syntax_bool, no_fence_bool = _parse_json_strict(response)
        syntax_valid = float(syntax_bool)
        no_markdown_fence = float(no_fence_bool)
        schema_valid = _validate_schema(parsed, spec.get("json_schema")) if syntax_bool else 0.0
        if schema_valid is None:
            schema_valid = syntax_valid
        format_compliance = float(bool(syntax_valid) and bool(schema_valid) and (allow_fence or bool(no_markdown_fence)))
    elif expected_format == "yaml":
        parsed, syntax_bool, no_fence_bool = _parse_yaml_strict(response)
        syntax_valid = float(syntax_bool)
        no_markdown_fence = float(no_fence_bool)
        schema_valid = _validate_schema(parsed, spec.get("json_schema")) if syntax_bool else 0.0
        if schema_valid is None:
            schema_valid = syntax_valid
        format_compliance = float(bool(syntax_valid) and bool(schema_valid) and (allow_fence or bool(no_markdown_fence)))
    elif expected_format == "csv":
        syntax_valid, schema_valid = _csv_validation(response, spec)
        no_markdown_fence = float(not str(response or "").strip().startswith("```"))
        format_compliance = float(bool(syntax_valid) and bool(schema_valid) and (allow_fence or bool(no_markdown_fence)))
    elif expected_format == "markdown_table":
        syntax_valid, schema_valid = _markdown_table_validation(response, spec)
        no_markdown_fence = float(not str(response or "").strip().startswith("```"))
        format_compliance = float(bool(syntax_valid) and bool(schema_valid) and (allow_fence or bool(no_markdown_fence)))

    return {
        "strict_exact_match": strict_exact,
        "normalized_exact_match": normalized_exact,
        "contains_expected_answer": contains_expected,
        "numeric_match": numeric_match,
        "deterministic_pass": deterministic_pass,
        "deterministic_method": deterministic_method,
        # Backward-compatible metric name.
        "deterministic_exact_match": deterministic_pass,
        "structured_required": structured_required,
        "structured_format": expected_format if structured_required else "",
        "syntax_valid": syntax_valid,
        "schema_valid": schema_valid,
        "no_markdown_fence": no_markdown_fence,
        "format_compliance": format_compliance,
        # Backward-compatible names, now limited to actual JSON-output tasks.
        "json_required": expected_format == "json",
        "json_valid": syntax_valid if expected_format == "json" else None,
    }
