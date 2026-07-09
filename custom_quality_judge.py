import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("API_KEY")
DEFAULT_JUDGE_MODEL = "qwen/qwen3.7-max"

SCORE_KEYS = [
    "correctness",
    "completeness",
    "instruction_following",
    "hallucination_safety",
    "format_quality",
    "overall_quality",
]


def latest_run_dir():
    runs_dir = Path("runs")

    if not runs_dir.exists():
        raise FileNotFoundError("No runs directory found.")

    candidates = [
        p for p in runs_dir.iterdir()
        if p.is_dir() and (p / "custom_model_logs.csv").exists()
    ]

    if not candidates:
        raise FileNotFoundError("No run folder with custom_model_logs.csv found.")

    return sorted(candidates)[-1]


def truncate_text(text, max_chars):
    text = "" if pd.isna(text) else str(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n[TRUNCATED_FOR_JUDGE]"


def clean_json_text(text):
    text = str(text).strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()

    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0).strip()

    return text


def normalize_key_name(text):
    text = str(text).lower().strip()
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    return text


def extract_score_from_text(raw_text, key):
    text = str(raw_text)

    variants = {
        "correctness": ["correctness"],
        "completeness": ["completeness"],
        "instruction_following": ["instruction_following", "instruction following"],
        "hallucination_safety": ["hallucination_safety", "hallucination safety", "hallucination risk"],
        "format_quality": ["format_quality", "format quality"],
        "overall_quality": ["overall_quality", "overall quality"],
    }[key]

    for variant in variants:
        pattern = rf'"?{re.escape(variant)}"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))

    for variant in variants:
        pos = re.search(re.escape(variant), text, flags=re.IGNORECASE)
        if not pos:
            continue

        block = text[pos.start():pos.start() + 700]

        score_match = re.search(
            r'score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
            block,
            flags=re.IGNORECASE,
        )

        if score_match:
            return float(score_match.group(1))

        slash_match = re.search(
            r'([0-9]+(?:\.[0-9]+)?)\s*/\s*(5|10)',
            block,
            flags=re.IGNORECASE,
        )

        if slash_match:
            return float(slash_match.group(1))

    return None


def parse_judge_response(raw_text):
    cleaned = clean_json_text(raw_text)

    parsed = {}

    try:
        loaded = json.loads(cleaned)
        if isinstance(loaded, dict):
            parsed = loaded
    except Exception:
        parsed = {}

    normalized = {}

    for key, value in parsed.items():
        normalized[normalize_key_name(key)] = value

    final = {}

    for key in SCORE_KEYS:
        if key in normalized:
            final[key] = normalized[key]
        else:
            extracted = extract_score_from_text(raw_text, key)
            if extracted is not None:
                final[key] = extracted

    missing = [key for key in SCORE_KEYS if key not in final]

    if missing:
        raise ValueError(
            "Could not parse judge scores. Missing keys: "
            + ", ".join(missing)
            + " | Raw response: "
            + str(raw_text)[:500]
        )

    return final


def clamp_score(value, min_value, max_value):
    try:
        value = float(value)
    except Exception:
        return 0.0

    return max(min_value, min(max_value, value))


def build_judge_prompt(row):
    use_case = row.get("use_case", "")
    expected_format = row.get("expected_format", "text")
    question = truncate_text(row.get("question", ""), 2500)
    reference_text = truncate_text(row.get("reference_text", ""), 3500)
    answer = truncate_text(row.get("answer", ""), 5000)
    json_expected = row.get("json_expected", False)
    json_valid = row.get("json_valid", None)
    json_error = row.get("json_error", "")

    return f"""
You are a strict LLM evaluation judge.

Evaluate the model answer.

Use case:
{use_case}

Expected format:
{expected_format}

JSON expected:
{json_expected}

JSON valid:
{json_valid}

JSON error:
{json_error}

Reference material:
{reference_text}

User prompt:
{question}

Model answer:
{answer}

Score:
- correctness: 0 to 5
- completeness: 0 to 5
- instruction_following: 0 to 5
- hallucination_safety: 0 to 5
- format_quality: 0 to 5
- overall_quality: 0 to 10

Rules:
- Penalize unsupported claims.
- If reference material exists, judge only against that reference.
- If JSON was expected and the answer is not valid JSON, penalize instruction_following and format_quality.
- If JSON was expected and valid JSON was returned, reward format_quality.
- Be strict but fair.
- Output only JSON.
- Do not write explanations.
- Do not use markdown.

Return exactly:
{{
  "correctness": 0,
  "completeness": 0,
  "instruction_following": 0,
  "hallucination_safety": 0,
  "format_quality": 0,
  "overall_quality": 0
}}
""".strip()


def call_judge(prompt, judge_model):
    if not API_KEY:
        raise RuntimeError("Missing API_KEY. Check .env file.")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": judge_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict evaluator. Output only JSON with numeric scores.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0,
        "max_tokens": 700,
    }

    response = requests.post(
        ENDPOINT,
        headers=headers,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()
    data = response.json()

    message = data["choices"][0].get("message", {})
    content = message.get("content")

    if content is None:
        content = message.get("reasoning") or message.get("reasoning_content")

    if isinstance(content, list):
        content = " ".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in content
        )

    if content is None:
        raise ValueError("API returned no message content. Full response: " + str(data)[:1000])

    return content


def judge_one_row(row, judge_model):
    prompt = build_judge_prompt(row)
    last_error = ""

    for attempt in range(1, 4):
        try:
            raw = call_judge(prompt, judge_model)
            parsed = parse_judge_response(raw)
            return parsed, raw, ""
        except Exception as e:
            last_error = str(e)
            print(f"  Attempt {attempt}/3 failed for model={row.get('model')}: {last_error}")
            time.sleep(3 * attempt)

    return None, "", last_error


def normalize_lower_is_better(series):
    series = pd.to_numeric(series, errors="coerce")
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series([1.0] * len(series), index=series.index)

    return 1 - ((series - min_val) / (max_val - min_val))


def normalize_higher_is_better(series):
    series = pd.to_numeric(series, errors="coerce")
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series([1.0] * len(series), index=series.index)

    return (series - min_val) / (max_val - min_val)


def create_final_recommendations(run_dir, scores):
    summary_path = run_dir / "custom_summary_by_model.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")

    summary = pd.read_csv(summary_path)

    judge_summary = (
        scores.groupby("model")
        .agg(
            judged_rows=("model", "count"),
            avg_correctness=("correctness", "mean"),
            avg_completeness=("completeness", "mean"),
            avg_instruction_following=("instruction_following", "mean"),
            avg_hallucination_safety=("hallucination_safety", "mean"),
            avg_format_quality=("format_quality", "mean"),
            avg_overall_quality=("overall_quality", "mean"),
        )
        .reset_index()
    )

    final = summary.merge(judge_summary, on="model", how="left")

    final["quality_score"] = normalize_higher_is_better(final["avg_overall_quality"])
    final["hallucination_score"] = normalize_higher_is_better(final["avg_hallucination_safety"])
    final["latency_score"] = normalize_lower_is_better(final["avg_latency"])
    final["cost_score"] = normalize_lower_is_better(final["estimated_cost_per_1000_prompts_usd"])
    final["reliability_score"] = 1 - final["error_rate"]

    if "json_valid_rate" in final.columns:
        final["json_score"] = final["json_valid_rate"].fillna(1.0)
    else:
        final["json_score"] = 1.0

    final["final_score"] = (
        0.35 * final["quality_score"]
        + 0.20 * final["hallucination_score"]
        + 0.15 * final["latency_score"]
        + 0.15 * final["cost_score"]
        + 0.10 * final["reliability_score"]
        + 0.05 * final["json_score"]
    )

    final["final_rank"] = final["final_score"].rank(
        ascending=False,
        method="dense",
    ).astype(int)

    final = final.sort_values("final_rank")

    final.to_csv(run_dir / "custom_final_recommendations.csv", index=False, encoding="utf-8-sig")
    judge_summary.to_csv(run_dir / "custom_judge_summary_by_model.csv", index=False, encoding="utf-8-sig")

    return final, judge_summary


def create_final_summary(run_dir, final, judge_model):
    best = final.sort_values("final_rank").iloc[0]
    best_quality = final.sort_values("avg_overall_quality", ascending=False).iloc[0]
    fastest = final.sort_values("avg_latency").iloc[0]
    cheapest = final.sort_values("estimated_cost_per_1000_prompts_usd").iloc[0]

    lines = []
    lines.append("Custom LLM Evaluation Final Summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Judge model: {judge_model}")
    lines.append("")
    lines.append(f"Best final model: {best['model']}")
    lines.append(f"Final score: {best['final_score']:.4f}")
    lines.append(f"Judge quality: {best['avg_overall_quality']:.3f}/10")
    lines.append(f"Hallucination safety: {best['avg_hallucination_safety']:.3f}/5")
    lines.append(f"Average latency: {best['avg_latency']:.3f}s")
    lines.append(f"Estimated cost / 1000 prompts: ${best['estimated_cost_per_1000_prompts_usd']:.4f}")
    lines.append("")
    lines.append("Specialized winners")
    lines.append("-" * 60)
    lines.append(f"Best quality: {best_quality['model']} ({best_quality['avg_overall_quality']:.3f}/10)")
    lines.append(f"Fastest: {fastest['model']} ({fastest['avg_latency']:.3f}s)")
    lines.append(f"Lowest cost: {cheapest['model']} (${cheapest['estimated_cost_per_1000_prompts_usd']:.4f} / 1000 prompts)")
    lines.append("")
    lines.append("Model ranking")
    lines.append("-" * 60)

    for _, row in final.iterrows():
        lines.append(
            f"{int(row['final_rank'])}. {row['model']} | "
            f"final={row['final_score']:.4f} | "
            f"quality={row['avg_overall_quality']:.3f}/10 | "
            f"latency={row['avg_latency']:.3f}s | "
            f"cost/1000=${row['estimated_cost_per_1000_prompts_usd']:.4f}"
        )

    text = "\n".join(lines)

    with open(run_dir / "custom_final_summary.txt", "w", encoding="utf-8") as f:
        f.write(text)

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = latest_run_dir()

    logs_path = run_dir / "custom_model_logs.csv"

    if not logs_path.exists():
        raise FileNotFoundError(f"Cannot find {logs_path}")

    logs = pd.read_csv(logs_path)

    results = []
    failed = []

    print(f"Run directory: {run_dir}")
    print(f"Judge model: {args.judge_model}")
    print(f"Rows to judge: {len(logs)}")
    print("-" * 60)

    for i, row in logs.iterrows():
        print(f"[{i + 1}/{len(logs)}] judging model={row.get('model')} | prompt_id={row.get('prompt_id')}")

        parsed, raw, error = judge_one_row(row, args.judge_model)

        if parsed is None:
            failed.append({
                "model": row.get("model", ""),
                "prompt_id": row.get("prompt_id", ""),
                "error": error,
            })
            continue

        results.append({
            "model": row.get("model", ""),
            "prompt_id": row.get("prompt_id", ""),
            "use_case": row.get("use_case", ""),
            "difficulty": row.get("difficulty", ""),
            "correctness": clamp_score(parsed.get("correctness", 0), 0, 5),
            "completeness": clamp_score(parsed.get("completeness", 0), 0, 5),
            "instruction_following": clamp_score(parsed.get("instruction_following", 0), 0, 5),
            "hallucination_safety": clamp_score(parsed.get("hallucination_safety", 0), 0, 5),
            "format_quality": clamp_score(parsed.get("format_quality", 0), 0, 5),
            "overall_quality": clamp_score(parsed.get("overall_quality", 0), 0, 10),
            "judge_model": args.judge_model,
            "raw_judge_response": raw,
        })

        time.sleep(0.4)

    scores = pd.DataFrame(results)
    failed_df = pd.DataFrame(failed)

    scores.to_csv(run_dir / "custom_judge_scores.csv", index=False, encoding="utf-8-sig")
    failed_df.to_csv(run_dir / "custom_judge_failed_rows.csv", index=False, encoding="utf-8-sig")

    if len(scores) == 0:
        print("No judge scores were created.")
        return

    final, judge_summary = create_final_recommendations(run_dir, scores)
    final_summary = create_final_summary(run_dir, final, args.judge_model)

    print()
    print("Done.")
    print(f"Created: {run_dir / 'custom_judge_scores.csv'}")
    print(f"Created: {run_dir / 'custom_judge_summary_by_model.csv'}")
    print(f"Created: {run_dir / 'custom_judge_failed_rows.csv'}")
    print(f"Created: {run_dir / 'custom_final_recommendations.csv'}")
    print(f"Created: {run_dir / 'custom_final_summary.txt'}")
    print()
    print(final_summary)


if __name__ == "__main__":
    main()
