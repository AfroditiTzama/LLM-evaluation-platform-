import argparse
import json
import os
import time
from datetime import datetime
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

MODELS = [
    "minimax/minimax-m2.5",
    "minimax/minimax-m2.7",
    "minimax/minimax-m3",
    "qwen/qwen3.6-plus",
    "qwen/qwen3.7-max",
    "qwen/qwen3.7-plus",
]

PRICE_PER_1M_TOKENS = {
    "minimax/minimax-m2.5": {"input": 0.30, "output": 1.20},
    "minimax/minimax-m2.7": {"input": 0.30, "output": 1.20},
    "minimax/minimax-m3": {"input": 0.30, "output": 1.20},
    "qwen/qwen3.6-plus": {"input": 0.50, "output": 3.00},
    "qwen/qwen3.7-plus": {"input": 0.40, "output": 1.60},
    "qwen/qwen3.7-max": {"input": 2.50, "output": 7.50},
}


def now_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clean_answer(answer):
    answer = "" if answer is None else str(answer)
    answer = answer.strip()

    prefixes = [
        "```json",
        "```html",
        "```jsx",
        "```javascript",
        "```js",
        "```css",
        "```python",
        "```",
    ]

    for prefix in prefixes:
        if answer.lower().startswith(prefix):
            answer = answer[len(prefix):].strip()
            break

    if answer.endswith("```"):
        answer = answer[:-3].strip()

    return answer


def extract_json_candidate(text):
    text = clean_answer(text)

    if not text:
        return ""

    first_obj = text.find("{")
    last_obj = text.rfind("}")

    first_arr = text.find("[")
    last_arr = text.rfind("]")

    candidates = []

    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        candidates.append(text[first_obj:last_obj + 1])

    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        candidates.append(text[first_arr:last_arr + 1])

    if candidates:
        return max(candidates, key=len)

    return text


def check_json_validity(answer, expected_format, question):
    expected_format = str(expected_format).lower().strip()
    question = str(question).lower()

    json_expected = (
        expected_format in ["json", "structured_json", "json_object", "json_array"]
        or "json" in question
        or "structured output" in question
    )

    if not json_expected:
        return False, None, ""

    candidate = extract_json_candidate(answer)

    try:
        json.loads(candidate)
        return True, True, ""
    except Exception as e:
        return True, False, str(e)


def infer_max_tokens(use_case, expected_format, row_max_tokens=None):
    try:
        if row_max_tokens is not None and str(row_max_tokens).strip() != "":
            return int(float(row_max_tokens))
    except Exception:
        pass

    use_case = str(use_case).lower()
    expected_format = str(expected_format).lower()

    if expected_format in ["json", "structured_json", "json_object", "json_array"]:
        return 1000

    if use_case == "website_generation":
        return 3000

    if use_case == "customer_support":
        return 700

    if use_case == "document_understanding":
        return 1000

    return 1200


def build_system_prompt(use_case, expected_format):
    use_case = str(use_case).lower()
    expected_format = str(expected_format).lower()

    if expected_format in ["json", "structured_json", "json_object", "json_array"]:
        return (
            "You are a precise assistant. Return only valid JSON. "
            "Do not include markdown. Do not include explanations outside JSON."
        )

    if use_case == "website_generation":
        return (
            "You are a frontend development assistant. Generate clean, usable, responsive code. "
            "If code is requested, return only raw code. Do not wrap code in markdown."
        )

    if use_case == "customer_support":
        return (
            "You are a careful customer support assistant. Answer politely and only use the provided information. "
            "If something is not specified, say that it is not specified."
        )

    if use_case == "document_understanding":
        return (
            "You are a document understanding assistant. Answer only from the provided reference material. "
            "If the answer is not supported by the reference, say that the document does not specify it."
        )

    return (
        "You are a helpful assistant. Follow the user instruction exactly. "
        "Be accurate, concise, and avoid unsupported claims."
    )


def build_user_message(question, reference_text):
    question = str(question).strip()
    reference_text = "" if pd.isna(reference_text) else str(reference_text).strip()

    if reference_text:
        return f"""Reference material:
{reference_text}

User prompt:
{question}"""

    return question


def call_model(model, question, use_case, expected_format, reference_text, max_tokens, temperature):
    if not API_KEY:
        raise RuntimeError("Missing API_KEY. Make sure .env contains API_KEY=...")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": build_system_prompt(use_case, expected_format),
            },
            {
                "role": "user",
                "content": build_user_message(question, reference_text),
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    start = time.time()

    response = requests.post(
        ENDPOINT,
        headers=headers,
        json=payload,
        timeout=240,
    )

    latency = time.time() - start
    response.raise_for_status()

    data = response.json()

    message = data["choices"][0].get("message", {})
    answer = message.get("content")

    if answer is None:
        answer = message.get("reasoning") or message.get("reasoning_content") or ""

    answer = clean_answer(answer)

    usage = data.get("usage", {}) or {}

    input_tokens = (
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("input")
        or 0
    )

    output_tokens = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("output")
        or 0
    )

    total_tokens = (
        usage.get("total_tokens")
        or usage.get("total")
        or int(input_tokens) + int(output_tokens)
    )

    return answer, latency, int(input_tokens), int(output_tokens), int(total_tokens)


def compute_cost(model, input_tokens, output_tokens):
    prices = PRICE_PER_1M_TOKENS.get(model)

    if not prices:
        return 0.0, 0.0, 0.0

    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    total_cost = input_cost + output_cost

    return input_cost, output_cost, total_cost


def load_prompts(args):
    if args.prompt:
        df = pd.DataFrame([
            {
                "prompt_id": "custom_prompt_001",
                "use_case": args.use_case,
                "difficulty": args.difficulty,
                "expected_format": args.expected_format,
                "question": args.prompt,
                "reference_text": args.reference_text or "",
                "max_tokens": args.max_tokens or "",
            }
        ])
        return df

    if not args.input:
        raise ValueError("Use either --prompt or --input custom_prompts.csv")

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {input_path}")

    df = pd.read_csv(input_path)

    if "question" not in df.columns:
        if "prompt" in df.columns:
            df = df.rename(columns={"prompt": "question"})
        else:
            raise ValueError("Input CSV must contain a 'question' column or a 'prompt' column.")

    if "prompt_id" not in df.columns:
        df["prompt_id"] = [f"prompt_{i+1:04d}" for i in range(len(df))]

    if "use_case" not in df.columns:
        df["use_case"] = "custom"

    if "difficulty" not in df.columns:
        df["difficulty"] = "custom"

    if "expected_format" not in df.columns:
        df["expected_format"] = "text"

    if "reference_text" not in df.columns:
        df["reference_text"] = ""

    if "max_tokens" not in df.columns:
        df["max_tokens"] = ""

    return df


def run_evaluation(prompts, output_dir, temperature):
    rows = []

    total_jobs = len(prompts) * len(MODELS)
    job_num = 0

    for _, prompt_row in prompts.iterrows():
        for model in MODELS:
            job_num += 1

            prompt_id = prompt_row.get("prompt_id", "")
            use_case = prompt_row.get("use_case", "custom")
            difficulty = prompt_row.get("difficulty", "custom")
            expected_format = prompt_row.get("expected_format", "text")
            question = prompt_row.get("question", "")
            reference_text = prompt_row.get("reference_text", "")
            row_max_tokens = prompt_row.get("max_tokens", "")

            max_tokens = infer_max_tokens(use_case, expected_format, row_max_tokens)

            print(f"[{job_num}/{total_jobs}] model={model} | prompt_id={prompt_id} | use_case={use_case}")

            error = ""
            answer = ""
            latency = 0.0
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

            for attempt in range(1, 4):
                try:
                    answer, latency, input_tokens, output_tokens, total_tokens = call_model(
                        model=model,
                        question=question,
                        use_case=use_case,
                        expected_format=expected_format,
                        reference_text=reference_text,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    error = ""
                    break

                except Exception as e:
                    error = str(e)
                    print(f"  Attempt {attempt}/3 failed: {error}")
                    time.sleep(3 * attempt)

            input_cost, output_cost, total_cost = compute_cost(
                model,
                input_tokens,
                output_tokens,
            )

            json_expected, json_valid, json_error = check_json_validity(
                answer,
                expected_format,
                question,
            )

            rows.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": model,
                "prompt_id": prompt_id,
                "use_case": use_case,
                "difficulty": difficulty,
                "expected_format": expected_format,
                "question": question,
                "reference_text": reference_text,
                "answer": answer,
                "latency_seconds": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "input_cost_usd": input_cost,
                "output_cost_usd": output_cost,
                "total_cost_usd": total_cost,
                "json_expected": json_expected,
                "json_valid": json_valid,
                "json_error": json_error,
                "error": error,
            })

            time.sleep(0.4)

    results = pd.DataFrame(rows)

    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "custom_model_logs.csv"
    results.to_csv(results_path, index=False, encoding="utf-8-sig")

    return results


def normalize_lower_is_better(series):
    series = pd.to_numeric(series, errors="coerce")

    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series([1.0] * len(series), index=series.index)

    return 1 - ((series - min_val) / (max_val - min_val))


def create_summaries(results, output_dir):
    results = results.copy()

    results["has_error"] = results["error"].fillna("").astype(str).str.strip() != ""
    results["empty_answer"] = results["answer"].fillna("").astype(str).str.strip() == ""
    results["json_expected_num"] = results["json_expected"].fillna(False).astype(bool).astype(int)
    results["json_valid_num"] = results["json_valid"].fillna(False).astype(bool).astype(int)

    by_model = (
        results.groupby("model")
        .agg(
            rows=("prompt_id", "count"),
            unique_prompts=("prompt_id", "nunique"),
            avg_latency=("latency_seconds", "mean"),
            median_latency=("latency_seconds", "median"),
            total_input_tokens=("input_tokens", "sum"),
            total_output_tokens=("output_tokens", "sum"),
            total_tokens=("total_tokens", "sum"),
            total_cost_usd=("total_cost_usd", "sum"),
            avg_cost_per_prompt_usd=("total_cost_usd", "mean"),
            errors=("has_error", "sum"),
            empty_answers=("empty_answer", "sum"),
            json_expected_count=("json_expected_num", "sum"),
            json_valid_count=("json_valid_num", "sum"),
        )
        .reset_index()
    )

    by_model["error_rate"] = by_model["errors"] / by_model["rows"]
    by_model["estimated_cost_per_1000_prompts_usd"] = by_model["avg_cost_per_prompt_usd"] * 1000
    by_model["json_valid_rate"] = by_model.apply(
        lambda r: r["json_valid_count"] / r["json_expected_count"]
        if r["json_expected_count"] > 0 else None,
        axis=1,
    )

    by_model_usecase = (
        results.groupby(["model", "use_case"])
        .agg(
            rows=("prompt_id", "count"),
            unique_prompts=("prompt_id", "nunique"),
            avg_latency=("latency_seconds", "mean"),
            total_tokens=("total_tokens", "sum"),
            total_cost_usd=("total_cost_usd", "sum"),
            avg_cost_per_prompt_usd=("total_cost_usd", "mean"),
            errors=("has_error", "sum"),
            json_expected_count=("json_expected_num", "sum"),
            json_valid_count=("json_valid_num", "sum"),
        )
        .reset_index()
    )

    by_model_usecase["error_rate"] = by_model_usecase["errors"] / by_model_usecase["rows"]
    by_model_usecase["estimated_cost_per_1000_prompts_usd"] = by_model_usecase["avg_cost_per_prompt_usd"] * 1000
    by_model_usecase["json_valid_rate"] = by_model_usecase.apply(
        lambda r: r["json_valid_count"] / r["json_expected_count"]
        if r["json_expected_count"] > 0 else None,
        axis=1,
    )

    # Preliminary recommendation before LLM-as-a-judge quality scoring.
    rec = by_model.copy()

    rec["latency_score"] = normalize_lower_is_better(rec["avg_latency"])
    rec["cost_score"] = normalize_lower_is_better(rec["estimated_cost_per_1000_prompts_usd"])
    rec["reliability_score"] = 1 - rec["error_rate"]

    # If JSON was not expected, do not penalize the model.
    rec["json_score"] = rec["json_valid_rate"].fillna(1.0)

    rec["preliminary_score"] = (
        0.30 * rec["latency_score"]
        + 0.25 * rec["cost_score"]
        + 0.25 * rec["reliability_score"]
        + 0.20 * rec["json_score"]
    )

    rec["preliminary_rank"] = rec["preliminary_score"].rank(
        ascending=False,
        method="dense",
    ).astype(int)

    rec = rec.sort_values("preliminary_rank")

    by_model.to_csv(output_dir / "custom_summary_by_model.csv", index=False, encoding="utf-8-sig")
    by_model_usecase.to_csv(output_dir / "custom_summary_by_model_usecase.csv", index=False, encoding="utf-8-sig")
    rec.to_csv(output_dir / "custom_preliminary_recommendations.csv", index=False, encoding="utf-8-sig")

    summary_txt = create_summary_text(by_model, rec)

    with open(output_dir / "custom_run_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_txt)

    return by_model, by_model_usecase, rec, summary_txt


def create_summary_text(by_model, rec):
    best = rec.sort_values("preliminary_rank").iloc[0]
    fastest = by_model.sort_values("avg_latency").iloc[0]
    cheapest = by_model.sort_values("estimated_cost_per_1000_prompts_usd").iloc[0]

    lines = []
    lines.append("Custom LLM Evaluation Run Summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append("This summary is preliminary and does not include LLM-as-a-judge quality scoring yet.")
    lines.append("")
    lines.append(f"Best preliminary model: {best['model']}")
    lines.append(f"Preliminary score: {best['preliminary_score']:.4f}")
    lines.append("")
    lines.append(f"Fastest model: {fastest['model']} ({fastest['avg_latency']:.3f}s avg latency)")
    lines.append(f"Lowest cost model: {cheapest['model']} (${cheapest['estimated_cost_per_1000_prompts_usd']:.4f} / 1000 prompts)")
    lines.append("")
    lines.append("Model summary:")
    lines.append("-" * 60)

    for _, row in by_model.sort_values("avg_latency").iterrows():
        json_info = ""
        if row["json_expected_count"] > 0:
            json_info = f" | JSON valid rate: {row['json_valid_rate']:.3f}"

        lines.append(
            f"{row['model']} | latency: {row['avg_latency']:.3f}s | "
            f"cost/1000: ${row['estimated_cost_per_1000_prompts_usd']:.4f} | "
            f"errors: {int(row['errors'])}/{int(row['rows'])}"
            f"{json_info}"
        )

    return "\n".join(lines)


def save_run_config(output_dir, args, run_id):
    config = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "models": MODELS,
        "endpoint": ENDPOINT,
        "temperature": args.temperature,
        "input": args.input,
        "single_prompt_mode": bool(args.prompt),
        "use_case": args.use_case,
        "difficulty": args.difficulty,
        "expected_format": args.expected_format,
    }

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default=None, help="CSV file with prompts.")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to evaluate.")
    parser.add_argument("--use-case", type=str, default="custom", help="Use case label for single prompt.")
    parser.add_argument("--difficulty", type=str, default="custom", help="Difficulty label for single prompt.")
    parser.add_argument("--expected-format", type=str, default="text", help="text or json.")
    parser.add_argument("--reference-text", type=str, default="", help="Optional reference material for single prompt.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional max_tokens override.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    run_id = now_run_id()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("runs") / run_id

    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args)
    prompts.to_csv(output_dir / "input_prompts.csv", index=False, encoding="utf-8-sig")

    save_run_config(output_dir, args, run_id)

    print(f"Run ID: {run_id}")
    print(f"Output directory: {output_dir}")
    print(f"Prompts: {len(prompts)}")
    print(f"Models: {len(MODELS)}")
    print("-" * 60)

    results = run_evaluation(
        prompts=prompts,
        output_dir=output_dir,
        temperature=args.temperature,
    )

    by_model, by_model_usecase, rec, summary_txt = create_summaries(results, output_dir)

    print()
    print("Done.")
    print(f"Created: {output_dir / 'custom_model_logs.csv'}")
    print(f"Created: {output_dir / 'custom_summary_by_model.csv'}")
    print(f"Created: {output_dir / 'custom_summary_by_model_usecase.csv'}")
    print(f"Created: {output_dir / 'custom_preliminary_recommendations.csv'}")
    print(f"Created: {output_dir / 'custom_run_summary.txt'}")
    print()
    print(summary_txt)


if __name__ == "__main__":
    main()
