import os
import json
import time
import re
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


INPUT_FILE = "all_model_logs_merged_fixed.csv"

OUT_FULL = "judge_scores_full.csv"
OUT_MODEL = "judge_summary_by_model.csv"
OUT_MODEL_USECASE = "judge_summary_by_model_usecase.csv"
OUT_FAILED = "judge_failed_rows.csv"

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("API_KEY")

# Stronger model for judge scoring.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen3.7-max")

# 6 models * 3 use cases * 10 prompts = 180 rows
SAMPLE_PER_MODEL_USECASE = 10

REQUEST_TIMEOUT = 180
MAX_ATTEMPTS = 3
SLEEP_BETWEEN_CALLS = 0.5


MENU_TEXT = """
Estrella del caribe menu.

Coffee:
Espresso 1.50, Double espresso 2.00, Freddo espresso 2.00,
Cappuccino 2.20, Double cappuccino 2.40, Freddo cappuccino 2.20.

Instant Coffee:
Frappe 2.00, Hot instant coffee 2.00.

Filter Coffee:
Filter coffee 2.00.

Flavored Filter Coffee:
Hazelnut cream 2.20, French vanilla 2.20.

Greek Coffee:
Single Greek coffee 1.20, Double Greek coffee 1.50.

Desserts:
Brownie 4.00, Apple pie 3.00, Cookies 1.00, Cake 1.50, Yogurt 3.50.

Mini Sandwiches:
Brioche with ham 1.80, Brioche with turkey 1.80,
Prosciutto mozzarella 2.00, Parmesan 2.00, Salmon 2.00, Milano salami 2.00.

Time for Sandwiches:
Greek sandwich 2.70, Aeros sandwich 3.00, Manouri sandwich 2.60,
Arabic pita with tuna 3.00, Arabic pita with turkey 3.00,
Guacamole sandwich 3.50, Ham - Edam sandwich 2.50,
Turkey - Edam sandwich 2.50, Smoked pork sandwich 2.50,
Goat cheese sandwich 4.00, Prosciutto mozzarella sandwich 4.00,
Salmon sandwich 4.00, Steak with mustard sauce sandwich 3.00,
Chicken sandwich 3.50.

Hot Snacks:
Toast 1.80, Ham and cheese pie 2.00, Cheese pie 1.50,
Emmental pie 2.00, Chocolate croissant 2.00, Butter croissant 1.30,
Club sandwich 4.00.
""".strip()


DOCUMENT_TEXT = """
Restaurant and Online Service Policy.

Refund Policy:
Customers can request a refund within 14 days of purchase.
Refunds are not available for used digital products.
Refund requests must include order number and short explanation.
If approved, amount returned to original payment method.
VIP customers may receive store credit instead of cash refund.

Cancellation Policy:
Cancellations accepted up to 24 hours before scheduled service.
Late cancellations may be charged 50% of service price.
Same-day cancellations reviewed by a manager.
No-show reservations may not be eligible for compensation.

Delivery Policy:
Standard delivery takes 30 to 60 minutes.
Missing item: customer should contact support within 2 hours.
Support can offer replacement, store credit, or escalation to manager.
Support agents must not promise refunds unless refund policy clearly allows it.

Account and Privacy Policy:
Customers may request deletion of account data.
Payment information is not stored.
Order history may be retained for analytics/legal compliance.
Marketing emails can be disabled from account settings.

Support Policy:
Support agents should respond politely and avoid blaming customer.
If information missing, ask for order number.
If request cannot be answered from policy, say policy does not specify it.
""".strip()


SCORE_KEYS = [
    "correctness",
    "completeness",
    "instruction_following",
    "hallucination_safety",
    "format_quality",
    "overall_quality",
]


def get_reference_text(use_case):
    if use_case == "customer_support":
        return MENU_TEXT

    if use_case == "document_understanding":
        return DOCUMENT_TEXT

    return "No reference document. Evaluate based on frontend/code quality and instruction following."


def truncate_text(text, max_chars):
    text = str(text)
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
    text = text.lower().strip()
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    return text


def extract_score_from_text(raw_text, key):
    text = str(raw_text)

    key_variants = {
        "correctness": ["correctness"],
        "completeness": ["completeness"],
        "instruction_following": ["instruction_following", "instruction following", "instruction-following"],
        "hallucination_safety": ["hallucination_safety", "hallucination safety", "hallucination-safety", "hallucination risk"],
        "format_quality": ["format_quality", "format quality", "format-quality"],
        "overall_quality": ["overall_quality", "overall quality", "overall-quality"],
    }

    variants = key_variants[key]

    # Direct JSON-like pattern: "correctness": 5
    for variant in variants:
        pattern = rf'"?{re.escape(variant)}"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))

    # Section pattern:
    # Correctness (0-5): ... Score: 5
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

    for k, v in parsed.items():
        normalized[normalize_key_name(k)] = v

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
    difficulty = row.get("difficulty", "")
    question = truncate_text(row.get("question", ""), 2000)
    answer = truncate_text(row.get("answer", ""), 4500)
    reference_text = truncate_text(get_reference_text(use_case), 3000)

    return f"""
You are a strict LLM benchmark judge.

Evaluate the assistant answer.

Use case:
{use_case}

Difficulty:
{difficulty}

Reference material:
{reference_text}

User prompt:
{question}

Assistant answer:
{answer}

Score:
- correctness: 0 to 5
- completeness: 0 to 5
- instruction_following: 0 to 5
- hallucination_safety: 0 to 5
- format_quality: 0 to 5
- overall_quality: 0 to 10

Rules:
- For customer_support, penalize invented menu items, invented prices, or unsupported promises.
- For document_understanding, penalize contradictions or unsupported policy claims.
- For website_generation, evaluate code completeness, responsiveness, cleanliness, and instruction following.
- Be strict but fair.
- Output only JSON.
- Do not write analysis.
- Do not write explanations.
- Do not use markdown.

Return exactly this JSON:
{{
  "correctness": 0,
  "completeness": 0,
  "instruction_following": 0,
  "hallucination_safety": 0,
  "format_quality": 0,
  "overall_quality": 0
}}
""".strip()


def call_judge(prompt):
    if not API_KEY:
        raise RuntimeError("Missing API_KEY. Check your .env file.")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict evaluator. Output only JSON with numeric scores."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        "temperature": 0,
        "max_tokens": 700,
    }

    response = requests.post(
        ENDPOINT,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
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


def judge_one_row(row):
    prompt = build_judge_prompt(row)
    last_error = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = call_judge(prompt)
            parsed = parse_judge_response(raw)
            return parsed, raw, ""
        except Exception as e:
            last_error = str(e)
            print(f"Attempt {attempt}/{MAX_ATTEMPTS} failed for row_id={row['row_id']}: {last_error}")
            time.sleep(3 * attempt)

    return None, "", last_error


def build_balanced_sample(df):
    # 6 models * 3 use cases * 10 rows = 180 rows
    sampled = (
        df.groupby(["model", "use_case"], group_keys=False)
        .apply(lambda g: g.sample(n=min(SAMPLE_PER_MODEL_USECASE, len(g)), random_state=42))
    )

    sampled = sampled.sample(frac=1, random_state=42).copy()
    return sampled


def main():
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f"Cannot find {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index

    df_to_judge = build_balanced_sample(df)

    results = []
    failed = []

    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Rows to judge: {len(df_to_judge)}")
    print("-" * 60)

    for _, row in df_to_judge.iterrows():
        done_count = len(results) + len(failed) + 1

        print(
            f"Judging {done_count}/{len(df_to_judge)} | "
            f"row_id={row['row_id']} | model={row['model']} | use_case={row['use_case']}"
        )

        parsed, raw_judge_response, error = judge_one_row(row)

        if parsed is None:
            failed.append({
                "row_id": row["row_id"],
                "model": row.get("model", ""),
                "prompt_id": row.get("prompt_id", ""),
                "use_case": row.get("use_case", ""),
                "difficulty": row.get("difficulty", ""),
                "error": error,
            })
            continue

        result = {
            "row_id": row["row_id"],
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
            "judge_model": JUDGE_MODEL,
            "raw_judge_response": raw_judge_response,
        }

        results.append(result)
        time.sleep(SLEEP_BETWEEN_CALLS)

    scores = pd.DataFrame(results)
    failed_df = pd.DataFrame(failed)

    scores.to_csv(OUT_FULL, index=False, encoding="utf-8-sig")
    failed_df.to_csv(OUT_FAILED, index=False, encoding="utf-8-sig")

    if len(scores) == 0:
        print("No successful judge scores were created.")
        print(f"Created: {OUT_FAILED}")
        return

    by_model = (
        scores.groupby("model")
        .agg(
            rows=("row_id", "count"),
            avg_correctness=("correctness", "mean"),
            avg_completeness=("completeness", "mean"),
            avg_instruction_following=("instruction_following", "mean"),
            avg_hallucination_safety=("hallucination_safety", "mean"),
            avg_format_quality=("format_quality", "mean"),
            avg_overall_quality=("overall_quality", "mean"),
        )
        .reset_index()
        .sort_values("avg_overall_quality", ascending=False)
    )

    by_model_usecase = (
        scores.groupby(["model", "use_case"])
        .agg(
            rows=("row_id", "count"),
            avg_correctness=("correctness", "mean"),
            avg_completeness=("completeness", "mean"),
            avg_instruction_following=("instruction_following", "mean"),
            avg_hallucination_safety=("hallucination_safety", "mean"),
            avg_format_quality=("format_quality", "mean"),
            avg_overall_quality=("overall_quality", "mean"),
        )
        .reset_index()
        .sort_values(["use_case", "avg_overall_quality"], ascending=[True, False])
    )

    by_model.to_csv(OUT_MODEL, index=False, encoding="utf-8-sig")
    by_model_usecase.to_csv(OUT_MODEL_USECASE, index=False, encoding="utf-8-sig")

    print()
    print("Done.")
    print(f"Created: {OUT_FULL}")
    print(f"Created: {OUT_MODEL}")
    print(f"Created: {OUT_MODEL_USECASE}")
    print(f"Created: {OUT_FAILED}")
    print()
    print("Judge summary by model:")
    print(by_model.to_string(index=False))

    if len(failed_df) > 0:
        print()
        print("Failed rows:")
        print(failed_df.to_string(index=False))


if __name__ == "__main__":
    main()
