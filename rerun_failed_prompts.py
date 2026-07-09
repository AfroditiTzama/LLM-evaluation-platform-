import os
import time
import csv
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    print("Missing API_KEY in .env file.")
    exit()

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

ERROR_FILE = "error_rows_to_rerun.csv"
MERGED_FILE = "all_model_logs_merged.csv"

RERUN_OUTPUT_FILE = "rerun_failed_results.csv"
FIXED_MERGED_FILE = "all_model_logs_merged_fixed.csv"
REMAINING_ERRORS_FILE = "error_rows_remaining.csv"


MENU_TEXT = """
Estrella del caribe

Coffee
Espresso: 1.50
Double espresso: 2.00
Freddo espresso: 2.00
Cappuccino: 2.20
Double cappuccino: 2.40
Freddo cappuccino: 2.20

Instant Coffee
Frappe: 2.00
Hot instant coffee: 2.00

Filter Coffee
Filter coffee: 2.00

Flavored Filter Coffee
Hazelnut cream: 2.20
French vanilla: 2.20

Greek Coffee
Single Greek coffee: 1.20
Double Greek coffee: 1.50

Desserts
Brownie: 4.00
Apple pie: 3.00
Cookies: 1.00
Cake: 1.50
Yogurt: 3.50

Mini Sandwiches
Brioche with ham: 1.80
Brioche with turkey: 1.80
Prosciutto mozzarella: 2.00
Parmesan: 2.00
Salmon: 2.00
Milano salami: 2.00

Time for Sandwiches
Greek sandwich: 2.70
Aeros sandwich: 3.00
Manouri sandwich: 2.60
Arabic pita with tuna: 3.00
Arabic pita with turkey: 3.00
Guacamole sandwich: 3.50
Ham - Edam sandwich: 2.50
Turkey - Edam sandwich: 2.50
Smoked pork sandwich: 2.50
Goat cheese sandwich: 4.00
Prosciutto mozzarella sandwich: 4.00
Salmon sandwich: 4.00
Steak with mustard sauce sandwich: 3.00
Chicken sandwich: 3.50

Hot Snacks
Toast: 1.80
Ham and cheese pie: 2.00
Cheese pie: 1.50
Emmental pie: 2.00
Chocolate croissant: 2.00
Butter croissant: 1.30
Club sandwich: 4.00
""".strip()


DOCUMENT_TEXT = """
Restaurant and Online Service Policy

Refund Policy:
Customers can request a refund within 14 days of purchase.
Refunds are not available for used digital products.
Refund requests must include the order number and a short explanation.
If the refund is approved, the amount is returned to the original payment method.
VIP customers may receive store credit instead of a cash refund.

Cancellation Policy:
Cancellations are accepted up to 24 hours before the scheduled service.
Late cancellations may be charged 50% of the service price.
Same-day cancellations are reviewed by a manager.
No-show reservations may not be eligible for compensation.

Delivery Policy:
Standard delivery takes 30 to 60 minutes.
If an item is missing, the customer should contact support within 2 hours.
Support can offer replacement, store credit, or escalation to a manager.
Support agents must not promise refunds unless the refund policy clearly allows it.

Account and Privacy Policy:
Customers may request deletion of their account data.
Payment information is not stored by the company.
Order history may be retained for analytics and legal compliance.
Marketing emails can be disabled from account settings.

Support Policy:
Support agents should respond politely and avoid blaming the customer.
If information is missing, the agent should ask for the order number.
If a request cannot be answered from the policy, the agent should say that the policy does not specify it.
""".strip()


SYSTEM_PROMPTS = {
    "website_generation": """
You are a frontend development assistant.
Generate clean, usable, responsive code.
Follow the user's instructions exactly.
If the user asks for code, return only raw code.
Do not wrap the code in markdown.
Do not use triple backticks.
Do not include unnecessary explanations.
""".strip(),

    "customer_support": f"""
You are a polite customer support assistant for restaurants, cafes, and food shops.
Answer menu-related questions based ONLY on the provided menu.
Do not invent products, prices, categories, ingredients, offers, or menu sections.
If something is not visible in the menu, clearly say that it is not shown in the provided menu.
Be brief, accurate, polite, and helpful.

MENU:
{MENU_TEXT}
""".strip(),

    "document_understanding": f"""
You are a document understanding assistant.
Answer only using the provided document.
Do not invent rules, exceptions, dates, prices, or conditions.
If the answer is not available in the document, say clearly that the document does not specify it.

DOCUMENT:
{DOCUMENT_TEXT}
""".strip(),
}


MAX_TOKENS_BY_USE_CASE = {
    "website_generation": 3000,
    "customer_support": 600,
    "document_understanding": 900,
}


def clean_answer(answer):
    answer = str(answer).strip()

    markdown_prefixes = [
        "```html",
        "```jsx",
        "```javascript",
        "```js",
        "```css",
        "```python",
        "```",
    ]

    for prefix in markdown_prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
            break

    if answer.endswith("```"):
        answer = answer[:-3].strip()

    return answer


def extract_answer(data):
    choices = data.get("choices", [])

    if not choices:
        return ""

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
        return "".join(parts)

    return str(content)


def extract_usage(data):
    usage = data.get("usage", {})

    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def ask_model(model, use_case, question):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    system_prompt = SYSTEM_PROMPTS[use_case]

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        "max_tokens": MAX_TOKENS_BY_USE_CASE.get(use_case, 700),
        "temperature": 0.2,
    }

    start = time.time()

    try:
        response = requests.post(
            ENDPOINT,
            headers=headers,
            json=payload,
            timeout=240,
        )
    except requests.exceptions.RequestException as e:
        latency = round(time.time() - start, 3)
        return "", latency, {"input_tokens": 0, "output_tokens": 0}, str(e)

    latency = round(time.time() - start, 3)

    if response.status_code != 200:
        return "", latency, {"input_tokens": 0, "output_tokens": 0}, f"{response.status_code}: {response.text}"

    try:
        data = response.json()
    except Exception as e:
        return "", latency, {"input_tokens": 0, "output_tokens": 0}, f"JSON parse error: {e}"

    answer = clean_answer(extract_answer(data))
    usage = extract_usage(data)

    if not answer:
        return "", latency, usage, "Empty answer"

    return answer, latency, usage, ""


def save_rerun_row(row, answer, latency, usage, error):
    file_exists = Path(RERUN_OUTPUT_FILE).exists()

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = input_tokens + output_tokens

    fieldnames = [
        "source_file",
        "timestamp",
        "model",
        "prompt_id",
        "use_case",
        "difficulty",
        "question",
        "answer",
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "error",
    ]

    with open(RERUN_OUTPUT_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "source_file": row["source_file"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": row["model"],
            "prompt_id": row["prompt_id"],
            "use_case": row["use_case"],
            "difficulty": row["difficulty"],
            "question": row["question"],
            "answer": answer,
            "latency_seconds": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "error": error,
        })


def rerun_failed_rows():
    if not Path(ERROR_FILE).exists():
        raise FileNotFoundError(f"Cannot find {ERROR_FILE}")

    failed_df = pd.read_csv(ERROR_FILE)

    if failed_df.empty:
        print("No failed rows to rerun.")
        return

    print(f"Failed rows to rerun: {len(failed_df)}")
    print("-" * 60)

    for idx, row in failed_df.iterrows():
        model = row["model"]
        prompt_id = row["prompt_id"]
        use_case = row["use_case"]
        difficulty = row["difficulty"]
        question = row["question"]

        print(f"[{idx + 1}/{len(failed_df)}] {model} | {prompt_id} | {use_case} | {difficulty}")

        answer = ""
        latency = 0
        usage = {"input_tokens": 0, "output_tokens": 0}
        error = ""

        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            print(f"Attempt {attempt}/{max_attempts}...")

            answer, latency, usage, error = ask_model(
                model=model,
                use_case=use_case,
                question=question,
            )

            if not error:
                print("Success.")
                break

            print(f"Error: {error}")

            if attempt < max_attempts:
                sleep_seconds = 20 * attempt
                print(f"Waiting {sleep_seconds}s before retry...")
                time.sleep(sleep_seconds)

        save_rerun_row(
            row=row,
            answer=answer,
            latency=latency,
            usage=usage,
            error=error,
        )

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens

        print(
            f"Latency: {latency}s | "
            f"Tokens: In {input_tokens} / Out {output_tokens} / Total {total_tokens}"
        )

        if error:
            print(f"Final status: FAILED | {error}")
        else:
            print(f"Answer preview: {answer[:150].replace(chr(10), ' ')}")

        print("-" * 60)

        time.sleep(8)

    print(f"Rerun results saved to: {RERUN_OUTPUT_FILE}")


def create_fixed_merged_file():
    if not Path(MERGED_FILE).exists():
        print(f"{MERGED_FILE} not found. Skipping fixed merged file creation.")
        return

    if not Path(RERUN_OUTPUT_FILE).exists():
        print(f"{RERUN_OUTPUT_FILE} not found. Skipping fixed merged file creation.")
        return

    merged_df = pd.read_csv(MERGED_FILE)
    rerun_df = pd.read_csv(RERUN_OUTPUT_FILE)

    successful_reruns = rerun_df[
        rerun_df["error"].fillna("").astype(str).str.strip() == ""
    ].copy()

    print(f"Successful reruns: {len(successful_reruns)}")

    for _, row in successful_reruns.iterrows():
        mask = (
            (merged_df["model"] == row["model"]) &
            (merged_df["prompt_id"] == row["prompt_id"])
        )

        if mask.sum() == 0:
            print(f"Warning: no matching row found for {row['model']} / {row['prompt_id']}")
            continue

        for col in [
            "timestamp",
            "answer",
            "latency_seconds",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "error",
        ]:
            merged_df.loc[mask, col] = row[col]

    merged_df.to_csv(FIXED_MERGED_FILE, index=False, encoding="utf-8-sig")
    print(f"Fixed merged file saved to: {FIXED_MERGED_FILE}")

    remaining_errors = merged_df[
        merged_df["error"].fillna("").astype(str).str.strip() != ""
    ].copy()

    remaining_errors.to_csv(REMAINING_ERRORS_FILE, index=False, encoding="utf-8-sig")
    print(f"Remaining errors saved to: {REMAINING_ERRORS_FILE}")
    print(f"Remaining errors count: {len(remaining_errors)}")


def main():
    rerun_failed_rows()
    create_fixed_merged_file()


if __name__ == "__main__":
    main()
