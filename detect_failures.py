import pandas as pd
from pathlib import Path

INPUT_FILE = "all_model_logs_merged_fixed.csv"

OUT_FAILURE_ANALYSIS = "failure_analysis.csv"
OUT_MODEL_SUMMARY = "failure_summary_by_model.csv"
OUT_MODEL_USECASE_SUMMARY = "failure_summary_by_model_usecase.csv"
OUT_PROMPT_SUMMARY = "failure_summary_by_prompt.csv"

MAX_TOKENS_BY_USE_CASE = {
    "website_generation": 3000,
    "customer_support": 600,
    "document_understanding": 900,
}

MENU_ITEMS = [
    "espresso",
    "double espresso",
    "freddo espresso",
    "cappuccino",
    "double cappuccino",
    "freddo cappuccino",
    "frappe",
    "hot instant coffee",
    "filter coffee",
    "hazelnut cream",
    "french vanilla",
    "single greek coffee",
    "double greek coffee",
    "brownie",
    "apple pie",
    "cookies",
    "cake",
    "yogurt",
    "brioche with ham",
    "brioche with turkey",
    "prosciutto mozzarella",
    "parmesan",
    "salmon",
    "milano salami",
    "greek sandwich",
    "aeros sandwich",
    "manouri sandwich",
    "arabic pita with tuna",
    "arabic pita with turkey",
    "guacamole sandwich",
    "ham - edam sandwich",
    "turkey - edam sandwich",
    "smoked pork sandwich",
    "goat cheese sandwich",
    "prosciutto mozzarella sandwich",
    "salmon sandwich",
    "steak with mustard sauce sandwich",
    "chicken sandwich",
    "toast",
    "ham and cheese pie",
    "cheese pie",
    "emmental pie",
    "chocolate croissant",
    "butter croissant",
    "club sandwich",
]

NON_EXISTENT_MENU_TERMS = [
    "cheesecake",
    "to share",
    "from the sea",
    "from the land",
    "vegan section",
    "gluten-free",
    "gluten free",
    "dinner menu",
]

DOCUMENT_REQUIRED_TERMS = [
    "14 days",
    "24 hours",
    "50%",
    "30 to 60 minutes",
    "2 hours",
    "order number",
    "store credit",
    "original payment method",
    "manager",
]


def contains_any(text, terms):
    text = str(text).lower()
    return any(term.lower() in text for term in terms)


def detect_website_failures(row):
    answer = str(row["answer"])
    lower = answer.lower()
    failures = []

    if row["empty_answer"]:
        failures.append("empty_answer")

    if row["possible_truncation"]:
        failures.append("possible_truncation")

    if answer.strip().startswith("```") or "```html" in lower or "```jsx" in lower:
        failures.append("markdown_code_block")

    code_signals = [
        "<html",
        "<!doctype",
        "<section",
        "<div",
        "function ",
        "const ",
        "export default",
        "import react",
        "className=",
        "<style",
        ".container",
        "@media",
    ]

    if not contains_any(lower, code_signals):
        failures.append("no_clear_code_output")

    if "responsive" in str(row["question"]).lower():
        responsive_signals = ["@media", "grid", "flex", "responsive", "minmax", "fr"]
        if not contains_any(lower, responsive_signals):
            failures.append("missing_responsive_signal")

    return failures


def detect_customer_support_failures(row):
    answer = str(row["answer"])
    lower_answer = answer.lower()
    question = str(row["question"]).lower()
    failures = []

    if row["empty_answer"]:
        failures.append("empty_answer")

    if row["possible_truncation"]:
        failures.append("possible_truncation")

    # If prompt asks for non-existent menu section/item, answer should clearly deny or say not shown.
    if contains_any(question, NON_EXISTENT_MENU_TERMS):
        denial_signals = [
            "not shown",
            "not visible",
            "not listed",
            "not available",
            "does not include",
            "is not included",
            "no",
            "cannot confirm",
        ]
        if not contains_any(lower_answer, denial_signals):
            failures.append("possible_menu_hallucination")

    # Refund is not part of menu customer support context, so model should not promise a refund.
    if "refund" in question:
        unsafe_refund_signals = [
            "you will receive a refund",
            "we will refund",
            "refund will be issued",
            "i can process a refund",
            "you are eligible for a refund",
        ]
        if contains_any(lower_answer, unsafe_refund_signals):
            failures.append("unsafe_refund_promise")

    # If asking about price, answer should contain a number or say not shown.
    if "price" in question or "cost" in question or "how much" in question:
        has_number = any(char.isdigit() for char in answer)
        denial_signals = ["not shown", "not visible", "not listed", "not available"]
        if not has_number and not contains_any(lower_answer, denial_signals):
            failures.append("missing_price_or_denial")

    return failures


def detect_document_failures(row):
    answer = str(row["answer"])
    lower_answer = answer.lower()
    question = str(row["question"]).lower()
    failures = []

    if row["empty_answer"]:
        failures.append("empty_answer")

    if row["possible_truncation"]:
        failures.append("possible_truncation")

    # If the question asks what is not specified/unclear, answer should mention uncertainty.
    uncertainty_questions = [
        "not specified",
        "unclear",
        "what remains unclear",
        "ambiguity",
        "missing",
        "not fully covered",
        "uncertainty",
    ]

    uncertainty_signals = [
        "does not specify",
        "not specified",
        "unclear",
        "not stated",
        "not mentioned",
        "cannot determine",
        "the document does not say",
    ]

    if contains_any(question, uncertainty_questions):
        if not contains_any(lower_answer, uncertainty_signals):
            failures.append("missing_uncertainty_handling")

    # Basic document-grounding check for questions involving key policy facts.
    if "refund period" in question and "14" not in lower_answer:
        failures.append("wrong_or_missing_refund_period")

    if "late cancellation" in question and "50" not in lower_answer:
        failures.append("wrong_or_missing_late_cancellation_penalty")

    if "standard delivery" in question and ("30" not in lower_answer or "60" not in lower_answer):
        failures.append("wrong_or_missing_delivery_time")

    if "missing item" in question and "2" in question:
        # no special handling needed
        pass

    if "payment information" in question and "not stored" not in lower_answer:
        failures.append("wrong_or_missing_payment_storage_rule")

    return failures


def add_failure_labels(df):
    df["answer"] = df["answer"].fillna("").astype(str)
    df["error"] = df["error"].fillna("").astype(str)
    df["output_tokens"] = pd.to_numeric(df["output_tokens"], errors="coerce").fillna(0)
    df["latency_seconds"] = pd.to_numeric(df["latency_seconds"], errors="coerce").fillna(0)

    df["empty_answer"] = df["answer"].str.strip() == ""
    df["has_error"] = df["error"].str.strip() != ""
    df["max_tokens_expected"] = df["use_case"].map(MAX_TOKENS_BY_USE_CASE).fillna(999999)
    df["possible_truncation"] = df["output_tokens"] >= df["max_tokens_expected"]
    df["high_latency"] = df["latency_seconds"] > 30

    all_failure_labels = []
    all_failure_counts = []

    for _, row in df.iterrows():
        failures = []

        if row["has_error"]:
            failures.append("api_or_runtime_error")

        if row["high_latency"]:
            failures.append("high_latency")

        if row["use_case"] == "website_generation":
            failures.extend(detect_website_failures(row))
        elif row["use_case"] == "customer_support":
            failures.extend(detect_customer_support_failures(row))
        elif row["use_case"] == "document_understanding":
            failures.extend(detect_document_failures(row))

        # Remove duplicates while preserving order
        failures = list(dict.fromkeys(failures))

        all_failure_labels.append(";".join(failures))
        all_failure_counts.append(len(failures))

    df["failure_labels"] = all_failure_labels
    df["failure_count"] = all_failure_counts
    df["has_failure"] = df["failure_count"] > 0

    return df


def summarize_failures(df, group_cols):
    return (
        df.groupby(group_cols)
        .agg(
            rows=("prompt_id", "count"),
            failed_rows=("has_failure", "sum"),
            failure_rate=("has_failure", "mean"),
            avg_failure_count=("failure_count", "mean"),
            high_latency_count=("high_latency", "sum"),
            possible_truncations=("possible_truncation", "sum"),
            empty_answers=("empty_answer", "sum"),
        )
        .reset_index()
    )


def main():
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f"Cannot find {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    df = add_failure_labels(df)

    df.to_csv(OUT_FAILURE_ANALYSIS, index=False, encoding="utf-8-sig")

    model_summary = summarize_failures(df, ["model"])
    model_usecase_summary = summarize_failures(df, ["model", "use_case"])
    prompt_summary = summarize_failures(df, ["prompt_id", "use_case", "difficulty"])

    model_summary.to_csv(OUT_MODEL_SUMMARY, index=False, encoding="utf-8-sig")
    model_usecase_summary.to_csv(OUT_MODEL_USECASE_SUMMARY, index=False, encoding="utf-8-sig")
    prompt_summary.to_csv(OUT_PROMPT_SUMMARY, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Created: {OUT_FAILURE_ANALYSIS}")
    print(f"Created: {OUT_MODEL_SUMMARY}")
    print(f"Created: {OUT_MODEL_USECASE_SUMMARY}")
    print(f"Created: {OUT_PROMPT_SUMMARY}")

    print("\nFailure summary by model:")
    print(model_summary.to_string(index=False))


if __name__ == "__main__":
    main()
