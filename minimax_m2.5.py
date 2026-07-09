import os
import time
import csv
from pathlib import Path
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    print("Missing API_KEY in .env file.")
    exit()

MODEL = "minimax/minimax-m2.5"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Fixed log file so the old benchmark is not overwritten
LOG_FILE = "chat_logs_minimax_m2.5.csv"


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


PROMPT_BANK = {
    "website_generation": {
        "easy": [
            "Create a simple responsive HTML and CSS landing page for a cafe with a hero section, opening hours, and contact section.",
            "Create a responsive navigation bar using HTML and CSS.",
            "Create a simple product card using HTML and CSS.",
            "Create a basic footer for a restaurant website using HTML and CSS.",
            "Create a responsive menu section for a cafe website.",
            "Create a simple contact form using HTML and CSS.",
            "Create a hero section for a bakery website.",
            "Create a simple pricing section with three cards.",
            "Create a basic about-us section for a local coffee shop.",
            "Create a responsive image gallery layout using HTML and CSS.",
            "Create a simple restaurant header with logo, navigation links, and reservation button.",
            "Create a simple HTML section showing three popular dishes.",
            "Create a basic CSS card layout for menu items.",
            "Create a responsive opening-hours section for a cafe.",
            "Create a simple landing page section for a coffee subscription service.",
            "Create a simple HTML and CSS button group for online ordering.",
            "Create a simple responsive testimonial card.",
            "Create a simple cafe website banner with title, subtitle, and call-to-action button.",
            "Create a basic two-column section with text on the left and image placeholder on the right.",
            "Create a simple responsive contact information section with phone, email, and address.",
        ],
        "medium": [
            "Create a responsive restaurant landing page with hero, menu highlights, testimonials, and footer. Return only HTML and CSS.",
            "Create a React component for a product card with image placeholder, title, description, price, and button.",
            "Create a responsive FAQ section using HTML and CSS with five questions and answers.",
            "Create a responsive reservation form with name, date, time, number of people, and comments.",
            "Create a React component for a cafe menu preview with three categories and product cards.",
            "Create a responsive pricing table for three subscription plans using HTML and CSS.",
            "Create a restaurant homepage section with call-to-action buttons and menu preview.",
            "Create a responsive two-column layout for a food delivery website.",
            "Create a React component for customer reviews with three review cards.",
            "Create a responsive newsletter signup section for a restaurant website.",
            "Create a responsive landing page for a brunch cafe with hero, featured meals, and booking button.",
            "Create a React component for a menu category section that receives items as an array.",
            "Create an HTML and CSS section for restaurant promotions with three offer cards.",
            "Create a responsive grid layout for six food products with prices.",
            "Create a restaurant contact page section with map placeholder and contact form.",
            "Create a React component for an order summary card with subtotal, delivery fee, and total.",
            "Create a responsive staff/team section for a local restaurant.",
            "Create a responsive HTML/CSS page section for customer support with FAQ and contact form.",
            "Create a React component for a reusable button and use it in a hero section.",
            "Create a responsive comparison section for three restaurant service options: dine-in, takeaway, delivery.",
        ],
        "hard": [
            "Create a responsive React landing page for a cafe with reusable components: Header, Hero, MenuPreview, OpeningHours, and Footer.",
            "Create a complete HTML and CSS page for a restaurant reservation system with form validation hints.",
            "Create a responsive dashboard layout with sidebar, top bar, statistics cards, and an orders table. Return only HTML and CSS.",
            "Create a React page for an online food shop with product grid, category filters, and cart summary UI.",
            "Create a responsive admin dashboard for restaurant orders with status badges and summary cards.",
            "Create a React component structure for a customer support portal with ticket list, ticket details, and reply box.",
            "Create a full responsive landing page for an LLM benchmarking platform with hero, metrics, comparison table, and CTA.",
            "Create a responsive restaurant website section that includes menu categories, featured item, and reservation CTA.",
            "Create a React component for a model comparison table with latency, quality score, cost, and recommendation badge.",
            "Create a responsive HTML and CSS layout for a document analysis app with upload area, summary panel, and extracted fields.",
            "Create a React restaurant booking page with reusable components for form fields, booking summary, and confirmation message.",
            "Create a responsive food delivery homepage with header, category filters, product cards, and sticky cart summary.",
            "Create a React admin interface for managing menu items with table rows, edit buttons, and status labels.",
            "Create a responsive customer support dashboard with ticket statistics, priority list, and recent messages section.",
            "Create a complete responsive HTML and CSS page for a cafe loyalty program with hero, benefits, tiers, and signup form.",
            "Create a React component for a document upload and analysis workflow with upload area, loading state, and results section.",
            "Create a responsive landing page for a SaaS tool that compares LLM models, including pricing cards and benchmark charts placeholders.",
            "Create a React page with components for Header, Sidebar, MetricsCards, BenchmarkTable, and Footer.",
            "Create a complete HTML and CSS restaurant website homepage with hero, about, menu preview, reviews, reservation, and footer.",
            "Create a responsive web page for an online ordering system with menu list, item details, quantity controls, and checkout summary UI.",
        ],
    },

    "customer_support": {
        "easy": [
            "Does the menu include Freddo espresso? If yes, what is the price?",
            "How much does the Brownie cost?",
            "Is there a chicken sandwich on the menu?",
            "What coffees are listed under the Coffee category?",
            "Does the menu include Greek coffee?",
            "How much does the Toast cost?",
            "Is there an Apple pie on the menu?",
            "What desserts are available?",
            "How much does a Double cappuccino cost?",
            "Does the menu include a Salmon sandwich?",
            "How much does a Single Greek coffee cost?",
            "Is there a Filter coffee on the menu?",
            "What items are listed under Hot Snacks?",
            "How much does the Cheese pie cost?",
            "Does the menu include Frappe?",
            "What flavored filter coffees are available?",
            "How much does the Club sandwich cost?",
            "Does the menu include Yogurt?",
            "How much does the Butter croissant cost?",
            "Which mini sandwiches are listed on the menu?",
        ],
        "medium": [
            "A customer wants a sandwich with fish. Which option from the menu would you suggest?",
            "A customer asks for a dessert under 2 euros. Which items can you recommend?",
            "A customer asks if there is a product called Cheesecake. Answer based only on the menu.",
            "A customer wants a cold coffee and a dessert. Suggest one valid combination from the menu.",
            "A customer wants a sandwich that costs exactly 4 euros. Which options are available?",
            "A customer asks whether there is a vegan section. Answer carefully based only on the menu.",
            "A customer wants something with turkey. Which menu items can you suggest?",
            "A customer wants the cheapest coffee option. What should you recommend?",
            "A customer asks for the most expensive dessert. Answer based only on the menu.",
            "A customer wants a snack and a coffee under 4 euros total. Suggest a valid combination.",
            "A customer wants a mini sandwich that costs 2 euros. Which options are available?",
            "A customer wants something sweet but not more than 1.50 euros. What can you suggest?",
            "A customer wants a hot snack and asks for the cheapest option. What should you recommend?",
            "A customer asks whether the menu has seafood dishes. Answer carefully based on visible menu items.",
            "A customer wants a coffee for exactly 2 euros. Which options are available?",
            "A customer wants a sandwich with cheese. Which visible options might fit?",
            "A customer asks for the price difference between Espresso and Double espresso.",
            "A customer wants one coffee and one dessert for exactly 5 euros. Find a valid combination if possible.",
            "A customer asks if there is a section called From the Sea. Answer based only on the menu.",
            "A customer wants a recommendation for breakfast using only visible menu items.",
        ],
        "hard": [
            "A customer wants something sweet and a cold coffee, but does not want to spend more than 5 euros. Suggest a valid combination from the menu.",
            "A customer asks for vegan options. Answer carefully based only on the menu and avoid inventing ingredients.",
            "A customer says their order was wrong and asks for a refund. Write a polite support response without promising a refund policy that is not provided.",
            "A customer wants one sandwich, one dessert, and one coffee with a total budget of 8 euros. Suggest a valid combination and calculate the total.",
            "A customer asks for the To Share section. If it is not visible, say so and suggest available snack alternatives.",
            "A customer wants a high-protein option but the menu does not list nutrition facts. Answer safely using only visible information.",
            "A customer asks whether the Salmon item is a sandwich or mini sandwich. Explain based on the menu categories.",
            "A customer asks for a recommendation for someone who does not eat pork. Answer carefully using only visible menu names.",
            "A customer complains that the Cheesecake was missing from the order. Respond based only on the provided menu.",
            "A customer wants the cheapest possible order with one coffee and one food item. Calculate a valid option.",
            "A customer wants two drinks and two food items for under 10 euros. Suggest a valid combination and calculate the total.",
            "A customer asks whether the menu contains gluten-free items. Answer without inventing dietary information.",
            "A customer wants a cold coffee, a mini sandwich, and a dessert for under 6 euros. Suggest a valid combination if possible.",
            "A customer says: I do not eat meat, but I want a sandwich. Suggest only options whose names do not visibly contain meat or fish.",
            "A customer wants the most expensive possible combination of one coffee and one dessert. Calculate the total.",
            "A customer asks if the restaurant has a dinner menu with From the Land and From the Sea sections. Answer based only on the visible menu.",
            "A customer wants one item from Coffee, one from Desserts, and one from Hot Snacks for under 7 euros. Suggest a valid order.",
            "A customer asks for an item with tuna and wants to know whether it is in a sandwich category. Answer from the menu.",
            "A customer asks for a polite recommendation for a child, but the menu does not include age guidance. Answer safely using only item names and prices.",
            "A customer wants to avoid pork and asks about Brioche with ham, Milano salami, and Chicken sandwich. Explain what can and cannot be safely recommended from the visible names.",
        ],
    },

    "document_understanding": {
        "easy": [
            "What is the refund period according to the document?",
            "Are cancellations accepted according to the document?",
            "What happens with late cancellations?",
            "Does the company store payment information?",
            "How long does standard delivery take?",
            "What should a customer do if an item is missing?",
            "Are refunds available for used digital products?",
            "What can VIP customers receive instead of a cash refund?",
            "Can customers request deletion of their account data?",
            "Why may order history be retained?",
            "What must refund requests include?",
            "What happens if a refund is approved?",
            "Can marketing emails be disabled?",
            "What should support agents avoid doing?",
            "Who reviews same-day cancellations?",
            "What can support offer for a missing item?",
            "What payment data rule is stated in the document?",
            "What should support ask for if information is missing?",
            "Are no-show reservations always eligible for compensation?",
            "What is the document's rule about blaming the customer?",
        ],
        "medium": [
            "Summarize the refund and cancellation rules in three bullet points.",
            "Can a customer get a refund for a used digital product? Answer based only on the document.",
            "What exception is mentioned for VIP customers?",
            "Extract the delivery time and missing item rule from the document.",
            "Summarize the privacy policy in two sentences.",
            "What support options are available when an item is missing?",
            "Explain when a late cancellation may be charged.",
            "List all cases where the customer may not receive a normal cash refund.",
            "What information is not stored by the company?",
            "Compare the refund policy and cancellation policy in a short answer.",
            "Explain what information a customer needs to provide for a refund request.",
            "Summarize the support policy in three short points.",
            "What does the document say about no-show reservations and compensation?",
            "Extract all customer time limits mentioned in the document.",
            "Explain the difference between standard delivery time and missing item reporting time.",
            "Can support agents promise refunds for missing items? Answer based only on the document.",
            "What does the document say about account data deletion and order history retention?",
            "Explain how same-day cancellations are handled.",
            "Create a short list of actions support can take when an item is missing.",
            "Which parts of the policy mention manager involvement or escalation?",
        ],
        "hard": [
            "A customer cancelled 10 hours before the scheduled service. Explain what may happen according to the document.",
            "Extract the refund period, cancellation deadline, late cancellation penalty, and VIP exception from the document.",
            "Identify two cases where the customer may not receive a normal cash refund.",
            "A customer says an item was missing but contacts support after 5 hours. What does the document clearly say, and what is not specified?",
            "A VIP customer requests a cash refund. Explain what the document says and what remains unclear.",
            "A customer asks whether payment data can be deleted. Answer only using the privacy policy.",
            "Create a structured summary of the document with sections: Refunds, Cancellations, Delivery, Privacy.",
            "Identify all time limits mentioned in the document and explain what each one refers to.",
            "A customer used a digital product and asks for a refund after 7 days. Explain whether the refund is available according to the document.",
            "Find potential ambiguity in the policy document and explain what information is missing.",
            "A customer requests a refund but does not provide an order number. What does the document require and what should support do?",
            "A customer wants a refund for a missing delivery item. Explain what support can offer and what it must not promise.",
            "A customer wants account deletion but asks whether order history will disappear completely. Answer with the privacy policy and uncertainty.",
            "A customer cancelled on the same day and asks if they will definitely be charged 50%. Explain the difference between same-day and late cancellation rules.",
            "A no-show customer asks for compensation. Explain what the document says and what remains unspecified.",
            "A customer reports a missing item within 90 minutes. What valid support actions are listed in the document?",
            "A customer asks to disable marketing emails and delete payment information. Explain what the document states for both requests.",
            "A support agent wants to blame the customer for a missing item. Explain why this conflicts with the support policy.",
            "Compare refund, cancellation, delivery, and privacy policies and identify the most important customer obligations.",
            "Create a careful answer for a customer whose case is not fully covered by the policy, without inventing rules.",
        ],
    },
}


def build_benchmark_prompts():
    prompts = []

    for use_case, difficulty_dict in PROMPT_BANK.items():
        for difficulty, prompt_list in difficulty_dict.items():
            for i, prompt in enumerate(prompt_list, start=1):
                prompts.append({
                    "prompt_id": f"{use_case}_{difficulty}_{i:02d}",
                    "use_case": use_case,
                    "difficulty": difficulty,
                    "prompt": prompt,
                    "max_tokens": MAX_TOKENS_BY_USE_CASE.get(use_case, 700),
                })

    return prompts


def validate_prompt_counts():
    prompts = build_benchmark_prompts()

    if len(prompts) != 180:
        raise ValueError(f"Expected 180 prompts, found {len(prompts)}")

    for use_case in ["website_generation", "customer_support", "document_understanding"]:
        for difficulty in ["easy", "medium", "hard"]:
            count = sum(
                1 for p in prompts
                if p["use_case"] == use_case and p["difficulty"] == difficulty
            )
            if count != 20:
                raise ValueError(
                    f"Expected 20 prompts for {use_case}/{difficulty}, found {count}"
                )


def clean_answer(answer):
    answer = answer.strip()

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


def save_log(prompt_item, answer, latency, usage, error=""):
    file_exists = Path(LOG_FILE).exists()

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = input_tokens + output_tokens

    with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": MODEL,
            "prompt_id": prompt_item["prompt_id"],
            "use_case": prompt_item["use_case"],
            "difficulty": prompt_item["difficulty"],
            "question": prompt_item["prompt"],
            "answer": answer,
            "latency_seconds": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "error": error,
        })


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


def ask_model(prompt_item):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    use_case = prompt_item["use_case"]
    system_prompt = SYSTEM_PROMPTS[use_case]

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt_item["prompt"],
            },
        ],
        "max_tokens": prompt_item.get("max_tokens", 700),
        "temperature": 0.2,
    }

    start = time.time()

    try:
        response = requests.post(
            ENDPOINT,
            headers=headers,
            json=payload,
            timeout=120,
        )
    except requests.exceptions.RequestException as e:
        latency = round(time.time() - start, 3)
        return None, latency, {}, str(e)

    latency = round(time.time() - start, 3)

    if response.status_code != 200:
        return None, latency, {}, f"{response.status_code}: {response.text}"

    try:
        data = response.json()
    except Exception as e:
        return None, latency, {}, f"JSON parse error: {e}"

    answer = clean_answer(extract_answer(data))
    usage = extract_usage(data)

    if not answer:
        return "", latency, usage, "Empty answer"

    return answer, latency, usage, ""


def run_benchmark():
    validate_prompt_counts()
    benchmark_prompts = build_benchmark_prompts()

    print(f"Running benchmark for model: {MODEL}")
    print(f"Total prompts: {len(benchmark_prompts)}")
    print(f"Log file: {LOG_FILE}")
    print("-" * 60)

    for idx, prompt_item in enumerate(benchmark_prompts, start=1):
        print(
            f"[{idx}/{len(benchmark_prompts)}] "
            f"{prompt_item['prompt_id']} | "
            f"{prompt_item['use_case']} | "
            f"{prompt_item['difficulty']}"
        )

        answer, latency, usage, error = ask_model(prompt_item)

        if error:
            print(f"ERROR: {error}")
            save_log(
                prompt_item=prompt_item,
                answer=answer or "",
                latency=latency,
                usage=usage,
                error=error,
            )
            print()
            continue

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens

        print(
            f"Latency: {latency}s | "
            f"Tokens: In {input_tokens} / Out {output_tokens} / Total {total_tokens}"
        )
        print(f"Answer preview: {answer[:150].replace(chr(10), ' ')}")
        print()

        save_log(
            prompt_item=prompt_item,
            answer=answer,
            latency=latency,
            usage=usage,
        )

    print("Benchmark completed.")


if __name__ == "__main__":
    run_benchmark()
