# Benchmark v1.1 Corrections

The original 240 model responses and 120 judge requests were preserved. No API calls were repeated.

## Corrected items

1. **Output-format detection**
   - JSON validation now runs only on prompts that explicitly require JSON.
   - Python tasks that mention JSON and text tasks that discuss JSON validity are no longer mislabeled as JSON-output tasks.

2. **Structured outputs**
   - Separate checks for JSON, YAML, CSV and Markdown tables.
   - Separate syntax validity, schema validity and strict format compliance.
   - Markdown fences are treated as noncompliant when the prompt asks for “only JSON” or “only CSV”.

3. **Exact answers**
   - Strict, normalized, contains and numeric comparison are stored separately.
   - Numeric tasks support decimal commas, percent signs and tolerances.

4. **Mathematical reference**
   - `MR-H03` corrected from `20%` to `20.7142857%`.

5. **Judge outcomes**
   - The raw winner label is preserved.
   - `cannot_assess` becomes an effective tie only when both answers have equal recorded task-success status.

6. **Human sample**
   - Rebuilt as a 30-prompt stratified sample with equal category/difficulty coverage and balanced answer order.

## Corrected headline metrics

| Metric | Qwen 3.6 27B | Gemma 4 31B Instruct |
|---|---:|---:|
| API success | 100% | 100% |
| Mean latency | 9.468 s | 4.760 s |
| P95 latency | 42.058 s | 14.839 s |
| Total model cost | $0.080700 | $0.010275 |
| Deterministic pass rate | 74.07% | 85.19% |
| Structured syntax validity | 100% | 100% |
| Structured schema validity | 92.86% | 78.57% |
| Strict structured-format compliance | 78.57% | 28.57% |
| Judge overall quality | 4.5917/5 | 4.5750/5 |
| Judge task success | 91.67% | 95.00% |
| Judge hallucination rate | 3.33% | 0.83% |
| Decisive judge wins | 36/60 | 24/60 |

Latency and cost are specific to the providers selected by OpenRouter during this run.
