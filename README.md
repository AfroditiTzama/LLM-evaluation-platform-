# LLM Evaluation Platform

A hosted LLM evaluation platform for comparing multiple language models on custom prompts or CSV prompt sets.

The platform runs the same prompt across all configured models and automatically records latency, token usage, estimated cost, JSON validity, quality scores, hallucination safety scores and final model recommendations.

The application is deployed on Render and uses PostgreSQL for persistent storage.

## Live App

Hosted app:

https://llm-evaluation-platform-f8ds.onrender.com

The application is protected with an app password through the APP_PASSWORD environment variable.

## Main Features

- Single prompt evaluation from the web UI
- CSV prompt upload
- Multi-model evaluation
- Latency measurement
- Token usage tracking
- Estimated cost calculation
- JSON validity checking
- LLM-as-a-judge quality evaluation
- Hallucination safety evaluation
- Final model ranking and recommendation
- Persistent PostgreSQL storage
- Database-backed dashboard
- Password protection

## Evaluated Models

The platform currently evaluates:

- minimax/minimax-m2.5
- minimax/minimax-m2.7
- minimax/minimax-m3
- qwen/qwen3.6-plus
- qwen/qwen3.7-max
- qwen/qwen3.7-plus

The judge model is:

- qwen/qwen3.7-max

## Project Structure

- weekly_runs_dashboard_db.py: database-backed Streamlit dashboard
- weekly_runs_dashboard.py: folder-based Streamlit dashboard
- run_weekly_pipeline.py: full evaluation pipeline
- run_custom_evaluation.py: runs all models on custom prompts
- custom_quality_judge.py: quality and hallucination judge
- sync_run_to_db.py: syncs run outputs to the database
- dashboard_v2.py: static benchmark dashboard
- requirements.txt: Python dependencies

## Environment Variables

The application requires:

- API_KEY: OpenRouter API key
- DATABASE_URL: PostgreSQL connection string
- APP_PASSWORD: password for accessing the hosted app

Do not commit API keys, database URLs, passwords, .env files or local databases to GitHub.

## Local Setup

Create a virtual environment:

    python3 -m venv venv
    source venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

Run locally:

    python3 -m streamlit run weekly_runs_dashboard_db.py

## CSV Input Format

Required column:

- question

Optional columns:

- prompt_id
- use_case
- difficulty
- expected_format
- reference_text
- max_tokens

Example:

    prompt_id,use_case,difficulty,expected_format,question,reference_text
    p001,structured_json,easy,json,"Return a valid JSON object with name, category and price.",""
    p002,customer_support,easy,text,"Answer whether cheesecake is available.","Menu: Espresso, Brownie, Apple pie"

## Metrics

The platform records:

- latency
- input tokens
- output tokens
- total tokens
- estimated cost
- JSON validity
- correctness
- completeness
- instruction following
- hallucination safety
- format quality
- overall quality

## Recommendation Logic

The final recommendation combines:

- quality
- hallucination safety
- latency
- cost
- reliability
- JSON validity

The best model is selected based on the best overall trade-off, not only raw quality.

## Database Storage

The app stores results in PostgreSQL.

Main tables:

- runs
- eval_prompts
- model_outputs
- model_summary
- model_usecase_summary
- preliminary_recommendations
- judge_scores
- judge_summary_model
- final_recommendations

## Deployment

The app is deployed on Render.

Build command:

    pip install -r requirements.txt

Start command:

    python -m streamlit run weekly_runs_dashboard_db.py --server.address 0.0.0.0 --server.port $PORT --server.headless true

## Security Notes

The hosted app is protected with APP_PASSWORD.

The following should not be committed:

- .env
- llm_eval.db
- runs/
- uploaded_prompts/
- __pycache__/
- .DS_Store

## Limitations

- Quality judging can contain judge bias.
- Cost estimates depend on provider pricing.
- Hallucination safety depends on available reference text.
- Free Render services may cold start after inactivity.
- Free Render PostgreSQL instances may expire unless upgraded.

## Current Status

This version is an operational v1 of the LLM Evaluation Platform.
