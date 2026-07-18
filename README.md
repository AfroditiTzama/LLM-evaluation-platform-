# LLM Evaluation Platform

A hosted evaluation platform for comparing multiple large language models using individual prompts or CSV-based test sets.

The platform executes the same prompts across all configured models and records performance, cost, reliability, formatting, and response-quality metrics. It also applies an LLM-as-a-judge evaluation workflow and produces a final model ranking based on the overall trade-off between quality, safety, latency, cost, and reliability.

The application is deployed on Render and uses PostgreSQL for persistent storage.

## Live Application

**Hosted application:**

https://llm-evaluation-platform-f8ds.onrender.com

Access to the application is password-protected through the `APP_PASSWORD` environment variable.

> Free Render services may require additional startup time after periods of inactivity.

## Main Features

* Single-prompt evaluation through the web interface
* CSV prompt-set upload
* Multi-model evaluation through OpenRouter
* Consistent prompt execution across all configured models
* Latency measurement
* Input, output, and total token tracking
* Estimated API cost calculation
* JSON validity checking
* LLM-as-a-judge quality evaluation
* Hallucination safety evaluation
* Per-use-case model comparison
* Final model ranking and recommendation
* Persistent PostgreSQL storage
* Database-backed Streamlit dashboard
* Password-protected hosted application

## Evaluation Workflow

The platform follows the workflow below:

1. A user enters a single prompt or uploads a CSV prompt set.
2. The same prompt is submitted to every configured model.
3. Raw responses and execution metadata are recorded.
4. Deterministic metrics are calculated, including:

   * latency,
   * token usage,
   * estimated cost,
   * JSON validity,
   * retry information,
   * request reliability.
5. The judge model evaluates response quality and hallucination safety.
6. Results are aggregated by model and use case.
7. The platform produces a final ranking and model recommendation.
8. All evaluation outputs are stored in PostgreSQL and displayed in the dashboard.

## Evaluated Models

The current platform configuration evaluates the following models:

* `minimax/minimax-m2.5`
* `minimax/minimax-m2.7`
* `minimax/minimax-m3`
* `qwen/qwen3.6-plus`
* `qwen/qwen3.7-max`
* `qwen/qwen3.7-plus`

### Judge Model

The current LLM-as-a-judge model is:

```text
qwen/qwen3.7-max
```

The evaluated models and judge model can be changed through the project configuration.

## Evaluation Metrics

The platform records technical, economic, and quality-related metrics.

### Performance Metrics

* Latency
* Input tokens
* Output tokens
* Total tokens
* Retry count
* Execution reliability

### Cost Metrics

* Estimated cost per request
* Aggregated model cost
* Estimated cost by use case
* Cost-performance trade-off

### Deterministic Validation

* JSON validity
* Expected-format compliance
* Response availability
* Execution success or failure

### LLM-as-a-Judge Metrics

* Correctness
* Completeness
* Instruction following
* Hallucination safety
* Format quality
* Overall quality

## Recommendation Logic

The final recommendation is based on the overall trade-off between:

* response quality,
* hallucination safety,
* latency,
* estimated cost,
* reliability,
* JSON validity.

The selected model is therefore not necessarily the model with the highest raw quality score. The recommendation aims to identify the model that provides the most suitable balance for the evaluated workload.

Recommendations are generated both:

* globally across the complete evaluation set,
* and separately for individual use cases.

## Project Structure

```text
llm-evaluation-platform/
├── weekly_runs_dashboard_db.py
├── weekly_runs_dashboard.py
├── run_weekly_pipeline.py
├── run_custom_evaluation.py
├── custom_quality_judge.py
├── sync_run_to_db.py
├── dashboard_v2.py
├── requirements.txt
├── README.md
├── .gitignore
└── additional configuration and utility files
```

### Main Files

#### `weekly_runs_dashboard_db.py`

Database-backed Streamlit dashboard used by the deployed application.

It reads evaluation results from PostgreSQL and provides interfaces for:

* running custom evaluations,
* uploading prompt CSV files,
* inspecting model outputs,
* comparing model-level summaries,
* viewing judge scores,
* reviewing final recommendations.

#### `weekly_runs_dashboard.py`

Folder-based Streamlit dashboard for evaluation outputs stored locally instead of in PostgreSQL.

#### `run_weekly_pipeline.py`

Runs the complete evaluation pipeline, including model execution, result processing, judging, summarization, and recommendation generation.

#### `run_custom_evaluation.py`

Executes all configured models against custom prompts or uploaded prompt sets.

#### `custom_quality_judge.py`

Evaluates model responses using the configured judge model.

It generates scores for:

* correctness,
* completeness,
* instruction following,
* hallucination safety,
* format quality,
* overall quality.

#### `sync_run_to_db.py`

Synchronizes evaluation outputs and summary files with the PostgreSQL database.

#### `dashboard_v2.py`

Static benchmark dashboard used for inspecting previously generated benchmark results.

## Database Storage

Evaluation data is stored in PostgreSQL.

The primary database tables include:

```text
runs
eval_prompts
model_outputs
model_summary
model_usecase_summary
preliminary_recommendations
judge_scores
judge_summary_model
final_recommendations
```

### Table Overview

* `runs`: metadata for each evaluation run
* `eval_prompts`: prompts included in each run
* `model_outputs`: raw model responses and execution metadata
* `model_summary`: aggregated technical metrics by model
* `model_usecase_summary`: aggregated metrics by model and use case
* `preliminary_recommendations`: recommendations before judge results
* `judge_scores`: detailed judge scores for individual responses
* `judge_summary_model`: aggregated judge results by model
* `final_recommendations`: final ranked model recommendations

## Environment Variables

The application requires the following environment variables:

```text
API_KEY
DATABASE_URL
APP_PASSWORD
```

### `API_KEY`

OpenRouter API key used to call the configured language models.

### `DATABASE_URL`

PostgreSQL connection string used by the hosted application.

### `APP_PASSWORD`

Password required to access the Streamlit application.

Create a local `.env` file when running the project locally:

```env
API_KEY=your_openrouter_api_key
DATABASE_URL=your_postgresql_connection_string
APP_PASSWORD=your_application_password
```

Never commit the `.env` file or its contents to GitHub.

## Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AfroditiTzama/llm-evaluation-platform.git
cd llm-evaluation-platform
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate it on macOS or Linux:

```bash
source venv/bin/activate
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file containing:

```env
API_KEY=your_openrouter_api_key
DATABASE_URL=your_postgresql_connection_string
APP_PASSWORD=your_application_password
```

### 5. Run the Application

```bash
python3 -m streamlit run weekly_runs_dashboard_db.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## CSV Input Format

Uploaded CSV files must contain a `question` column.

### Required Column

```text
question
```

### Optional Columns

```text
prompt_id
use_case
difficulty
expected_format
reference_text
max_tokens
```

### Example

```csv
prompt_id,use_case,difficulty,expected_format,question,reference_text
p001,structured_json,easy,json,"Return a valid JSON object with name, category and price.",""
p002,customer_support,easy,text,"Answer whether cheesecake is available.","Menu: Espresso, Brownie, Apple pie"
```

### Column Descriptions

| Column            | Description                                                       |
| ----------------- | ----------------------------------------------------------------- |
| `prompt_id`       | Unique identifier for the prompt                                  |
| `use_case`        | Evaluation category or task type                                  |
| `difficulty`      | Prompt difficulty level                                           |
| `expected_format` | Expected response type, such as text or JSON                      |
| `question`        | Prompt submitted to each model                                    |
| `reference_text`  | Supporting evidence used for quality and hallucination evaluation |
| `max_tokens`      | Optional maximum output-token limit                               |

When `reference_text` is available, it is used to improve correctness and hallucination-safety evaluation.

## Render Deployment

The application is deployed as a Render Web Service.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
python -m streamlit run weekly_runs_dashboard_db.py \
  --server.address 0.0.0.0 \
  --server.port $PORT \
  --server.headless true
```

### Required Render Environment Variables

Configure the following variables in the Render dashboard:

```text
API_KEY
DATABASE_URL
APP_PASSWORD
```

Sensitive values must be stored only in the Render environment-variable configuration and must never be committed to the repository.

## Security

The hosted application is protected using the `APP_PASSWORD` environment variable.

The following files and directories must not be committed:

```text
.env
.streamlit/secrets.toml
llm_eval.db
runs/
uploaded_prompts/
venv/
.venv/
__pycache__/
*.pyc
.DS_Store
```

API keys, database credentials, passwords, local databases, and generated evaluation results should remain outside version control unless the data has been reviewed and intentionally anonymized.

## Limitations

* LLM-as-a-judge evaluation may contain model bias.
* A judge model may prefer responses that resemble its own style.
* Quality scores should ideally be validated with human evaluation.
* Cost estimates depend on current provider pricing and may change.
* Latency is influenced by both the model and the provider infrastructure.
* Hallucination-safety evaluation is more reliable when reference text is available.
* A high judge score does not guarantee factual correctness.
* Free Render services may cold-start after inactivity.
* Free PostgreSQL instances may expire or become unavailable unless upgraded.
* Benchmark conclusions apply primarily to the selected prompts, models, providers, and evaluation configuration.

## Responsible Interpretation

The platform is intended to support comparative model evaluation, not to establish a universally best language model.

Results should be interpreted in relation to:

* the selected prompt set,
* the evaluated use cases,
* prompt difficulty,
* expected output format,
* provider infrastructure,
* model configuration,
* judge-model limitations,
* and the availability of reference answers.

Human review remains important for validating model quality and final deployment decisions.

## Current Status

This repository contains an operational **v1** of the LLM Evaluation Platform.

The current version supports:

* custom prompt evaluation,
* CSV-based evaluation,
* multi-model execution,
* automatic metric collection,
* judge-based quality scoring,
* PostgreSQL persistence,
* model ranking,
* hosted dashboard access.

Future improvements may include stronger human-evaluation workflows, configurable metric weights, additional providers, automated regression testing, and expanded visual analytics.
