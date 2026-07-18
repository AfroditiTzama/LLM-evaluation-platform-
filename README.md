# Greek LLM Evaluation Platform — SQLite Edition

A hosted Streamlit platform for comparing **Qwen 3.6 27B** and **Gemma 4 31B Instruct** on a research-inspired Greek benchmark.

The repository contains:

- 120 original Greek prompts across 10 categories
- 240 model responses from the completed benchmark run
- 120 blind pairwise judgments from an independent judge
- corrected deterministic evaluation metrics
- a stratified 30-prompt blind human-review sample
- a preloaded SQLite seed database
- a password-protected Streamlit dashboard

## Important benchmark corrections

Version 1.1 corrects the issues identified after the first run:

- JSON validity is calculated only for tasks that explicitly require JSON output.
- JSON, YAML, CSV and Markdown tables have separate syntax and schema checks.
- Structured-format compliance also checks instructions such as “only JSON” and the absence of Markdown fences.
- Exact-answer tasks use explicit strict, normalized, contains or numeric comparison.
- `MR-H03` now uses the correct reference value of approximately `20.7142857%`.
- Raw `cannot_assess` judge labels are preserved, while cases in which both answers have the same task-success status are reported as effective ties.
- The human sample is stratified: one easy, one medium and one hard prompt from each category, with balanced A/B assignment.

No model or judge API requests were repeated to produce the corrected reports.

## Project structure

```text
.
├── app.py                       # SQLite-backed Streamlit dashboard
├── main.py                      # Qwen/Gemma benchmark and judge pipeline
├── evaluation.py               # corrected deterministic checks
├── reporting.py                # summaries, CIs and human sample
├── database.py                 # SQLite schema and import functions
├── rebuild_existing_run.py     # rebuild existing raw run without API calls
├── benchmark_prompts.json      # benchmark v1.1
├── seed/
│   └── llm_eval_seed.db        # preloaded database for first deployment
├── render.yaml                 # Render paid service + persistent disk
├── requirements.txt
└── .env.example
```

## Local dashboard

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python -m streamlit run app.py
```

The first launch copies `seed/llm_eval_seed.db` to `data/llm_eval.db`.

## Running a future benchmark locally

Add your OpenRouter API key to `.env`, then run:

```bash
python main.py
```

New runs are written to `results/` and synchronized to the local SQLite database automatically.

## Importing a future local run into the hosted database

The hosted dashboard includes an **Admin import** tab. Upload:

- `run_<id>_metadata.json`
- `run_<id>_results.json`
- `run_<id>_pairwise_judgments.json`

The app recalculates the corrected metrics and imports the run without making any API calls.

## Render deployment with SQLite persistence

SQLite is stored in a file. On Render, persistent writes require a **paid web service with a persistent disk**. The included `render.yaml` uses:

```text
DATABASE_PATH=/var/data/llm_eval.db
Disk mount path=/var/data
Disk size=1 GB
```

Deploy through a Render Blueprint or create the service manually with:

```text
Build command:
pip install -r requirements.txt

Start command:
python -m streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true
```

Set `APP_PASSWORD` in the Render environment settings.

### Free Render limitation

A free Render web service does not support persistent disks. The bundled seed database can still display the existing benchmark, but human reviews and imported runs written at runtime can disappear after a restart or redeploy. Use the paid disk configuration when database persistence is required.

## Interpretation limits

- The benchmark contains one response per prompt and model, so it does not estimate generation variance.
- Provider routing affects latency and cost.
- An LLM judge can have preference and verbosity bias.
- The results apply to this prompt set, configuration and provider routes; they do not establish a universally best model.
