# AI-Assisted Answer Sheet Grading System

This FastAPI and Streamlit project grades scanned answer sheets using OCR, semantic answer mapping, optional Chroma-backed reference retrieval, and structured LLM grading. Week 17 adds a measured agent-evaluation workflow around the existing Week 16 agent loop: versioned prompts, MLflow experiments, fixed golden regression cases, Evidently reports, and local Ollama judging.

## Project progression

| Stage | Delivered capability |
| --- | --- |
| Week 15 | OCR, semantic answer mapping, structured grading, SQLite persistence, FastAPI, and Streamlit. |
| Week 16 | Question-level agent loop with retrieve, grade, verify, clarify, and finish actions; retries, fallback integration, and safe trajectory state. |
| Week 17 | `uv`-locked environment; V1/V2/V3 prompt/config experiments in MLflow; fixed six-case regression evaluation with Evidently and local Qwen judge. |

The [architecture document](docs/architecture.md) describes the application flow. Semantic mapping assigns OCR segments to questions; RAG separately retrieves official/reference evidence to support grading.

## Architecture

```text
Exam/model answer + optional references ──> extraction ──> Chroma RAG corpus
Student PDF ──> TrOCR ──> semantic answer-to-question mapping
Question + mapped answer + retrieved evidence ──> Week 16 agent loop
                                             └──> structured GradeEvaluation ──> SQLite

Week 17: versioned agent configuration ──> MLflow runs/traces
         fixed golden cases + candidate output ──> Evidently LLM judge + HTML reports
```

The agent trace records explicit actions, arguments, result summaries, tool errors, iterations, terminal status/reason, and final grades. It does not record inferred hidden model reasoning.

## Environment and clean-clone setup

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Poppler for PDF conversion, and (for live Week 17 commands) a running local Ollama installation.

```bash
# Install uv if it is not already installed; see the uv link above.
uv sync

# Ubuntu/Debian only, if Poppler is not on PATH.
sudo apt-get install poppler-utils
```

`pyproject.toml` and `uv.lock` are the canonical environment. `uv sync` creates the project environment; no requirements-file install is needed for local work. A clean temporary source copy was verified with `uv sync --frozen` and `uv run pytest -q` (44 passed).

Create local configuration without committing it:

```bash
cp backend/.env.example backend/.env
```

Important variables in this project:

```dotenv
LLM_PROVIDER=gemini             # gemini, vllm, or ollama
GEMINI_API_KEY=                 # required only for Gemini execution
GEMINI_MODEL=gemini-3.6-flash
OLLAMA_MODEL=qwen2.5:7b         # supplied at command time for Week 17 runs
VLLM_BASE_URL=http://localhost:8001/v1
VLLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
CHROMA_PATH=../data/chroma
DATABASE_PATH=../grades.db
```

`backend/.env` is ignored. Do not place API keys in source files, README, Docker Compose files, or reports.

## Run the application

Backend and health check:

```bash
uv run uvicorn backend.app.main:app --reload
curl http://127.0.0.1:8000/api/health
```

Frontend, in another terminal:

```bash
BACKEND_URL=http://127.0.0.1:8000 uv run streamlit run frontend/app.py
```

The main API routes are `GET /api/health`, `POST /api/rag/ingest`, and `POST /api/grading/evaluate`. Docker Compose remains available for the original application stack with `docker compose up --build`; its Dockerfiles retain their existing requirements-file setup, while local Week 17 work uses uv.

## Local Ollama for Week 17

Pull and confirm the tested local model:

```bash
ollama pull qwen2.5:7b
ollama list
```

The recorded experiments used `qwen2.5:7b` through the existing Ollama CLI provider. It returns structured JSON for planner, verification, and grading calls. The local provider does not expose reliable token counts, so those metrics remain unavailable. Live reproduction requires Ollama to be running and the model to be available locally; it was verified for the recorded runs but is not rerun by setup.

## Test and local observability commands

```bash
uv lock --check
uv run python -m compileall backend frontend tests
uv run pytest -q

# Local MLflow metadata: ignored SQLite database; artifacts: ignored mlruns/.
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The current suite has 44 tests. The MLflow UI and `/api/health` were both started and checked locally during final audit.

## Versioned MLflow experiments

Configurations live in `backend/app/mlops/configs/`; prompts are in `backend/app/mlops/prompts/`. V1 preserves the Week 16 system prompt, V2 changes only planner state guards, and V3 retains V2 planning while adding exact-match grading calibration.

Run deterministic V1 harness tracking:

```bash
uv run python -m backend.app.mlops.experiments --config v1
```

Run a live local configuration (repeat with `v2` or `v3`):

```bash
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:7b \
  uv run python -m backend.app.mlops.experiments --config v3 --execution-mode ollama-live
```

The experiment runner logs parameters that the app actually consumes, aggregate metrics, full safe traces, and selected representative traces. MLflow run IDs and the full methodology are in [Phase 2 notes](docs/week17-phase2-experiments.md).

| Version | Completion | Tool correctness | Avg. iterations | Latency | Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 | 0.333 | 0.941 | 2.83 | 40.85 s | 2 |
| V2 | 0.167 | 1.000 | 3.50 | 73.65 s | 2 |
| V3 | 0.333 | 1.000 | 1.83 | 13.73 s | 0 |

Actual experiment story: V1 exposed invalid verification order, repeated verification, maximum-iteration exits, and poor exact-match calibration. V2 added planner-state guards, improving tool correctness but worsening completion and latency—a genuine regression. V3 kept those guards and added targeted exact-match calibration; it eliminated recorded provider/tool errors and reduced iterations and latency. It is not a claim of general grading quality.

## Fixed regression evaluation

The six approved, project-owned cases are in [`backend/app/mlops/golden/regression_v1.json`](backend/app/mlops/golden/regression_v1.json). They cover exact, partial, incorrect, retrieval, ambiguous-OCR, and loop-guard behavior; they are not generated V3 outputs.

```bash
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:7b \
  uv run python -m backend.app.mlops.regression --config all
```

Each version gets a separate MLflow run. Evidently 0.7.23 uses local Qwen through Ollama for two checks: `reference_correctness` and `workflow_adherence`. A separate project-contract sanity check compares expected status, actions, call limit, and marks. Any judge/contract disagreement is retained for human review; promotion is interpretation-only.

| Version | Passed | pct_tests_passed | Regression MLflow run |
| --- | ---: | ---: | --- |
| V1 | 1/6 | 16.67% | `a9eba64e6b884ec8952056509f6bb4b3` |
| V2 | 1/6 | 16.67% | `4ad434109ce94b3c9dc396501f023b50` |
| V3 | 2/6 | 33.33% | `438a8c2460434881903ffe2241926a34` |

These fresh runs also log `pct_tests_passed` with the same value as the preserved `regression_pass_percentage` metric. V3 improved the fixed set but still fails partial-credit, incorrect-answer, retrieval, and ambiguous-OCR behavior. Review the recorded candidate/reference, verdicts, explanations, and sanity results before any release decision. See [Phase 3 regression notes](docs/week17-phase3-regression.md).

## Submission evidence and repository layout

The committed evidence should include the lockfile, prompt/config files, golden dataset, tests, documentation, machine-readable Phase 2 comparison, and generated Evidently reports. Runtime databases, MLflow artifact directories, Chroma data, model caches, `.venv`, and `backend/.env` are intentionally ignored.

```text
backend/app/
  agent/                 # Week 16 loop and state schemas
  evaluation/            # deterministic harness
  mlops/
    configs/             # v1, v2, v3
    golden/              # fixed regression_v1.json
    experiments.py       # MLflow experiment runner
    regression.py        # Evidently/golden regression runner
docs/
  week17-phase0-audit.md
  week17-phase1-baseline.md
  week17-phase2-experiments.md
  week17-phase2-comparison.json
  week17-phase3-regression.md
  week17-final-audit.md
reports/evidently/
  v1_regression.html
  v2_regression.html
  v3_regression.html
tests/
pyproject.toml
uv.lock
README.md
```

## Known limitations and optional work

- V3 has only a 2/6 fixed regression pass rate; it is not fully successful.
- The Phase 2 live harness uses deterministic retrieval and has several indistinguishable inputs that expect different action paths; those metrics are useful operational signals, not a general quality benchmark.
- OCR quality depends on scan quality and handwriting; TrOCR weights download on first use. vLLM integration needs compatible hardware and model availability.
- Rate limiting is in-process and is suitable for a single backend process.
- **Airflow is optional bonus work and is not implemented.**

For the requirement-by-requirement final audit, hygiene review, and manual submission instructions, see [docs/week17-final-audit.md](docs/week17-final-audit.md).
