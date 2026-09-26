# Week 17 Phase 0: Research, Architecture Audit, and Runtime Setup

## Scope and result

This audit establishes a reproducible local environment for the existing Week 16
application. It deliberately does **not** add MLflow, Evidently, prompt experiments,
regression evaluation, or any other Week 17 feature.

The canonical Python environment is now the root `pyproject.toml` and `uv.lock`.
It targets CPython 3.12 and preserves the direct dependencies from the existing
backend and frontend requirements files. The requirements files remain in place
for the existing Docker images.

`backend/.env` was created from the example with blank credentials. It is ignored.
The repository's example configuration was corrected so it contains no API key.

## Actual execution flow

```text
Streamlit frontend
  -> POST /api/rag/ingest or POST /api/grading/evaluate
  -> FastAPI upload validation and temporary files
  -> exam parsing (Q/A pairs)
  -> student-PDF rasterization with Poppler and TrOCR
  -> SentenceTransformer embedding and semantic answer-to-question mapping
  -> one agentic grading loop per question
       -> optional Chroma RAG retrieval
       -> configured Gemini, vLLM, or Ollama provider
       -> optional provider tool call, grade, and verification
  -> persist completed grades to SQLite
  -> structured API response rendered by Streamlit
```

RAG ingestion is a separate path. It parses the exam and optional reference files,
chunks them, embeds the chunks with the same SentenceTransformer model, and writes
them to a per-reference-set persistent Chroma collection. During grading,
`retrieve_rag_context` embeds the question/student-answer query and first tries a
question-number filter, then a broader fallback query.

Key implementation locations:

- Entry point and routes: `backend/app/main.py`, `backend/app/api/`
- OCR, parsing, semantic mapping, and batch workflow:
  `backend/app/services/`
- Chroma ingestion, storage, and retrieval: `backend/app/rag/`
- provider selection and resilience: `backend/app/llm/`
- per-question decision loop: `backend/app/agent/`
- SQLite persistence: `backend/app/database/sqlite.py`
- deterministic evaluation harness: `backend/app/evaluation/`

## Week 16 agent telemetry

`run_question_agent` creates an `AgentState` for each question. The state is the
authoritative in-memory trace while grading:

| Requirement | Existing recording |
| --- | --- |
| Iterations | `current_iteration`; every `ToolCallRecord` has its iteration number. |
| Decisions/actions | validated `AgentDecision.action`: retrieve, grade, verify, clarification, or finish. |
| Tool arguments | `ToolCallRecord.arguments` is the decision serialized with non-null fields. |
| Results | `result_summary`, retrieved chunks, current grade, and verification result. |
| Errors | failed planner/tool records have `success=False` and an `error`; terminal reason is `clarification_reason`. |
| Termination | `status` is `completed` or `manual_review`; a max-iteration or unsafe action becomes manual review. |
| Token usage | per-record and aggregate `TokenUsage` records prompt, completion, total, availability, and provider source when supplied. |
| Trajectory | ordered `state.trajectory` is returned in the grading API response for every question. |

Important boundary: SQLite currently persists only students, answers, and completed
grades. It does **not** persist `AgentState`, trajectories, individual tool calls,
or token usage. A Week 17 trace store must build on the existing `AgentState` /
`ToolCallRecord` schema rather than create competing telemetry, while adding durable
storage if assignment artifacts need to survive the API response.

Provider usage is available from Gemini and vLLM response metadata when those
providers return it. Ollama currently reports no usage. The deterministic harness
also intentionally has no live-provider token counts.

## Reproducible local setup

From a clean clone with uv installed:

```bash
uv sync
uv run python -m compileall backend frontend tests
uv run pytest -q
uv run uvicorn backend.app.main:app --reload
BACKEND_URL=http://127.0.0.1:8000 uv run streamlit run frontend/app.py
```

The expected writable runtime locations are repository-relative:

- SQLite: `grades.db` (from `DATABASE_PATH=../grades.db`, resolved relative to `backend/`)
- Chroma: `data/chroma`
- temporary uploads: OS-managed temporary directories only

None of those data locations, virtual environments, caches, or secrets are tracked.
A small missing test fixture, `data/exam_files/qna.txt`, was restored so the existing
document-parser test can execute on a clean clone.

## Runtime observed during this audit

| Check | Observed state |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS, Linux 7.0.0-31-generic, x86_64 |
| Python | CPython 3.12.3 at `/usr/bin/python3` |
| uv | Installed for this setup: 0.12.19 (`~/.local/bin/uv`) |
| Docker / Compose | Docker 29.8.0; Docker Compose 5.5.1; daemon responding |
| Poppler | `pdftoppm` 24.02.0 on PATH |
| Ollama | 0.23.2 installed; no local models listed |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB; driver 580.173.02 |
| CUDA | No `nvcc`; uv-installed PyTorch 2.14.0+cu130 reports CUDA 13.0 and `cuda.is_available() == True` |
| Ports | 8000, 8001, and 8501 were free before verification |
| Git | This supplied directory has no `.git` metadata, so branch and change status cannot be determined here |

External requirements that were not fabricated:

- `GEMINI_API_KEY` is blank; Gemini live grading is blocked until a valid key is
  supplied locally.
- No Ollama model is installed; the configured `mistral` model must be pulled before
  an Ollama grading run.
- No vLLM process answered on port 8001. Its configured TinyLlama model must be
  downloaded and vLLM must be started, with sufficient compatible GPU memory.
- First OCR/embedding/RAG use needs Hugging Face downloads for
  `microsoft/trocr-base-handwritten` and `all-MiniLM-L6-v2`. Network access and
  local model-cache capacity are therefore required for a first real document run.

## Verification results

| Check | Result |
| --- | --- |
| `uv sync` | PASS — resolved and installed 153 packages. |
| `uv run python -m compileall backend frontend tests` | PASS. |
| `uv run pytest -q` | PASS — 31 passed; one pre-existing PyPDF2 deprecation warning. |
| `uv run uvicorn backend.app.main:app ...` + `/api/health` | PASS — server returned `{"status":"ok"}`. |
| Streamlit command | READY — launch command is documented above; a browser session was not needed for this backend-focused audit. |
| Live Gemini grading | BLOCKED BY EXTERNAL REQUIREMENT — no API key. |
| Live Ollama grading | BLOCKED BY EXTERNAL REQUIREMENT — no model installed. |
| Live vLLM grading | BLOCKED BY EXTERNAL REQUIREMENT — endpoint/model absent. |

## Docker audit

The existing Dockerfiles are internally consistent with their legacy
`requirements.txt` files and install Poppler in the backend image. They do not yet
consume `uv.lock`, so Docker remains a separately resolved compatibility path;
local `uv sync` is the canonical reproducible Week 17 environment.

`docker compose config -q` passes. Compose always defines backend, frontend, and
vLLM. The backend's `depends_on: vllm` means a normal `docker compose up` also
starts the heavyweight vLLM container even when Gemini is the intended provider.
The vLLM image uses the mutable `latest` tag and downloads the selected model into
the named Hugging Face volume; this is not a quick, fully pinned CPU-only development
path. The compose file also has no explicit GPU reservation/device declaration,
so host NVIDIA-container-runtime configuration still determines whether vLLM can use
the observed GPU.

Docker Compose uses environment substitution and does not read `backend/.env` for
substitution. Keep real Compose secrets in an ignored root `.env` or exported shell
variables, never in source. The app-data and Hugging Face cache volumes are the
correct persistence points. No Docker redesign was made in this phase; validate and
pin a vLLM image/model together before changing that deployment path.

## Week 17 Track B gaps (research only)

1. uv requirement: completed locally with `pyproject.toml` and `uv.lock`; Docker
   still needs a deliberate future decision on consuming the lockfile.
2. MLflow experiment tracking: absent. Existing `AgentState`, `ToolCallRecord`,
   `TokenUsage`, provider/model settings, and `GradeEvaluation` are the natural
   sources for runs, params, metrics, and artifacts.
3. Prompt/configuration versioning: absent. Prompts are inline in `llm/base.py`,
   `gemini_provider.py`, `vllm_provider.py`, and `ollama_provider.py`; settings are
   environment-derived but have no version identifier or snapshot per run.
4. Full durable agent trace: partially present in API response only. Persisted trace
   artifacts and a safe redaction policy are absent.
5. Three or more prompt/config experiments: absent. The current deterministic
   harness tests agent behavior, not provider/prompt quality.
6. Representative live success/failure traces: absent. The evaluation cases are
   scripted simulations and correctly report zero provider tokens.
7. Regression/golden dataset: absent. The six deterministic cases are useful loop
   safety fixtures, but they are not a representative labelled grading dataset.

The next phase should add these capabilities without altering the existing Week 16
agent loop semantics or duplicating its telemetry schema.
