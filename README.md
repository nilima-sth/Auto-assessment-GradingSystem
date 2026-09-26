**AI-Assisted Answer Sheet Grading System**

 **Overview**

This project is a production-oriented FastAPI and Streamlit application for automatically grading scanned student answer sheets. It refactors the original notebook prototype into a backend API, adds Gemini structured grading, OCR, semantic answer routing, genuine RAG with ChromaDB, reliability features, and a simple teacher-facing web UI.

 **Features**
- Upload exam/model-answer files and student answer-sheet PDFs.
- Extract handwritten answers with TrOCR.
- Map OCR answer chunks to questions with all-MiniLM-L6-v2 sentence embeddings.
- Ingest optional rubric/reference documents into ChromaDB.
- Retrieve question-aware RAG context during grading.
- Grade answers with Gemini using prompt engineering, configured temperature / top_p, function calling, and Pydantic-validated structured output.
- Retry external LLM calls and fall back from Gemini to a vLLM local model endpoint when appropriate.
- Rate limit expensive API endpoints.
- Process batch grading through asynchronous API handlers with bounded OCR and LLM concurrency.
- Persist grading records in SQLite.
- Provide a Streamlit frontend for teacher workflows.
- Run locally or through Docker Compose.

## Architecture

The system architecture and Week 16 agentic loop are documented here:

[View Architecture Diagram](docs/architecture.md)
 **Current AI Pipeline**

 exam/model answer/reference docs

  → text extraction

  → RAG chunking

  → all-MiniLM-L6-v2 embeddings

  → persistent ChromaDB

  → student PDF OCR with TrOCR

  → semantic answer-to-question mapping

  → question-aware RAG retrieval

  → Gemini structured grading with function calling

  → retry and Gemini-to-vLLM fallback

  → GradeEvaluation validation

  → SQLite persistence



 **Difference Between Semantic Mapping and RAG**

 Semantic mapping is used to assign OCR text from a student answer sheet to likely exam questions. It does not add outside knowledge.

 RAG is used separately for grading support. The official exam/model answers and optional rubric/reference documents are chunked, embedded, stored in ChromaDB, retrieved by reference_set_id, and passed to the LLM as supporting context.
 **Technology Stack**
- FastAPI, Pydantic, Uvicorn
- Streamlit
- Gemini through the Google Gen AI SDK
- vLLM through its OpenAI-compatible HTTP API
- TrOCR through Transformers/PyTorch
- SentenceTransformers and scikit-learn
- ChromaDB persistent vector storage
- SQLite
- Docker and Docker Compose
 **Project Structure**

  backend/app/api/          FastAPI routes

  backend/app/core/         config and rate limiting

  backend/app/database/     SQLite helpers

  backend/app/llm/          Gemini, vLLM, Ollama, retry, fallback, tools

  backend/app/rag/          ingestion, vector store, retrieval

  backend/app/services/     OCR, parsing, embeddings, grading workflow

  frontend/app.py           Streamlit teacher UI

  docs/architecture.md      architecture diagram

  docs/week16-compliance-audit.md  Week 16 requirement scorecard

  tests/                    pytest suite


 **Environment Variables**

 Copy the example file and fill local values:

 cp backend/.env.example backend/.env



 Set Gemini key in backend/.env:

 GEMINI_API_KEY=


 Important variables:

  LLM_PROVIDER=gemini

  GEMINI_MODEL=gemini-3.6-flash

  LLM_TEMPERATURE=0.2

  LLM_TOP_P=0.9

  LLM_RETRY_ATTEMPTS=3

  ENABLE_VLLM_FALLBACK=true

  VLLM_BASE_URL=http://localhost:8001/v1

  VLLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0

  RATE_LIMIT_REQUESTS=10

  RATE_LIMIT_WINDOW_SECONDS=60

  OCR_CONCURRENCY=1

  LLM_CONCURRENCY=3

  CHROMA_PATH=../data/chroma


 **Local Setup**

 python3 -m venv .venv

  source .venv/bin/activate

  pip install -r backend/requirements.txt


 Install Poppler for PDF conversion. On Ubuntu/Debian:

 sudo apt-get install poppler-utils

 If Poppler is not on PATH, set POPPLER_PATH in backend/.env.

**Gemini API Key Setup**
1. Copy backend/.env.example to backend/.env.
2. Set `GEMINI_API_KEY` to the local key value.
3. Keep LLM_PROVIDER=gemini.
4. Use the configured model in GEMINI_MODEL.

 The real key must stay local and must not appear in README, tests, Docker Compose, or source code.
 **Running Backend**

 .venv/bin/uvicorn backend.app.main:app --reload

  Health check: curl http://127.0.0.1:8000/api/health


**Running Streamlit Frontend**

 BACKEND_URL=http://127.0.0.1:8000 .venv/bin/streamlit run frontend/app.py

 Open the Streamlit URL, upload the exam/model-answer file, optionally ingest rubric/reference files, then upload student answer-sheet PDFs and start grading.
 **Running vLLM**

 The backend expects vLLM through its OpenAI-compatible endpoint.

 python -m vllm.entrypoints.openai.api_server \

    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \

    --host 0.0.0.0 \

    --port 8001


 Full local vLLM inference requires suitable GPU/VRAM and model download time. This repository implements the integration and Docker Compose service; do not claim successful vLLM runtime inference unless that service has actually generated a response in your environment.

**Running Tests**

 .venv/bin/python -m compileall backend frontend tests

  .venv/bin/python -m pytest -q

  **Docker**

 docker compose up --build


 Compose starts:
- backend on port 8000
- frontend on port 8501
- vllm on host port 8001
- persistent app and Hugging Face cache volumes

 Docker Compose uses shell variables or a root .env file for substitution. It does not read backend/.env, because docker compose config expands env_file values and can expose secrets in terminal output.

**API Endpoints**

 GET  /api/health

  POST /api/rag/ingest

  POST /api/grading/evaluate

 POST /api/rag/ingest accepts an exam/model-answer file and optional reference files, then returns a reference_set_id.

 POST /api/grading/evaluate accepts an exam/model-answer file, one or more student PDFs, and optional reference_set_id.

**Reliability**
- Retry: external LLM calls are retried with bounded exponential backoff.
- Fallback: Gemini is primary; after retry failure the provider falls back to vLLM for retryable external errors.
- Rate limiting: expensive endpoints return HTTP 429 after the configured request limit.
- Graceful errors: invalid uploads return 400, dependency failures return 503 where applicable, and unexpected failures return safe 500 responses without exposing secrets.
 **Performance**
- Async API handlers keep expensive requests from blocking the event loop.
- OCR and semantic mapping are offloaded to worker threads with bounded concurrency.
- LLM grading is also bounded to avoid unbounded simultaneous provider calls.
- TrOCR and SentenceTransformer models are cached/reused by service loaders.
- ChromaDB retrieval limits context with RAG_TOP_K.

- **ONNX Decision**

 ONNX was considered but is not necessary for this implementation:
- Hosted Gemini cannot be converted to ONNX by this application.
- Local generative LLM inference is optimized through vLLM, including PagedAttention and request batching.
- TrOCR remains a Transformers/PyTorch model in this project.
- SentenceTransformer inference is already reused and bounded.

 Therefore ONNX conversion is documented as not applicable for the final assignment implementation.


**Deployment Instructions**

 For local Docker deployment:

 cp backend/.env.example backend/.env

  # fill backend/.env for direct backend runs

  # for Compose, also export GEMINI_API_KEY or create an ignored root .env

  docker compose up --build



 For a VPS/server deployment:
1. Install Docker and Docker Compose.
2. Copy the project to the server.
3. Create backend/.env from backend/.env.example.
4. Set GEMINI_API_KEY, model names, paths, and limits.
5. Ensure the server has GPU/VRAM if vLLM will be used for real local inference.
6. Run docker compose up --build -d.
7. Put a reverse proxy such as Nginx/Caddy in front of ports 8000 and 8501 if exposing publicly.

 Cloud deployment can use the same container layout, but no cloud-specific deployment is required for this submission.

**Limitations**
- TrOCR model weights must be downloaded/cached before first full OCR runtime verification.
- vLLM runtime verification requires sufficient GPU/VRAM and model download time.
- In-process rate limiting is suitable for a single backend process; a distributed deployment should use shared storage such as Redis.
- OCR quality depends on scan clarity and handwriting quality.
- Existing local files such as grades.db and data/student_pdfs/s1.pdf may contain non-synthetic historical data and are ignored by Git.
# Week 16 Work

## Context Engineering Technique

The planner is given a small context pack on each pass: the question, reference answer, mapped student answer, any flagged OCR text, up to four retrieved chunks, the current grade or verification result, and a short list of recent actions. The full trajectory stays in `AgentState`, so the whole history is not sent back to the model every time.

## Agentic Pattern

I used one agent for each question. It chooses one of five actions: `retrieve_context`, `grade_answer`, `verify_grade`, `request_clarification`, or `finish`. Python runs the selected action, updates the question state, and asks the planner what to do next. The loop ends when the agent finishes, asks for review, hits a tool error, or reaches `AGENT_MAX_ITERATIONS` (default 5). There is no forced retrieve-then-grade sequence.

The Week 15 parts that already worked, such as uploads, parsing, OCR, answer mapping, and database access, are still used. Week 16 changes only the per-question decision step that used to retrieve once and grade once.

## Evaluation Harness

I kept the evaluation code small and local in `backend/app/evaluation/`. It runs six scripted cases through `run_question_agent` and records the final outcome, selected actions, path length, failures, and token information. The results are in `docs/evaluation-results.md`. These cases do not call an external model, so their LLM token count is zero.

## Skill vs Agent

In this project, a skill is one of the fixed capabilities: OCR, answer mapping, retrieval, mark validation, or database persistence. The agent is the part that decides which capability to use next after seeing the current result. It cannot write to SQLite or call arbitrary Python functions.

## Token and Cost Accounting

`AgentState.token_usage` keeps input, output, and total token counts for a question. Gemini usage metadata and the OpenAI-compatible `usage` object from vLLM are read when they are returned, including grading retries and fallback calls. Each trajectory entry keeps the usage for that action. Ollama does not expose matching counts here, so those values are left unavailable. I have not added a dollar cost because there was no live provider run or price data to support one.

## Failure Injection Test

For `retrieval-timeout`, the harness makes the retrieval function raise a `TimeoutError`. The agent records the failed call and returns `manual_review`; it does not invent evidence, call grading, or insert a final grade row. I classified this as a soft failure because the problem was contained.

## Tool vs Agent Boundary

The model only selects from the small action list and supplies checked arguments. Python runs the tools and handles errors, limits, and terminal states. The application, not the model, writes grades to SQLite. A question sent for manual review keeps that status and does not get an artificial final grade row.
