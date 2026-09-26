# Week 17 regression evaluation

## Fixed dataset and checks

The fixed six-case dataset is
`backend/app/mlops/golden/regression_v1.json`. It covers exact matches, partial
credit, incorrect answers, retrieval-supported grading, ambiguous OCR/manual
review, and a post-grade loop guard. Cases have approved expected status, marks,
actions, and reference responses; they are not generated from V3 outputs.

Evidently 0.7.23 evaluates each candidate with local Ollama `qwen2.5:7b` using
two checks: `reference_correctness` and `workflow_adherence`. A separate project
contract check validates expected status, required actions, call limit, grade
presence, and awarded marks. Judge/contract disagreements are retained for human
review; promotion is not automatic.

## Recorded results

| Version | MLflow run | Passed | `pct_tests_passed` |
| --- | --- | ---: | ---: |
| V1 | `a9eba64e6b884ec8952056509f6bb4b3` | 1/6 | 16.67% |
| V2 | `4ad434109ce94b3c9dc396501f023b50` | 1/6 | 16.67% |
| V3 | `438a8c2460434881903ffe2241926a34` | 2/6 | 33.33% |

V3 passes the exact-match and loop-guard cases but still fails partial-credit,
incorrect-answer, retrieval, and ambiguous-OCR behavior. It is an improvement,
not a release gate.

Run all configurations locally after pulling the model:

```bash
ollama pull qwen2.5:7b
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:7b \
  uv run python -m backend.app.mlops.regression --config all
```

Rendered evidence is in `reports/evidently/v1_regression.html`,
`v2_regression.html`, and `v3_regression.html`.
