# Week 17 Phase 2: Live Prompt/Configuration Experiments

## Scope

This phase runs real local Ollama planner and grading calls through the existing
Week 16 agent loop. It does not add Evidently, Airflow, a golden dataset, a judge,
or a regression gate. The original deterministic V1 baseline remains separate as
MLflow run `ef47fd8a2ad34fabb3fb952eb2aa9efc`.

## Local provider setup

Selected model: `qwen2.5:7b` (Ollama ID `845dbda0ea48`, 4.7 GB local size).
It is a 7.62B Q4 model with a 32K context window. Ollama documents its structured
JSON-output capability, which fits the existing JSON `AgentDecision`,
`GradeVerification`, and `GradeEvaluation` schemas. Its 4.7 GB footprint fit the
available 8 GB RTX 5060 GPU while leaving working headroom; larger 14B models do
not. [Ollama model details](https://ollama.com/library/qwen2.5%3A7b)

`ollama --version` reported 0.23.2. The service was available at the standard local
endpoint, and the model passed a standalone JSON-mode call plus a project
`OllamaProvider.grade_answer` call.

### Provider compatibility fix

The existing CLI provider requested JSON only in natural-language prompts. V1 setup
revealed that CLI word wrapping injected ANSI cursor bytes into captured stdout,
causing JSON parsing to fail and return an indistinguishable zero-mark fallback.
The provider now invokes:

```text
ollama run <model> --format json --nowordwrap --hidethinking
```

This is a narrow structured-output compatibility fix. Invalid JSON now raises a
provider error, which the existing retry/agent loop records in its trajectory. It
does not change Gemini or vLLM behavior. Ollama's existing CLI path does not expose
reliable token counts to this provider, so token metrics remain unavailable.

Live experiments use:

```bash
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5:7b \
  uv run python -m backend.app.mlops.experiments --config v1 --execution-mode ollama-live
```

The harness's planner and grader calls are live. Its retrieval tool remains the
existing deterministic harness retrieval; no claim is made that these runs tested a
real Chroma retrieval corpus.

## V1 live baseline

MLflow run: `2a00595ef67642cb93e63ad01a59f0ea`.

Observed V1 weaknesses were trace-based:

- `rag-needed` and `retrieval-timeout` selected `verify_grade` before a grade,
  causing the recorded error `Cannot verify before a grade exists.`
- `rewrite-retrieval` repeatedly selected `verify_grade` after verification had
  already failed and reached the five-iteration safety limit.
- `direct-grade` graded an exact model-answer match as 2/5, then requested manual
  review after repeated failed verification.

No hidden reasoning was inferred from these events; the evidence is the selected
actions, structured arguments, summaries, errors, and final grades in the trace.

## V2: planner state guards

MLflow run: `b665af9936f446c98186c944e980dfc4`.

**V1 problem → evidence → V2 change:** V1 chose invalid pre-grade verification and
repeated verification. `planner_v2.txt` adds only action-state guards: no verification
or finish without a grade, no repeated verification once a verification record exists,
and explicit valid next actions. The grading system prompt is unchanged from V1.

**Result:** V2 avoided the V1 pre-grade `verify_grade` action but did not improve the
overall loop. It repeatedly re-graded after a failed verification, reached the maximum
iteration count in three cases, and produced two malformed `GradeVerification`
objects where `suggested_next_step` was a list rather than the required string.
It also continued to award 2–3/5 for an exact model-answer match.

## V3: exact-match grading calibration

MLflow run: `f188ccc5a3394912abf5bb9bd0c69447`.

**V2 problem → evidence → V3 change:** V2 traces repeatedly awarded 2–3/5 although
the student answer exactly matched the model answer. V3 keeps the V2 planner prompt
unchanged and adds one grading-system sentence: material matches without omissions
relative to the official model answer receive full marks; extra detail not present in
the official answer is not required.

**Result:** V3 awarded 5/5 in every non-ambiguous case, including the exact-match
`direct-grade` trace. It produced no recorded provider/tool errors, reduced average
iterations to 1.83, and reduced run latency to 13.73 seconds. It did not increase the
scripted-harness task-completion rate beyond V1's 0.333.

## Side-by-side comparison

The machine-readable form is [week17-phase2-comparison.json](week17-phase2-comparison.json).

| Live config | Task completion | Tool correctness | Avg. iterations | Latency | Errors | Hard failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 | 0.333 | 0.941 | 2.83 | 40.85 s | 2 | 4 |
| V2 | 0.167 | 1.000 | 3.50 | 73.65 s | 2 | 4 |
| V3 | 0.333 | 1.000 | 1.83 | 13.73 s | 0 | 0 |

The result supports V3 as the operationally preferable configuration in this limited
test: it matches V1 completion while eliminating recorded runtime errors, cutting
iterations and latency, and correcting the observed exact-match grade. It does not
establish general grading accuracy.

A central limitation is visible in the existing deterministic cases: `direct-grade`,
`rag-needed`, `rewrite-retrieval`, `verification-replan`, and `retrieval-timeout`
provide the live planner essentially the same question, model answer, mapped answer,
and reference-set state while expecting different tool paths. A live model cannot
reliably infer those invisible case labels. V3's valid grade-and-finish behavior is
therefore counted incomplete for retrieval-required scripted cases. This is not
corrected in Phase 2 because a representative labelled/golden dataset is explicitly
out of scope.

## Representative traces

Every run has complete safe traces in its `agent_traces.json` MLflow artifact and
2–3 selected cases in `representative_traces.json`:

- V1: run `2a00595ef67642cb93e63ad01a59f0ea` — recorded tool error, max-iteration
  path, and manual-review trace.
- V2: run `b665af9936f446c98186c944e980dfc4` — malformed verification schema,
  max-iteration path, and manual-review trace.
- V3: run `f188ccc5a3394912abf5bb9bd0c69447` — successful exact-match grade,
  manual-review case, and additional distinct case.

Artifacts are safe structured JSON; they contain application-level actions,
arguments, result summaries, errors, usage fields when available, and final result.
They do not contain model chain-of-thought.

## Trade-offs and remaining blockers

- A 7B local model enables repeatable private local execution, but responses are
  stochastic and runs were GPU-warm at different points; latency comparisons are
  indicative, not benchmark-grade measurements.
- The local provider has no token usage accounting through its CLI.
- Gemini remains blocked by its missing API key; vLLM remains unavailable. Neither
  behavior was changed.
- The six existing cases are agent-loop safety tests, not a real grading-quality
  dataset. Their indistinguishable live input states limit action-path metrics.

## Ready for Phase 3

The project now has three observed, versioned live configurations; comparable MLflow
runs; safe trajectories; failure rationale; automatic representative-trace artifacts;
and a working local Ollama path. It is ready for the next explicitly authorized
evaluation/observability phase, while Evidently, golden data, judge evaluation,
promotion gates, and Airflow remain unimplemented.
