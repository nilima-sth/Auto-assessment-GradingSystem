# Week 16 Architecture

The system uses one grading agent per question. The agent starts after OCR and semantic answer mapping and ends before deterministic SQLite persistence.

```mermaid
flowchart TD
    U[Exam and student uploads] --> P[Exam parsing]
    P --> O[OCR student PDF]
    O --> M[Semantic answer mapping]
    M --> A[Question AgentState]
    A --> C[Compact context builder]
    C --> L[LLM planner]
    L --> D{Validated action}
    D -->|retrieve_context| R[RAG retrieval tool]
    D -->|grade_answer| G[Existing grading provider]
    D -->|verify_grade| V[Independent verification provider]
    D -->|request_clarification| H[Manual review]
    D -->|finish| F[Finalized question result]
    R --> A
    G --> A
    V --> A
    A -->|iteration < AGENT_MAX_ITERATIONS| C
    A -->|tool failure or limit| H
    F --> S[Application-controlled SQLite persistence]
    H --> N[No final grade row; explicit manual_review response]
```

The model selects only schema-validated actions. Python owns tool execution, iteration limits, validation, and persistence. Full trajectories remain in `AgentState`; the planner receives bounded evidence and concise action summaries.

The architecture is single-agent rather than multi-agent because each question has one tightly scoped decision process. Separate agents would add coordination and token overhead without a separate ownership boundary in this application.

