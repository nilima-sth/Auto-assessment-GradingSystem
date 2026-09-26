# Week 16 Architecture

There is one grading agent for each question. It starts after OCR and answer mapping, and it hands the result back to the application before anything is written to SQLite.

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

The model chooses from the small set of allowed actions. Python runs the tools, checks the arguments, enforces the iteration limit, and handles persistence. The complete path is kept in `AgentState`; the planner only sees the shortened context needed for its next decision.

I chose a single-agent design because each question has one short decision process. Splitting that work across several agents would add coordination and extra model calls without giving any part of the application a separate responsibility.
