from __future__ import annotations

from backend.app.agent.schemas import AgentState


def build_agent_context(state: AgentState, *, max_chunks: int = 4, max_chars: int = 12000) -> str:
    """Prepare the short context sent to the planner."""
    chunks = state.retrieved_chunks[:max_chunks]
    evidence = "\n".join(
        f"[{item.get('source', 'reference')}#{item.get('chunk_index', 0)}] "
        f"{str(item.get('text', ''))[:2400]}"
        for item in chunks
    ) or "No retrieved evidence."
    summaries = "\n".join(state.previous_action_summaries[-6:]) or "No previous actions."
    context = f"""Question: {state.question_text}
Maximum marks: {state.max_marks}
Model answer: {state.model_answer}
Mapped student answer: {state.mapped_student_answer or '[empty]'}
Flagged or unmapped OCR text: {state.flagged_ocr_text or '[none]'}
Retrieved evidence:
{evidence}
Current grade: {state.current_grade.model_dump_json() if state.current_grade else '[none]'}
Verification: {state.verification.model_dump_json() if state.verification else '[none]'}
Previous action summaries:
{summaries}
Iteration: {state.current_iteration}"""
    return context[:max_chars]
