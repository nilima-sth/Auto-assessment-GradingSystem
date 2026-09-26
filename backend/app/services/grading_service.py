import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from backend.app.core.config import get_settings
from backend.app.agent.loop import run_question_agent
from backend.app.agent.schemas import AgentState
from backend.app.database.sqlite import insert_answer, insert_grade, insert_student
from backend.app.llm.ollama_provider import OllamaProvider
from backend.app.llm.provider_factory import get_llm_provider
from backend.app.schemas.grading import GradeEvaluation
from backend.app.rag.retrieval import retrieve_rag_context
from backend.app.services.document_service import load_exam
from backend.app.services.embedding_service import map_answers_to_questions
from backend.app.services.ocr_service import ocr_pdf_to_text


def grade_with_mistral(
    question_text: str,
    model_answer: str,
    student_answer: str,
    max_marks: int,
) -> dict:
    """Backward-compatible helper for local Ollama/Mistral grading."""
    evaluation = OllamaProvider().grade_answer(
        question_number="Q",
        question_text=question_text,
        model_answer=model_answer,
        student_answer=student_answer,
        max_marks=max_marks,
        retrieved_context=None,
    )
    return {"Marks": evaluation.awarded_marks, "Feedback": evaluation.feedback}


def grade_answer(
    question_number: str,
    question_text: str,
    model_answer: str,
    student_answer: str,
    max_marks: float,
    retrieved_context: str | None = None,
) -> GradeEvaluation:
    """Grade one answer through the configured LLM provider."""
    return get_llm_provider().grade_answer(
        question_number=question_number,
        question_text=question_text,
        model_answer=model_answer,
        student_answer=student_answer,
        max_marks=max_marks,
        retrieved_context=retrieved_context,
    )


def grade_question_with_agent(
    *,
    question_number: str,
    question_text: str,
    model_answer: str,
    student_answer: str,
    max_marks: float,
    flagged_ocr_text: str | None,
    reference_set_id: str | None,
) -> AgentState:
    """Grade one question through the Week 16 decision loop."""
    provider = get_llm_provider()
    planner = (
        provider
        if hasattr(provider, "plan_agent_action")
        else _LegacyProviderPlanner(provider, use_retrieval=reference_set_id is not None)
    )
    return run_question_agent(
        question_number=question_number,
        question_text=question_text,
        max_marks=max_marks,
        model_answer=model_answer,
        mapped_student_answer=student_answer,
        flagged_ocr_text=flagged_ocr_text,
        reference_set_id=reference_set_id,
        planner=planner,
        grade_fn=provider.grade_answer,
        retrieve_fn=retrieve_rag_context,
        max_iterations=get_settings().agent_max_iterations,
    )


def _grade_from_agent_state(state: AgentState) -> GradeEvaluation:
    if state.status == "completed" and state.current_grade is not None:
        return state.current_grade
    return GradeEvaluation(
        question_number=state.question_number,
        awarded_marks=0.0,
        max_marks=state.max_marks,
        feedback=f"Manual review required: {state.clarification_reason or 'agent did not produce a safe grade.'}",
    )


class _LegacyProviderPlanner:
    """Compatibility adapter for Week 15 provider test doubles."""

    def __init__(self, provider, *, use_retrieval: bool) -> None:
        self.provider = provider
        self.use_retrieval = use_retrieval
        self._graded = False
        self._retrieved = False

    def plan_agent_action(self, _context: str):
        from backend.app.agent.schemas import AgentDecision

        if self.use_retrieval and not self._retrieved:
            decision = AgentDecision(action="retrieve_context")
            self._retrieved = True
        else:
            decision = AgentDecision(action="grade_answer" if not self._graded else "finish")
            self._graded = True
        return decision, None

    def verify_grade(self, _context: str):
        raise RuntimeError("Legacy provider does not support verification.")


def _result_from_agent_state(state: AgentState) -> dict:
    evaluation = _grade_from_agent_state(state)
    return {
        "awarded_marks": evaluation.awarded_marks,
        "max_marks": evaluation.max_marks,
        "feedback": evaluation.feedback,
        "retrieved_chunks": state.retrieved_chunks,
        "status": state.status,
        "manual_review_reason": state.clarification_reason,
        "trajectory": [record.model_dump() for record in state.trajectory],
        "token_usage": state.token_usage.model_dump(),
    }


def grade_batch(
    exam_path: str,
    student_pdf_paths: list[str],
    reference_set_id: str | None = None,
) -> dict:
    """Run the original sequential grading pipeline for one or more PDFs."""
    settings = get_settings()
    questions, model_answers = load_exam(exam_path)
    results = {}

    for pdf_path in student_pdf_paths:
        raw = ocr_pdf_to_text(pdf_path)
        mapped = map_answers_to_questions(
            questions,
            raw,
            chunk_size=settings.answer_chunk_size,
            overlap=settings.answer_chunk_overlap,
            threshold=settings.similarity_threshold,
        )

        student_name = os.path.basename(pdf_path)
        student_id = insert_student(student_name, student_name)

        student_result = {}
        for question_id, question_meta in questions.items():
            student_answer = mapped.get(question_id, "").strip()
            answer_id = insert_answer(student_id, question_id, raw, student_answer)
            agent_state = grade_question_with_agent(
                question_number=question_id,
                question_text=question_meta["text"],
                model_answer=model_answers[question_id],
                student_answer=student_answer,
                max_marks=question_meta["marks"],
                flagged_ocr_text=mapped.get("Flagged"),
                reference_set_id=reference_set_id,
            )
            evaluation = _grade_from_agent_state(agent_state)
            if agent_state.status == "completed":
                insert_grade(answer_id, evaluation.awarded_marks, evaluation.feedback)
            student_result[question_id] = _result_from_agent_state(agent_state)

        if "Flagged" in mapped and mapped["Flagged"].strip():
            student_result["Flagged"] = mapped["Flagged"].strip()

        results[student_name] = student_result
    return results


async def _run_blocking(executor: ThreadPoolExecutor, func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(func, *args, **kwargs))


async def _ocr_and_map_pdf(
    executor: ThreadPoolExecutor,
    pdf_path: str,
    questions: dict[str, dict],
) -> tuple[str, str, dict[str, str]]:
    settings = get_settings()
    raw = await _run_blocking(executor, ocr_pdf_to_text, pdf_path)
    mapped = await _run_blocking(
        executor,
        map_answers_to_questions,
        questions,
        raw,
        chunk_size=settings.answer_chunk_size,
        overlap=settings.answer_chunk_overlap,
        threshold=settings.similarity_threshold,
    )
    return pdf_path, raw, mapped


async def _bounded_ocr_and_map(
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    pdf_path: str,
    questions: dict[str, dict],
) -> tuple[str, str, dict[str, str]]:
    async with semaphore:
        return await _ocr_and_map_pdf(executor, pdf_path, questions)


async def _bounded_grade_answer(
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    question_number: str,
    question_text: str,
    model_answer: str,
    student_answer: str,
    max_marks: float,
    retrieved_context: str | None,
    ) -> GradeEvaluation:
    async with semaphore:
        return await _run_blocking(
            executor,
            grade_answer,
            question_number,
            question_text,
            model_answer,
            student_answer,
            max_marks,
            retrieved_context=retrieved_context,
        )


async def _bounded_grade_question_with_agent(
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    question_number: str,
    question_text: str,
    model_answer: str,
    student_answer: str,
    max_marks: float,
    flagged_ocr_text: str | None,
    reference_set_id: str | None,
) -> AgentState:
    async with semaphore:
        return await _run_blocking(
            executor,
            grade_question_with_agent,
            question_number=question_number,
            question_text=question_text,
            model_answer=model_answer,
            student_answer=student_answer,
            max_marks=max_marks,
            flagged_ocr_text=flagged_ocr_text,
            reference_set_id=reference_set_id,
        )


async def grade_batch_async(
    exam_path: str,
    student_pdf_paths: list[str],
    reference_set_id: str | None = None,
) -> dict:
    """Grade PDFs with bounded concurrent OCR/mapping and LLM grading."""
    # Keep the Week 15 helper available for older callers and tests.
    _ = _bounded_grade_answer
    settings = get_settings()
    questions, model_answers = load_exam(exam_path)
    ocr_semaphore = asyncio.Semaphore(max(1, settings.ocr_concurrency))
    llm_semaphore = asyncio.Semaphore(max(1, settings.llm_concurrency))

    with ThreadPoolExecutor(max_workers=max(1, settings.ocr_concurrency)) as ocr_executor:
        mapped_students = await asyncio.gather(
            *[
                _bounded_ocr_and_map(ocr_semaphore, ocr_executor, pdf_path, questions)
                for pdf_path in student_pdf_paths
            ]
        )

    results = {}
    with ThreadPoolExecutor(max_workers=max(1, settings.llm_concurrency)) as llm_executor:
        for pdf_path, raw, mapped in mapped_students:
            student_name = os.path.basename(pdf_path)
            student_id = insert_student(student_name, student_name)
            student_result = {}
            grading_jobs = []

            for question_id, question_meta in questions.items():
                student_answer = mapped.get(question_id, "").strip()
                answer_id = insert_answer(student_id, question_id, raw, student_answer)
                grading_jobs.append(
                    (
                        question_id,
                        answer_id,
                        _bounded_grade_question_with_agent(
                            llm_semaphore,
                            llm_executor,
                            question_id,
                            question_meta["text"],
                            model_answers[question_id],
                            student_answer,
                            question_meta["marks"],
                            mapped.get("Flagged"),
                            reference_set_id,
                        ),
                    )
                )

            agent_states = await asyncio.gather(*(job[-1] for job in grading_jobs))
            for (question_id, answer_id, _task), agent_state in zip(grading_jobs, agent_states):
                evaluation = _grade_from_agent_state(agent_state)
                if agent_state.status == "completed":
                    insert_grade(answer_id, evaluation.awarded_marks, evaluation.feedback)
                student_result[question_id] = _result_from_agent_state(agent_state)

            if "Flagged" in mapped and mapped["Flagged"].strip():
                student_result["Flagged"] = mapped["Flagged"].strip()

            results[student_name] = student_result
    return results
