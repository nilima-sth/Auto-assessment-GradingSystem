import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from backend.app.core.config import get_settings
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
            rag_context_text = None
            retrieved_chunks = []
            if reference_set_id:
                rag_context = retrieve_rag_context(
                    reference_set_id=reference_set_id,
                    question_number=question_id,
                    question_text=question_meta["text"],
                    student_answer=student_answer,
                )
                retrieved_chunks = [chunk.model_dump() for chunk in rag_context.chunks]
                rag_context_text = rag_context.text or None
            evaluation = grade_answer(
                question_id,
                question_meta["text"],
                model_answers[question_id],
                student_answer,
                question_meta["marks"],
                retrieved_context=rag_context_text,
            )
            insert_grade(answer_id, evaluation.awarded_marks, evaluation.feedback)
            student_result[question_id] = {
                "awarded_marks": evaluation.awarded_marks,
                "max_marks": evaluation.max_marks,
                "feedback": evaluation.feedback,
                "retrieved_chunks": retrieved_chunks,
            }

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


async def grade_batch_async(
    exam_path: str,
    student_pdf_paths: list[str],
    reference_set_id: str | None = None,
) -> dict:
    """Grade PDFs with bounded concurrent OCR/mapping and LLM grading."""
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
                rag_context_text = None
                retrieved_chunks = []
                if reference_set_id:
                    rag_context = retrieve_rag_context(
                        reference_set_id=reference_set_id,
                        question_number=question_id,
                        question_text=question_meta["text"],
                        student_answer=student_answer,
                    )
                    retrieved_chunks = [chunk.model_dump() for chunk in rag_context.chunks]
                    rag_context_text = rag_context.text or None

                grading_jobs.append(
                    (
                        question_id,
                        answer_id,
                        retrieved_chunks,
                        _bounded_grade_answer(
                            llm_semaphore,
                            llm_executor,
                            question_id,
                            question_meta["text"],
                            model_answers[question_id],
                            student_answer,
                            question_meta["marks"],
                            rag_context_text,
                        ),
                    )
                )

            evaluations = await asyncio.gather(*(job[-1] for job in grading_jobs))
            for (question_id, answer_id, retrieved_chunks, _task), evaluation in zip(grading_jobs, evaluations):
                insert_grade(answer_id, evaluation.awarded_marks, evaluation.feedback)
                student_result[question_id] = {
                    "awarded_marks": evaluation.awarded_marks,
                    "max_marks": evaluation.max_marks,
                    "feedback": evaluation.feedback,
                    "retrieved_chunks": retrieved_chunks,
                }

            if "Flagged" in mapped and mapped["Flagged"].strip():
                student_result["Flagged"] = mapped["Flagged"].strip()

            results[student_name] = student_result
    return results
