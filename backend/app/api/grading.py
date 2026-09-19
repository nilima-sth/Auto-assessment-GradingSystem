import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from backend.app.core.rate_limit import expensive_endpoint_rate_limit
from backend.app.database.sqlite import init_db
from backend.app.schemas.grading import GradingResponse, QuestionGrade, StudentGradeResult
from backend.app.services.grading_service import grade_batch_async

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/grading", tags=["grading"])


def _validate_uploads(exam_file: UploadFile, student_files: list[UploadFile]) -> None:
    exam_suffix = Path(exam_file.filename or "").suffix.lower()
    if exam_suffix not in {".txt", ".pdf"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported exam file type; use .txt or .pdf.",
        )
    if not student_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one student PDF is required.",
        )
    for student_file in student_files:
        if Path(student_file.filename or "").suffix.lower() != ".pdf":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported student file type for {student_file.filename}; use .pdf.",
            )


def _save_upload(upload: UploadFile, destination: Path) -> None:
    with destination.open("wb") as output_file:
        shutil.copyfileobj(upload.file, output_file)


def _to_response(results: dict) -> GradingResponse:
    students = []
    for student_name, student_result in results.items():
        question_grades = []
        flagged_text = student_result.get("Flagged")
        for question_id, grade in student_result.items():
            if question_id == "Flagged":
                continue
            question_grades.append(
                QuestionGrade(
                    question_number=question_id,
                    awarded_marks=float(grade.get("awarded_marks", 0)),
                    max_marks=float(grade.get("max_marks", 0)),
                    feedback=str(grade.get("feedback", "")),
                    retrieved_chunks=list(grade.get("retrieved_chunks", [])),
                )
            )
        students.append(
            StudentGradeResult(
                student=student_name,
                questions=question_grades,
                total_awarded=sum(item.awarded_marks for item in question_grades),
                total_possible=sum(item.max_marks for item in question_grades),
                flagged_text=flagged_text,
            )
        )
    return GradingResponse(students=students)


@router.post("/evaluate", response_model=GradingResponse)
async def evaluate(
    exam_file: UploadFile = File(...),
    student_files: list[UploadFile] = File(...),
    reference_set_id: str | None = Form(default=None),
    _rate_limit: None = Depends(expensive_endpoint_rate_limit),
) -> GradingResponse:
    _validate_uploads(exam_file, student_files)
    try:
        init_db()
    except Exception as exc:
        logger.exception("Database initialization failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database initialization failed.",
        ) from exc

    with tempfile.TemporaryDirectory(prefix="auto_grading_") as temp_dir:
        temp_path = Path(temp_dir)
        exam_path = temp_path / f"exam{Path(exam_file.filename or '').suffix.lower()}"
        _save_upload(exam_file, exam_path)

        student_paths = []
        for index, student_file in enumerate(student_files, start=1):
            filename = Path(student_file.filename or f"student_{index}.pdf").name
            destination = temp_path / filename
            _save_upload(student_file, destination)
            student_paths.append(str(destination))

        try:
            results = await grade_batch_async(str(exam_path), student_paths, reference_set_id=reference_set_id)
        except ValueError as exc:
            logger.exception("Invalid grading input")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("External grading dependency failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc) or "External grading dependency failed.",
            ) from exc
        except Exception as exc:
            logger.exception("Grading failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Grading failed. Check server logs for details.",
            ) from exc

    return _to_response(results)
