import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.app.core.rate_limit import expensive_endpoint_rate_limit
from backend.app.rag.ingestion import ingest_reference_set
from backend.app.rag.schemas import IngestionResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag"])


def _save_upload(upload: UploadFile, destination: Path) -> None:
    with destination.open("wb") as output_file:
        shutil.copyfileobj(upload.file, output_file)


def _validate_exam(exam_file: UploadFile) -> None:
    if Path(exam_file.filename or "").suffix.lower() not in {".txt", ".pdf"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported exam file type; use .txt or .pdf.",
        )


def _validate_reference_files(reference_files: list[UploadFile]) -> None:
    for reference_file in reference_files:
        if Path(reference_file.filename or "").suffix.lower() not in {".txt", ".pdf"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported reference file type for {reference_file.filename}; use .txt or .pdf.",
            )


@router.post("/ingest", response_model=IngestionResult)
async def ingest(
    exam_file: UploadFile = File(...),
    reference_files: list[UploadFile] = File(default=[]),
    _rate_limit: None = Depends(expensive_endpoint_rate_limit),
) -> IngestionResult:
    _validate_exam(exam_file)
    _validate_reference_files(reference_files)

    with tempfile.TemporaryDirectory(prefix="rag_ingest_") as temp_dir:
        temp_path = Path(temp_dir)
        exam_path = temp_path / f"exam{Path(exam_file.filename or '').suffix.lower()}"
        _save_upload(exam_file, exam_path)

        reference_paths = []
        for index, reference_file in enumerate(reference_files, start=1):
            filename = Path(reference_file.filename or f"reference_{index}.txt").name
            destination = temp_path / filename
            _save_upload(reference_file, destination)
            reference_paths.append(destination)

        try:
            return ingest_reference_set(exam_path, reference_paths)
        except ValueError as exc:
            logger.exception("RAG ingestion input failed")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("RAG ingestion failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="RAG ingestion failed. Check server logs for details.",
            ) from exc
