import os
import re
from pathlib import Path
from typing import BinaryIO

from PyPDF2 import PdfReader

QA_REGEX = r"(Q\d+):\s*(.+?)\s*A\d+:\s*(.+?)(?=\nQ\d+:|$)"
MARKS_IN_Q = r"\((\d+)\)"
TRAILING_MARKS = r"\s*\((\d+)\)\s*$"


def read_pdf_text(file_path: str | os.PathLike[str] | BinaryIO) -> str:
    """Extract plain text from a PDF file."""
    reader = PdfReader(str(file_path)) if isinstance(file_path, (str, os.PathLike)) else PdfReader(file_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text.append(page_text)
    return "\n".join(text)


def read_text_file(file_path: str | os.PathLike[str] | BinaryIO) -> str:
    """Read a UTF-8 text exam/reference file."""
    if isinstance(file_path, (str, os.PathLike)):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    return file_path.read().decode("utf-8")


def parse_exam(text: str, default_marks: int = 5) -> tuple[dict[str, dict], dict[str, str]]:
    """Parse Q/A pairs from the existing notebook exam format."""
    matches = re.findall(QA_REGEX, text, flags=re.DOTALL)
    questions = {}
    model_answers = {}
    for q_id, q_text, a_text in matches:
        mark_match = re.search(MARKS_IN_Q, q_text)
        clean_answer = a_text.strip()
        if mark_match:
            marks = int(mark_match.group(1))
        else:
            trailing_mark_match = re.search(TRAILING_MARKS, clean_answer)
            marks = int(trailing_mark_match.group(1)) if trailing_mark_match else default_marks
            if trailing_mark_match:
                clean_answer = re.sub(TRAILING_MARKS, "", clean_answer).strip()
        clean_q_text = re.sub(MARKS_IN_Q, "", q_text).strip()
        questions[q_id] = {"text": clean_q_text, "marks": marks}
        model_answers[q_id] = clean_answer
    if not questions:
        raise ValueError("No Q/A pairs detected. Ensure format 'Q1: ... A1: ...'.")
    return questions, model_answers


def load_exam(exam_path: str | os.PathLike[str]) -> tuple[dict[str, dict], dict[str, str]]:
    """Load and parse a .txt or .pdf exam file."""
    path = Path(exam_path)
    extension = path.suffix.lower()
    if extension == ".pdf":
        text = read_pdf_text(path)
    elif extension == ".txt":
        text = read_text_file(path)
    else:
        raise ValueError("Unsupported exam file type; use .pdf or .txt")
    return parse_exam(text)


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Split OCR text into overlapping word chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    index = 0
    while index < len(words):
        chunks.append(" ".join(words[index : index + chunk_size]))
        index += max(chunk_size - overlap, 1)
    return chunks
