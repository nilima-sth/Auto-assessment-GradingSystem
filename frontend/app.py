import os
from typing import Iterable

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


def _file_tuple(uploaded_file) -> tuple[str, bytes, str]:
    return (
        uploaded_file.name,
        uploaded_file.getvalue(),
        uploaded_file.type or "application/octet-stream",
    )


def _post_files(path: str, files: Iterable[tuple[str, tuple[str, bytes, str]]], data: dict | None = None) -> dict:
    response = requests.post(f"{BACKEND_URL}{path}", files=list(files), data=data or {}, timeout=600)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(detail)
    return response.json()


st.set_page_config(page_title="Answer Sheet Grading", layout="wide")
st.title("AI-Assisted Answer Sheet Grading System")

exam_file = st.file_uploader("Exam / model-answer file", type=["txt", "pdf"])
reference_files = st.file_uploader(
    "Optional rubric or reference files",
    type=["txt", "pdf"],
    accept_multiple_files=True,
)
student_files = st.file_uploader(
    "Student answer-sheet PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

reference_set_id = st.session_state.get("reference_set_id")

col_ingest, col_grade = st.columns(2)
with col_ingest:
    if st.button("Ingest References", disabled=exam_file is None):
        try:
            files = [("exam_file", _file_tuple(exam_file))]
            files.extend(("reference_files", _file_tuple(item)) for item in reference_files)
            result = _post_files("/api/rag/ingest", files)
            st.session_state["reference_set_id"] = result["reference_set_id"]
            st.success(f"Indexed {result['chunks_indexed']} chunks.")
            st.code(result["reference_set_id"])
        except Exception as exc:
            st.error(f"Reference ingestion failed: {exc}")

with col_grade:
    if st.button("Start Grading", disabled=exam_file is None or not student_files):
        try:
            files = [("exam_file", _file_tuple(exam_file))]
            files.extend(("student_files", _file_tuple(item)) for item in student_files)
            data = {}
            if st.session_state.get("reference_set_id"):
                data["reference_set_id"] = st.session_state["reference_set_id"]
            st.session_state["grading_result"] = _post_files("/api/grading/evaluate", files, data=data)
        except Exception as exc:
            st.error(f"Grading failed: {exc}")

if reference_set_id:
    st.caption(f"Active reference set: {reference_set_id}")

result = st.session_state.get("grading_result")
if result:
    for student in result.get("students", []):
        st.subheader(student["student"])
        rows = [
            {
                "Question": item["question_number"],
                "Awarded": item["awarded_marks"],
                "Maximum": item["max_marks"],
                "Feedback": item["feedback"],
            }
            for item in student.get("questions", [])
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.metric("Total Marks", f"{student['total_awarded']} / {student['total_possible']}")
        if student.get("flagged_text"):
            st.warning(student["flagged_text"])
