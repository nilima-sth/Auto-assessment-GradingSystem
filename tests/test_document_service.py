from pathlib import Path

from backend.app.services.document_service import load_exam, parse_exam


def test_sample_qna_parses_questions_and_marks() -> None:
    questions, model_answers = load_exam(Path("data/exam_files/qna.txt"))

    assert sorted(questions) == ["Q1", "Q2"]
    assert questions["Q1"]["marks"] == 10
    assert questions["Q2"]["marks"] == 5
    assert "Difference Between e-Government" in model_answers["Q1"]
    assert not model_answers["Q1"].endswith("(10)")


def test_marks_in_question_text_still_parse() -> None:
    questions, model_answers = parse_exam("Q1: Explain testing. (7)\nA1: Testing checks behavior.")

    assert questions["Q1"]["marks"] == 7
    assert questions["Q1"]["text"] == "Explain testing."
    assert model_answers["Q1"] == "Testing checks behavior."
