from backend.app.schemas.grading import GradeEvaluation


def test_grade_evaluation_accepts_normal_grade() -> None:
    grade = GradeEvaluation(
        question_number="Q1",
        awarded_marks=4,
        max_marks=5,
        feedback="Mostly correct.",
    )

    assert grade.awarded_marks == 4


def test_grade_evaluation_clamps_negative_marks() -> None:
    grade = GradeEvaluation(
        question_number="Q1",
        awarded_marks=-2,
        max_marks=5,
        feedback="Incorrect.",
    )

    assert grade.awarded_marks == 0


def test_grade_evaluation_clamps_marks_above_max() -> None:
    grade = GradeEvaluation(
        question_number="Q1",
        awarded_marks=9,
        max_marks=5,
        feedback="Good answer.",
    )

    assert grade.awarded_marks == 5
