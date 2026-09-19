import inspect

from backend.app.services import grading_service


def test_grade_batch_async_exposes_async_bounded_pipeline() -> None:
    assert inspect.iscoroutinefunction(grading_service.grade_batch_async)
    names = grading_service.grade_batch_async.__code__.co_names
    assert "Semaphore" in names
    assert "ThreadPoolExecutor" in names
    assert "_bounded_ocr_and_map" in names
    assert "_bounded_grade_answer" in names
