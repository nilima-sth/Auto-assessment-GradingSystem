from __future__ import annotations

from google.genai import types


TOOL_NAMES = {
    "get_question_context",
    "validate_marks",
    "calculate_total_marks",
}


def build_grading_tool_declarations() -> list[types.FunctionDeclaration]:
    return [
        types.FunctionDeclaration(
            name="get_question_context",
            description="Return deterministic context for a question being graded.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "question_number": {"type": "string"},
                },
                "required": ["question_number"],
                "additionalProperties": False,
            },
        ),
        types.FunctionDeclaration(
            name="validate_marks",
            description="Check whether awarded marks are inside the allowed range.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "awarded_marks": {"type": "number"},
                    "max_marks": {"type": "number"},
                },
                "required": ["awarded_marks", "max_marks"],
                "additionalProperties": False,
            },
        ),
        types.FunctionDeclaration(
            name="calculate_total_marks",
            description="Calculate total awarded and possible marks for graded questions.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "question_grades": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "awarded_marks": {"type": "number"},
                                "max_marks": {"type": "number"},
                            },
                            "required": ["awarded_marks", "max_marks"],
                        },
                    },
                },
                "required": ["question_grades"],
                "additionalProperties": False,
            },
        ),
    ]


def execute_grading_tool(
    name: str,
    args: dict,
    *,
    question_number: str,
    question_text: str,
    model_answer: str,
    max_marks: float,
) -> dict:
    if name == "get_question_context":
        requested = str(args.get("question_number") or question_number)
        return {
            "question_number": requested,
            "question_text": question_text,
            "model_answer": model_answer,
            "max_marks": max_marks,
        }
    if name == "validate_marks":
        awarded = float(args.get("awarded_marks", 0.0))
        maximum = float(args.get("max_marks", max_marks))
        return {
            "valid": 0.0 <= awarded <= maximum,
            "awarded_marks": awarded,
            "max_marks": maximum,
        }
    if name == "calculate_total_marks":
        grades = args.get("question_grades") or []
        awarded_total = sum(float(item.get("awarded_marks", 0.0)) for item in grades)
        possible_total = sum(float(item.get("max_marks", 0.0)) for item in grades)
        return {
            "total_awarded": awarded_total,
            "total_possible": possible_total,
        }
    raise ValueError(f"Unsupported grading tool: {name}")
