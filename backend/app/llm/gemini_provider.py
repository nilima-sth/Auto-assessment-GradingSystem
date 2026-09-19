from google import genai
from google.genai import types

from backend.app.core.config import get_settings
from backend.app.llm.base import SYSTEM_INSTRUCTION, build_grading_user_content
from backend.app.llm.grading_tools import build_grading_tool_declarations, execute_grading_tool
from backend.app.schemas.grading import GradeEvaluation


class GeminiProvider:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini.")
        self.settings = settings
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def _extract_function_call(self, response: object) -> tuple[str, dict] | None:
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                function_call = getattr(part, "function_call", None)
                if function_call:
                    return str(function_call.name), dict(function_call.args or {})
        return None

    def _run_model_selected_tool(
        self,
        question_number: str,
        question_text: str,
        model_answer: str,
        max_marks: float,
    ) -> str | None:
        if not self.settings.llm_tool_calling_enabled:
            return None

        response = self.client.models.generate_content(
            model=self.settings.gemini_model,
            contents=(
                "Choose the most useful registered grading tool before evaluation. "
                f"Use get_question_context for question {question_number}."
            ),
            config=types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=build_grading_tool_declarations())],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.ANY,
                        allowed_function_names=["get_question_context"],
                    )
                ),
                temperature=0.0,
            ),
        )
        function_call = self._extract_function_call(response)
        if not function_call:
            return None

        name, args = function_call
        tool_result = execute_grading_tool(
            name,
            args,
            question_number=question_number,
            question_text=question_text,
            model_answer=model_answer,
            max_marks=max_marks,
        )
        return f"Gemini tool call {name} returned: {tool_result}"

    def grade_answer(
        self,
        question_number: str,
        question_text: str,
        model_answer: str,
        student_answer: str,
        max_marks: float,
        retrieved_context: str | None = None,
    ) -> GradeEvaluation:
        config_kwargs = {
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": self.settings.llm_temperature,
            "top_p": self.settings.llm_top_p,
            "response_mime_type": "application/json",
            "response_schema": GradeEvaluation,
        }
        if self.settings.llm_max_output_tokens:
            config_kwargs["max_output_tokens"] = self.settings.llm_max_output_tokens

        tool_context = self._run_model_selected_tool(
            question_number,
            question_text,
            model_answer,
            max_marks,
        )
        combined_context = retrieved_context
        if tool_context:
            combined_context = f"{retrieved_context or ''}\n{tool_context}".strip()

        response = self.client.models.generate_content(
            model=self.settings.gemini_model,
            contents=build_grading_user_content(
                question_number,
                question_text,
                model_answer,
                student_answer,
                max_marks,
                combined_context,
            ),
            config=types.GenerateContentConfig(**config_kwargs),
        )

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, GradeEvaluation):
            evaluation = parsed
        elif parsed is not None:
            evaluation = GradeEvaluation.model_validate(parsed)
        else:
            evaluation = GradeEvaluation.model_validate_json(response.text)

        return GradeEvaluation(
            question_number=question_number,
            awarded_marks=evaluation.awarded_marks,
            max_marks=max_marks,
            feedback=evaluation.feedback,
        )
