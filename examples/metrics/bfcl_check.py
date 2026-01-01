# bfcl_actions_metric.py
import json
import re
from typing import Any, Dict, List, Optional

from collabllm.metric import BaseMetric, SingleTurnOrChatMetric
from lic.tasks.actions.eval_bfcl import ast_checker, ast_parse


def _strip_code_fences(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    m = re.search(r"```(?:\w+)?\n(.*?)```", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text


def _maybe_json_load(x: Any) -> Any:
    # If it's a JSON-encoded string, decode; else return as-is
    if not isinstance(x, str):
        return x
    s = x.strip()
    if not s:
        return x
    if (s.startswith("{") and s.endswith("}")) or (
        s.startswith("[") and s.endswith("]")
    ):
        try:
            return json.loads(s)
        except Exception:
            return x
    return x


@SingleTurnOrChatMetric.register_metric("bfcl_actions")
class BFCLActionsMetric(BaseMetric):
    def score(
        self,
        prompt: str,
        groundtruth: str,
        completion: str,
        messages: Optional[List[Dict[str, str]]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if completion is None:
            raise ValueError("`completion` (model output) must be provided.")
        metadata = metadata or {}

        predicted = _strip_code_fences(completion)

        func_schema = _maybe_json_load(metadata.get("function"))
        reference_answer = _maybe_json_load(metadata.get("reference_answer"))
        language = metadata.get("language", "python")
        test_category = metadata.get("test_category", "")

        if func_schema in (None, "") or reference_answer in (None, ""):
            return {
                "score": 0.0,
                "is_correct": False,
                "error": "Missing function/reference_answer in metadata.",
            }

        try:
            decoded = ast_parse(predicted.strip(), language)
        except Exception as e:
            return {
                "score": 0.0,
                "is_correct": False,
                "error": f"AST parse failed: {e}",
            }

        result = ast_checker(
            func_schema,
            decoded,
            reference_answer,
            language,
            test_category,
            "gpt-4o",  # preserved from original TaskActions
        )
        valid = bool(result.get("valid", False))
        return {
            "score": 1.0 if valid else 0.0,
            "is_correct": valid,
            "error": result.get("error"),
        }
