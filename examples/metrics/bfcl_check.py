# bfcl_actions_metric.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from collabllm.metric import BaseMetric, SingleTurnOrChatMetric
from lic.tasks.actions.eval_bfcl import ast_checker, ast_parse


def _strip_code_fences(text: str) -> str:
    """
    BFCL answers are usually raw function-call strings, but models sometimes wrap
    in ```...``` fences. This keeps behavior faithful while being robust.
    """
    if text is None:
        return ""
    text = text.strip()

    # If there's a fenced block, prefer its contents (first block).
    m = re.search(r"```(?:\w+)?\n(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    return text


@SingleTurnOrChatMetric.register_metric("bfcl_actions")
class BFCLActionsMetric(BaseMetric):
    """
    Actions/BFCL metric using the original LIC evaluator:
      - ast_parse
      - ast_checker

    Expects in metadata:
      - function: function schema / tool definitions string
      - reference_answer: gold function call(s)
      - language: language identifier understood by ast_parse
      - test_category: BFCL category string (passed through)
    """

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
        if metadata is None:
            metadata = {}

        predicted_answer = _strip_code_fences(completion)

        # Pull required fields from metadata
        func_schema = metadata.get("function")
        reference_answer = metadata.get("reference_answer")
        language = metadata.get("language", "python")
        test_category = metadata.get("test_category", "")

        if func_schema is None or reference_answer is None:
            return {
                "score": 0.0,
                "is_correct": False,
                "error": "Missing required metadata: function and/or reference_answer.",
            }

        try:
            decoded_output = ast_parse(predicted_answer.strip(), language)
        except Exception as e:
            return {
                "score": 0.0,
                "is_correct": False,
                "error": f"Failing to parse the predicted answer as an AST: {e}",
            }

        # Note: original code hardcodes model_name="gpt-4o" as a parameter to ast_checker.
        # We preserve that for faithful reproduction.
        result = ast_checker(
            func_schema,
            decoded_output,
            reference_answer,
            language,
            test_category,
            "gpt-4o",
        )

        valid = bool(result.get("valid", False))
        return {
            "score": 1.0 if valid else 0.0,
            "is_correct": valid,
            "error": result.get("error"),
        }
