# lic_code_pass_rate_metric.py
from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional

# Adjust these imports to your CollabLLM metric registry locations
from collabllm.metrics.base import BaseMetric
from collabllm.metrics.registry import SingleTurnOrChatMetric

# This is from your provided codebase
from lic.tasks.code.eval_code import check_correctness


def _strip_markdown_fences(text: str) -> str:
    if text is None:
        return ""
    return text.replace("```python", "").replace("```", "").strip()


def _prepend_imports_from_prompt(pred_code: str, humaneval_prompt: str) -> str:
    """
    Mirrors TaskCode’s “extract imports from sample['prompt'] and prepend them”.
    """
    if not humaneval_prompt:
        return pred_code

    try:
        prompt_ast = ast.parse(humaneval_prompt)
    except Exception:
        return pred_code

    imports: list[str] = []
    for node in prompt_ast.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            try:
                imports.append(ast.unparse(node))
            except Exception:
                # best-effort; skip if unparsing fails
                pass

    if imports:
        return "\n".join(imports) + "\n\n" + pred_code
    return pred_code


def _force_function_name(pred_code: str, true_name: str) -> str:
    """
    Mirrors TaskCode’s name-forcing:
      old_name = pred_code.split("def ")[1].split("(")[0].strip()
      pred_code = pred_code.replace(old_name, true_name)
    but does it a bit more safely (replace only the first def name occurrence).
    """
    if not true_name or "def " not in pred_code:
        return pred_code

    try:
        after_def = pred_code.split("def ", 1)[1]
        old_name = after_def.split("(", 1)[0].strip()
    except Exception:
        return pred_code

    if not old_name or old_name == true_name:
        return pred_code

    # Replace only the first occurrence of `def old_name(`
    return pred_code.replace(f"def {old_name}(", f"def {true_name}(", 1)


@SingleTurnOrChatMetric.register_metric("lic_code_pass_rate")
class LICCodePassRateMetric(BaseMetric):
    """
    Executes model-generated Python code against LIC (LCB/HE) testcases.

    Expects dataset adaptor to provide in `metadata` at least:
      - testcases: JSON string with keys inputs/outputs/fn_name
      - func_name: canonical function name (optional but recommended)
      - source: "humaneval" or "lcb_*"
      - humaneval_raw_prompt: original HE prompt (optional; used for imports)
    """

    def score(
        self,
        prompt: str,
        groundtruth: str,
        completion: str,
        messages: Optional[List[Dict[str, str]]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        if completion is None:
            raise ValueError("`completion` (candidate code) must be provided.")
        if metadata is None:
            raise ValueError("`metadata` is required for code pass-rate evaluation.")

        pred_python_code = _strip_markdown_fences(completion)

        # Must contain at least one function
        if "def " not in pred_python_code:
            return 0.0

        source = metadata.get("source", None)

        # HumanEval-derived samples: prepend imports from the original prompt
        if source == "humaneval":
            pred_python_code = _prepend_imports_from_prompt(
                pred_python_code, metadata.get("humaneval_raw_prompt", "") or ""
            )

        # Force correct function name if provided
        true_name = metadata.get("func_name", None)
        if true_name:
            pred_python_code = _force_function_name(pred_python_code, true_name)

        # Run correctness check
        testcases = metadata.get("testcases", None)
        if not testcases:
            # If you later add HF to supply missing tests, this is where you'd hook it in.
            raise ValueError(
                "Missing `testcases` in metadata; cannot score completion."
            )

        output, _extra = check_correctness(
            metadata, pred_python_code, testcases, timeout=6
        )
        # passed = all(o is True for o in output)
        # return 1.0 if passed else 0.0
        num_passed = sum(1 for o in output if o is True)
        return num_passed / len(output) if output else 0.0
