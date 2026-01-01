# lic_code_dataset.py
from __future__ import annotations

import base64
import json
import pickle
import random
import zlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from collabllm.datasets.single_turn import SingleTurnDataset
from collabmem.constants import LIC_DATA_PATH, LIC_PROMPT_DIRECTORY

# ------------------------- LiC Default Prompt Paths ------------------------- #

LCB_FULL_PROMPT_PATH = LIC_PROMPT_DIRECTORY / "lcb/lcb_full_prompt.txt"
LCB_SYSTEM_PROMPT_PATH = LIC_PROMPT_DIRECTORY / "lcb/lcb_system_prompt.txt"
HUMANEVAL_FULL_PROMPT_PATH = (
    LIC_PROMPT_DIRECTORY / "humaneval/humaneval_full_prompt.txt"
)
HUMANEVAL_SYSTEM_PROMPT_PATH = (
    LIC_PROMPT_DIRECTORY / "humaneval/humaneval_system_prompt.txt"
)


# ------------------------- LiC test decoding ------------------------- #


def _decode_private_test_cases(value: str) -> list[dict]:
    """
    LiC json may store private_test_cases as either:
      - JSON string
      - base64(zlib(pickle(...))) string
    """
    try:
        return json.loads(value)
    except Exception:
        pass

    try:
        return json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(value.encode("utf-8"))))
        )
    except Exception:
        return []


def _load_testcases_as_json(sample: Dict[str, Any]) -> str:
    """
    Mirrors TaskCode.load_test_cases.

    Returns JSON string:
      { "inputs": [...], "outputs": [...], "fn_name": <func_name or None> }
    """
    public_test_cases = json.loads(sample["public_test_cases"])
    private_test_cases: list[dict] = []

    if "private_test_cases" in sample and sample["private_test_cases"]:
        private_test_cases = _decode_private_test_cases(sample["private_test_cases"])

    merged = public_test_cases + private_test_cases
    return json.dumps(
        {
            "inputs": [t.get("input") for t in merged],
            "outputs": [t.get("output") for t in merged],
            "fn_name": (sample.get("metadata") or {}).get("func_name", None),
        }
    )


# ------------------------- Prompt templating ------------------------- #


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _get_formatting_preamble_and_starter(sample: Dict[str, Any]) -> Tuple[str, str]:
    """
    Mirrors TaskCode.get_formatting_preamble.
    """
    call_preamble = (
        "You will use the following starter code to write the solution to the problem "
        "and enclose your code within delimiters."
    )
    starter = sample.get("starter_code") or "# YOUR CODE HERE"
    return call_preamble, f"```python\n{starter}\n```"


class LiCCode(SingleTurnDataset):
    """
    LiC Code (HumanEval + LiveCodeBench mix) ➟ SingleTurnDataset adaptor.

    Uses LiC json by default and prompt template files to create the *fully specified*
    prompt for each sample (matching TaskCode.populate_fully_specific_prompt).

    Each item contains at least:
      - prompt
      - completion   (JSON payload of ground-truth/eval fields)
      - split        (train/test)
    """

    def __init__(
        self,
        *,
        lic_json_path: str | Path = LIC_DATA_PATH,
        # prompt file paths (pass your actual paths here)
        lcb_full_prompt_path: str | Path = LCB_FULL_PROMPT_PATH,
        lcb_system_prompt_path: str | Path = LCB_SYSTEM_PROMPT_PATH,
        humaneval_full_prompt_path: str | Path = HUMANEVAL_FULL_PROMPT_PATH,
        humaneval_system_prompt_path: str | Path = HUMANEVAL_SYSTEM_PROMPT_PATH,
        # split control
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.train_ratio = train_ratio
        self.seed = seed

        # load prompt templates (exactly like TaskCode.__init__)
        self.fully_specified_prompt_lcb = _read_text(Path(lcb_full_prompt_path))
        self.system_prompt_lcb = _read_text(Path(lcb_system_prompt_path))
        self.fully_specified_prompt_humaneval = _read_text(
            Path(humaneval_full_prompt_path)
        )
        self.system_prompt_humaneval = _read_text(Path(humaneval_system_prompt_path))

        processed = self._preprocess_lic(Path(lic_json_path))
        super().__init__(processed, eval_ratio=1.0 - train_ratio, seed=seed)

    # ---------------------- TaskCode-equivalent prompt builders ---------------------- #

    def _populate_fully_specific_prompt_lcb(self, sample: Dict[str, Any]) -> str:
        query = sample["question_content"]
        formatting_preamble, starter_code = _get_formatting_preamble_and_starter(sample)

        return (
            self.fully_specified_prompt_lcb.replace("[[QUESTION]]", query)
            .replace("[[FORMATTING_PREAMBLE]]", formatting_preamble)
            .replace("[[FORMATTING]]", starter_code)
        )

    def _populate_fully_specific_prompt_humaneval(self, sample: Dict[str, Any]) -> str:
        user_query = sample["prompt"]
        return self.fully_specified_prompt_humaneval.replace(
            "[[INSTRUCTION]]",
            f"Complete the following incomplete function signature:\n```python\n{user_query}\n```",
        )

    def _populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        src = sample.get("source")
        if src in ["lcb_easy", "lcb_medium"]:
            return self._populate_fully_specific_prompt_lcb(sample)
        elif src == "humaneval":
            return self._populate_fully_specific_prompt_humaneval(sample)
        else:
            raise ValueError(f"Invalid source: {src}")

    def _system_prompt(self, sample: Dict[str, Any]) -> str:
        # TaskCode: use LCB system prompt for all samples
        return self.system_prompt_lcb

    # ---------------------- Preprocess ---------------------- #

    def _preprocess_lic(self, json_file: Path) -> List[Dict[str, Any]]:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        samples = [d for d in data if d.get("task") == "code"]

        # reproducible train/test split
        n_total = len(samples)
        n_train = int(n_total * self.train_ratio)

        random.seed(self.seed)
        indices = list(range(n_total))
        random.shuffle(indices)
        split_map = {
            idx: ("train" if i < n_train else "test") for i, idx in enumerate(indices)
        }

        processed: List[Dict[str, Any]] = []
        for i, sample in enumerate(samples):
            split_tag = split_map[i]

            task_id = sample.get("task_id", f"code/{i}")
            source = sample.get("source")

            # Stable id for downstream bookkeeping (in spirit of gsm8k/{idx})
            source_task_id = (
                str(task_id)
                .replace("sharded-", "")
                .replace("sharded_", "")
                .replace("sharded", "")
            )

            # fully specified prompt using the template files
            prompt = self._populate_fully_specific_prompt(sample)

            # tests
            testcases_json = _load_testcases_as_json(sample)
            func_name = (sample.get("metadata") or {}).get("func_name", None)

            # Ground-truth payload (like BigCodeBench’s completion packaging)
            ground_truth = {
                "dataset": "lic_code",
                "task_id": task_id,
                "source": source,
                "testcases": testcases_json,
                "fn_name": func_name,
                "starter_code": sample.get("starter_code", None),
                # keep original prompt for HE import extraction
                "humaneval_prompt": sample.get("prompt", None)
                if source == "humaneval"
                else None,
            }

            processed.append(
                {
                    # required
                    "prompt": prompt,
                    "completion": json.dumps(ground_truth),
                    # split
                    "split": split_tag,
                    # metadata (used by metric)
                    "task_id": task_id,
                    "source": source,
                    "source_task_id": source_task_id,
                    "system_prompt": self._system_prompt(sample),
                    "testcases": testcases_json,
                    "func_name": func_name,
                    "starter_code": sample.get("starter_code", None),
                    "humaneval_raw_prompt": sample.get("prompt", None)
                    if source == "humaneval"
                    else None,
                    # helpful extraction requirement
                    "extraction_requirement": (
                        "Return executable Python code only. Provide a valid function definition "
                        "(def ...) that satisfies all test cases. Do not include explanations."
                    ),
                }
            )

        return processed
