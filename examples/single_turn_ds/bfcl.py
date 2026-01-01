# actions_bfcl_dataset.py
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from collabllm.datasets.single_turn import SingleTurnDataset
from collabmem.constants import LIC_DATA_PATH, LIC_PROMPT_DIRECTORY


def _arrow_safe(v: Any) -> Any:
    """
    Ensure values are Arrow-friendly inside a struct column.
    - dict/list -> JSON string
    - None -> "" (or keep None if you prefer, but then keep it consistent)
    - primitives pass through
    """
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return v


class BFCLSingleTurnDataset(SingleTurnDataset):
    def __init__(
        self,
        *,
        dataset_path: str = LIC_DATA_PATH,
        prompt_dir: str = LIC_PROMPT_DIRECTORY / "actions",
        train_ratio: float = 0.5,
        seed: int = 42,
        prompt_mode: str = "fully_specified",  # "fully_specified" | "concat_shards"
    ):
        self.dataset_path = Path(dataset_path)
        self.prompt_dir = Path(prompt_dir)
        self.train_ratio = train_ratio
        self.seed = seed
        self.prompt_mode = prompt_mode

        self.system_prompt_template = (
            self.prompt_dir / "actions_system_prompt.txt"
        ).read_text()
        self.full_prompt_template = (
            self.prompt_dir / "actions_full_prompt.txt"
        ).read_text()

        rows = self._load_lic_json(self.dataset_path)
        processed = self._preprocess(rows)

        super().__init__(processed, eval_ratio=1.0 - train_ratio, seed=seed)

    def _load_lic_json(self, path: Path) -> List[Dict[str, Any]]:
        with path.open("r") as f:
            data = json.load(f)
        return [d for d in data if d.get("task") == "actions"]

    def _generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt_template.replace(
            "[[FUNCTIONS]]", "{}".format(sample["function"])
        )

    def _populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        question = sample["fully_specified_question"][0][0]["content"]
        return self.full_prompt_template.replace("[[QUESTION]]", question)

    def _populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        query = ""
        for shard in sample.get("shards", []):
            query += f"- {shard['shard']}\n"
        return self.full_prompt_template.replace("[[QUESTION]]", query)

    def _preprocess(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        has_split = all(("split" in r) for r in rows)

        split_map: Optional[Dict[int, str]] = None
        if not has_split:
            n_total = len(rows)
            n_train = int(n_total * self.train_ratio)

            random.seed(self.seed)
            indices = list(range(n_total))
            random.shuffle(indices)
            split_map = {
                idx: ("train" if i < n_train else "test")
                for i, idx in enumerate(indices)
            }

        processed: List[Dict[str, Any]] = []
        for i, sample in enumerate(rows):
            split_tag = sample.get("split") if has_split else split_map[i]

            prompt = (
                self._populate_concat_prompt(sample)
                if self.prompt_mode == "concat_shards"
                else self._populate_fully_specific_prompt(sample)
            )
            system_prompt = self._generate_system_prompt(sample)

            # Keep completion as JSON payload (BigCodeBench pattern)
            ground_truth = {
                "dataset": "actions_bfcl",
                "task": "actions",
                "task_id": sample.get("task_id"),
                "reference_answer": sample.get("reference_answer"),
                "language": sample.get("language"),
                "test_category": sample.get("test_category"),
            }

            # IMPORTANT: every metadata value must be Arrow-safe + consistent
            processed.append(
                {
                    "prompt": prompt,
                    "completion": json.dumps(ground_truth, ensure_ascii=False),
                    "split": split_tag,
                    # metadata (all Arrow-safe scalars)
                    "task_id": _arrow_safe(sample.get("task_id")),
                    "system_prompt": _arrow_safe(system_prompt),
                    "function": _arrow_safe(sample.get("function")),
                    "reference_answer": _arrow_safe(sample.get("reference_answer")),
                    "language": _arrow_safe(sample.get("language", "python")),
                    "test_category": _arrow_safe(sample.get("test_category", "")),
                    "extraction_requirement": _arrow_safe(
                        "Return a series of valid function calls in the format "
                        "[func_name1(param1=value1, ...), func_name2(...)]. "
                        "You may include multiple function calls. Output only the calls."
                    ),
                    "answer_extraction_strategy": "full_response",
                    "prompt_mode": _arrow_safe(self.prompt_mode),
                }
            )

        return processed
