# actions_bfcl_dataset.py
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from collabllm.datasets.single_turn import SingleTurnDataset
from collabmem.constants import LIC_DATA_PATH, LIC_PROMPT_DIRECTORY


class BFCLSingleTurnDataset(SingleTurnDataset):
    """
    Actions / Berkeley Function Calling Leaderboard (BFCL-style) ➟ SingleTurnDataset adaptor.

    - Loads LIC JSON (default: data/sharded_instructions_600.json)
    - Filters rows where row["task"] == "actions"
    - Builds fully-specified prompt using prompts/actions/actions_full_prompt.txt
    - Stores system_prompt (with functions inserted) in metadata
    - Stores groundtruth as JSON in `completion` (BigCodeBench-style)
    """

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

        # Load prompt templates (same as TaskActions.__init__)
        self.system_prompt_template = (
            self.prompt_dir / "actions_system_prompt.txt"
        ).read_text()
        self.full_prompt_template = (
            self.prompt_dir / "actions_full_prompt.txt"
        ).read_text()

        raw = self._load_lic_json(self.dataset_path)
        processed = self._preprocess(raw)

        super().__init__(processed, eval_ratio=1.0 - train_ratio, seed=seed)

    # ------------------------------------------------------------------ #
    # Loading                                                            #
    # ------------------------------------------------------------------ #
    def _load_lic_json(self, path: Path) -> List[Dict[str, Any]]:
        with path.open("r") as f:
            data = json.load(f)
        # same filter as TaskActions.get_samples()
        return [d for d in data if d.get("task") == "actions"]

    # ------------------------------------------------------------------ #
    # Prompt logic (reused from TaskActions)                              #
    # ------------------------------------------------------------------ #
    def _generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        # same as TaskActions.generate_system_prompt()
        return self.system_prompt_template.replace(
            "[[FUNCTIONS]]", "{}".format(sample["function"])
        )

    def _populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        # same as TaskActions.populate_fully_specific_prompt()
        question = sample["fully_specified_question"][0][0]["content"]
        return self.full_prompt_template.replace("[[QUESTION]]", question)

    def _populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        # same as TaskActions.populate_concat_prompt()
        query = ""
        for shard in sample.get("shards", []):
            query += f"- {shard['shard']}\n"
        return self.full_prompt_template.replace("[[QUESTION]]", query)

    # ------------------------------------------------------------------ #
    # Preprocessing                                                      #
    # ------------------------------------------------------------------ #
    def _preprocess(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # If the dataset already has a split field, respect it; otherwise create local split.
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

            # Prompt selection
            if self.prompt_mode == "concat_shards":
                prompt = self._populate_concat_prompt(sample)
            else:
                prompt = self._populate_fully_specific_prompt(sample)

            system_prompt = self._generate_system_prompt(sample)

            # BigCodeBench-style JSON-encoded ground truth payload in completion
            ground_truth = {
                "dataset": "actions_bfcl",
                "task": "actions",
                "task_id": sample.get("task_id"),
                "reference_answer": sample.get("reference_answer"),
                "language": sample.get("language"),
                "test_category": sample.get("test_category"),
            }

            processed.append(
                {
                    # required
                    "prompt": prompt,
                    "completion": json.dumps(ground_truth),
                    # metadata (available to metric)
                    "split": split_tag,
                    "task_id": sample.get("task_id"),
                    "system_prompt": system_prompt,
                    "function": sample.get("function"),
                    "reference_answer": sample.get("reference_answer"),
                    "language": sample.get("language", "python"),
                    "test_category": sample.get("test_category"),
                    # helpful / optional (mirrors TaskActions.get_answer_description)
                    "extraction_requirement": (
                        "Return a series of valid function calls in the format "
                        "[func_name1(param1=value1, ...), func_name2(...)]. "
                        "You may include multiple function calls. Output only the calls."
                    ),
                    "answer_extraction_strategy": "full_response",
                    "prompt_mode": self.prompt_mode,
                }
            )

        return processed
