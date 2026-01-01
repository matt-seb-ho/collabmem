# data2text_dataset.py
from __future__ import annotations

import json
import random
from typing import Any, Dict, List

from collabllm.datasets.single_turn import SingleTurnDataset
from collabmem.constants import LIC_DATA_PATH, LIC_PROMPT_DIRECTORY


class ToTToSingleTurn(SingleTurnDataset):
    """
    Data2Text ➟ SingleTurnDataset adaptor (LiC).

    Each row exposes:
        • prompt      – fully specified data-to-text prompt
        • completion  – JSON-encoded ground-truth payload (references, task info)
        • split       – train / test (local split)

    All other fields are treated as metadata.
    """

    def __init__(
        self,
        dataset_file: str = LIC_DATA_PATH,
        prompt_directory: str = LIC_PROMPT_DIRECTORY / "data2text",
        *,
        train_ratio: float = 0.5,
        seed: int = 42,
    ):
        self.dataset_file = dataset_file
        self.prompt_directory = prompt_directory
        self.train_ratio = train_ratio
        self.seed = seed

        # load prompt templates
        with open(self.prompt_directory / "data2text_full_prompt.txt", "r") as f:
            self.fully_specified_prompt = f.read()
        with open(self.prompt_directory / "data2text_system_prompt.txt", "r") as f:
            self.system_prompt = f.read()

        raw_samples = self._load_samples()
        processed = self._preprocess(raw_samples)

        super().__init__(processed, eval_ratio=1.0 - train_ratio, seed=seed)

    # ------------------------------------------------------------------ #
    # loading                                                             #
    # ------------------------------------------------------------------ #
    def _load_samples(self) -> List[Dict[str, Any]]:
        with open(self.dataset_file, "r") as f:
            data = json.load(f)
        return [d for d in data if d["task"] == "data2text"]

    # ------------------------------------------------------------------ #
    # preprocessing                                                       #
    # ------------------------------------------------------------------ #
    def _preprocess(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        n_total = len(samples)
        n_train = int(n_total * self.train_ratio)

        random.seed(self.seed)
        indices = list(range(n_total))
        random.shuffle(indices)

        split_map = {
            idx: ("train" if i < n_train else "test") for i, idx in enumerate(indices)
        }

        processed = []
        for idx, sample in enumerate(samples):
            split_tag = split_map[idx]

            prompt = self._populate_fully_specified_prompt(sample)

            # JSON payload for consistency with other CollabLLM datasets
            ground_truth = {
                "dataset": "data2text",
                "task_id": sample["task_id"],
                "references": sample["references"],
            }

            processed.append(
                {
                    # mandatory
                    "prompt": prompt,
                    "completion": json.dumps(ground_truth),
                    # metadata
                    "split": split_tag,
                    "task_id": sample["task_id"],
                    "references": sample["references"],
                    "table_html": sample["table_html"],
                    "table_highlighted_html": sample["table_highlighted_html"],
                    "metadata": sample["metadata"],
                    # "system_prompt": self.system_prompt,
                    "answer_extraction_strategy": "full_response",
                }
            )

        return processed

    # ------------------------------------------------------------------ #
    # prompt construction                                                 #
    # ------------------------------------------------------------------ #
    def _populate_fully_specified_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Direct port of TaskData2Text.populate_fully_specific_prompt.
        """
        prompt = self.fully_specified_prompt
        prompt = prompt.replace("[[TABLE_HTML]]", sample["table_highlighted_html"])
        prompt = prompt.replace(
            "[[FEWSHOT_DESCRIPTIONS]]", sample["fewshot_descriptions"]
        )

        metadata_str = ""
        for key, value in sample["metadata"].items():
            metadata_str += f"{key}: {value}\n"

        prompt = prompt.replace("[[CONTEXT]]", metadata_str)
        return prompt
