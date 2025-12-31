# gsm8k_dataset.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from collabllm.datasets.single_turn import SingleTurnDataset


class GSM8K(SingleTurnDataset):
    """
    GSM8K ➟ SingleTurnDataset adaptor.

    Produces rows with at minimum:
        • prompt
        • completion

    HF mode also includes:
        • split

    Both modes include metadata:
        • source_task_id  (gsm8k/{idx})
    """

    def __init__(
        self,
        repo_id: str = "openai/gsm8k",
        *,
        eval_ratio: float = 0.1,
        seed: int = 42,
        load_lic_data: bool = False,
        lic_json_path: Optional[str | Path] = None,
    ):
        if load_lic_data:
            if lic_json_path is None:
                raise ValueError(
                    "When load_lic_data=True, you must provide lic_json_path."
                )
            processed = self._preprocess_lic(Path(lic_json_path))
        else:
            raw_ds = load_dataset(repo_id)  # GSM8K is a standard HF dataset
            processed = self._preprocess_hf(raw_ds)

        super().__init__(processed, eval_ratio=eval_ratio, seed=seed)

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _preprocess_hf(raw_ds) -> List[Dict[str, Any]]:
        """
        Convert HF DatasetDict into flat dict format expected by SingleTurnDataset.

        Keys:
            prompt, completion, split, source_task_id
        """
        processed: List[Dict[str, Any]] = []

        for split_name, split in raw_ds.items():
            for idx, row in enumerate(split):
                processed.append(
                    {
                        "prompt": row["question"],
                        "completion": row["answer"],
                        "split": split_name,
                        "source_task_id": f"gsm8k/{idx}",
                    }
                )

        return processed

    @staticmethod
    def _preprocess_lic(json_file: Path) -> List[Dict[str, Any]]:
        """
        Preprocess LIC JSON file as specified:

        - keep only items where item["task"] == "math"
        - question -> prompt
        - answer -> completion
        - task_id -> source_task_id, replacing "sharded-GSM8K/{n}" -> "gsm8k/{n}"

        NOTE: No 'split' key here; SingleTurnDataset will create train/eval
        splits using eval_ratio.
        """
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        processed: List[Dict[str, Any]] = []
        for item in data:
            if item.get("task") != "math":
                continue

            processed.append(
                {
                    "prompt": item["question"],
                    "completion": item["answer"],
                    "source_task_id": item["task_id"].replace("sharded-GSM8K", "gsm8k"),
                }
            )

        return processed
