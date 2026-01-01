# spider_database_dataset.py
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from collabllm.datasets.single_turn import SingleTurnDataset
from collabmem.constants import LIC_DATA_PATH, LIC_PROMPT_DIRECTORY


def _arrow_safe(v: Any) -> Any:
    """Ensure metadata values are Arrow-friendly scalars."""
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return v


class SpiderDatabaseSingleTurn(SingleTurnDataset):
    """
    Spider-style text-to-SQL (LiC 'database' task) ➟ SingleTurnDataset adaptor.

    - prompt: filled database_full_prompt.txt
    - completion: JSON string payload (ground truth + identifiers)
    - metadata: includes system_prompt (schema-injected) for deterministic injection in simulator
    """

    def __init__(
        self,
        *,
        dataset_path: str = LIC_DATA_PATH,
        prompt_dir: str = LIC_PROMPT_DIRECTORY / "database",
        train_ratio: float = 0.5,
        seed: int = 42,
        prompt_mode: str = "fully_specified",  # "fully_specified" | "concat_shards"
    ):
        self.dataset_path = Path(dataset_path)
        self.prompt_dir = Path(prompt_dir)
        self.train_ratio = train_ratio
        self.seed = seed
        self.prompt_mode = prompt_mode

        self.full_prompt_template = (
            self.prompt_dir / "database_full_prompt.txt"
        ).read_text()
        self.system_prompt_template = (
            self.prompt_dir / "database_system_prompt.txt"
        ).read_text()

        rows = self._load_lic_json()
        processed = self._preprocess(rows)

        super().__init__(processed, eval_ratio=1.0 - train_ratio, seed=seed)

    def _load_lic_json(self) -> List[Dict[str, Any]]:
        with self.dataset_path.open("r") as f:
            data = json.load(f)
        return [d for d in data if d.get("task") == "database"]

    # ---------------- prompt logic (reused from TaskDatabase) ----------------
    def _generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt_template.replace("[[SCHEMA]]", sample["schema_sql"])

    def _populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        return self.full_prompt_template.replace(
            "[[DATABASE_SCHEMA]]", sample["schema_sql"]
        ).replace("[[USER_QUERY]]", sample["fully_specified_question"])

    def _populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        user_query = "Consider all the following:\n"
        for shard in sample.get("shards", []):
            user_query += f"- {shard['shard']}\n"

        return self.full_prompt_template.replace(
            "[[DATABASE_SCHEMA]]", sample["schema_sql"]
        ).replace("[[USER_QUERY]]", user_query)

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

            completion_payload = {
                "dataset": "spider_database",
                "task_id": sample.get("task_id"),
                "db_id": sample.get("db_id"),
                "reference_sql": sample.get("reference_sql"),
                "spider_difficulty": sample.get("spider_difficulty", "NA"),
            }

            processed.append(
                {
                    # required
                    "prompt": prompt,
                    "completion": json.dumps(completion_payload, ensure_ascii=False),
                    # optional split
                    "split": split_tag,
                    # metadata (Arrow-safe)
                    "task_id": _arrow_safe(sample.get("task_id")),
                    "db_id": _arrow_safe(sample.get("db_id")),
                    "schema_sql": _arrow_safe(sample.get("schema_sql")),
                    "reference_sql": _arrow_safe(sample.get("reference_sql")),
                    "spider_difficulty": _arrow_safe(
                        sample.get("spider_difficulty", "NA")
                    ),
                    # critical for your new deterministic injection
                    "system_prompt": _arrow_safe(system_prompt),
                    # misc
                    "prompt_mode": _arrow_safe(self.prompt_mode),
                    "answer_extraction_strategy": "prefix_suffix",
                    "extraction_requirement": _arrow_safe(
                        "Output ONLY a complete executable SQL query (no prose). "
                        "If you include a code fence, it must contain only the SQL."
                    ),
                }
            )

        return processed
