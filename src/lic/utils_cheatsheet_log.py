# lic/utils_cheatsheet_log.py
import hashlib
import json
import os
from typing import Any, Dict


def _sha256(txt: str) -> str:
    return hashlib.sha256((txt or "").encode("utf-8")).hexdigest()


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_cheatsheet_record(
    *,
    iteration: int,
    dataset_split: str,
    task: str,
    task_id: str,
    assistant_model: str,
    curator_model: str,
    mode: str,
    is_correct: bool | None,
    score: float | None,
    extracted_answer: str | None,
    old_cheatsheet: str,
    new_cheatsheet: str,
    curator_cost_usd: float | None,
    timestamp: str | None,
) -> Dict[str, Any]:
    return {
        "iteration": iteration,
        "dataset_split": dataset_split,
        "task": task,
        "task_id": task_id,
        "assistant_model": assistant_model,
        "curator_model": curator_model,
        "mode": mode,
        "is_correct": is_correct,
        "score": score,
        "extracted_answer": extracted_answer,
        "old_cheatsheet_hash": _sha256(old_cheatsheet),
        "new_cheatsheet_hash": _sha256(new_cheatsheet),
        "old_cheatsheet": old_cheatsheet,
        "new_cheatsheet": new_cheatsheet,
        "curator_cost_usd": curator_cost_usd,
        "timestamp": timestamp,
    }
