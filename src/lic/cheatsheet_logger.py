# lic/cheatsheet_logger.py
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class CheatsheetLogConfig:
    log_dir: str
    run_name: str  # e.g., "dc_warmup_math_gpt-5-mini"
    save_full_text: bool = True # save each revision as a .txt file
    jsonl_name: str = "cheatsheet_revisions.jsonl"


class CheatsheetLogger:
    def __init__(self, cfg: CheatsheetLogConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.cfg.log_dir, self.cfg.jsonl_name)

        # store texts here (optional)
        self.text_dir = os.path.join(self.cfg.log_dir, "cheatsheets")
        if self.cfg.save_full_text:
            os.makedirs(self.text_dir, exist_ok=True)

    def log_revision(
        self,
        *,
        dataset_fn: str,
        split_name: str,
        mode: str,
        sample: Dict[str, Any],
        models: Dict[str, str],  # assistant/system/user/curator
        eval_summary: Optional[Dict[str, Any]],
        curator_meta: Optional[Dict[str, Any]],
        cheatsheet_old: str,
        cheatsheet_new: str,
        episode_idx: int,
        cheatsheet_scope: Optional[str] = None,
    ) -> None:
        old_hash = _sha256(cheatsheet_old or "")
        new_hash = _sha256(cheatsheet_new or "")

        old_txt_path = None
        new_txt_path = None
        if self.cfg.save_full_text:
            old_txt_path = os.path.join(
                self.text_dir, f"{episode_idx:06d}_{old_hash}.txt"
            )
            new_txt_path = os.path.join(
                self.text_dir, f"{episode_idx:06d}_{new_hash}.txt"
            )
            # write (idempotent-ish)
            if not os.path.exists(old_txt_path):
                with open(old_txt_path, "w", encoding="utf-8") as f:
                    f.write(cheatsheet_old or "")
            if not os.path.exists(new_txt_path):
                with open(new_txt_path, "w", encoding="utf-8") as f:
                    f.write(cheatsheet_new or "")

        row = {
            "run_name": self.cfg.run_name,
            "dataset_fn": dataset_fn,
            "split": split_name,
            "mode": mode,
            "task": sample.get("task"),
            "task_id": sample.get("task_id"),
            "episode_idx": episode_idx,
            "models": models,
            "eval": eval_summary,
            "curator_meta": curator_meta,
            "cheatsheet_old_hash": old_hash,
            "cheatsheet_new_hash": new_hash,
            "cheatsheet_old_path": old_txt_path,
            "cheatsheet_new_path": new_txt_path,
            "cheatsheet_scope": cheatsheet_scope,
        }

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
