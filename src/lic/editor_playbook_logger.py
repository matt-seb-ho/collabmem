# lic/editor_cheatsheet_logger.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class EditorCheatsheetLogConfig:
    log_dir: str
    run_name: str
    save_full_text: bool = True  # save cheatsheet snapshots to txt


class EditorCheatsheetLogger:
    """
    Logs each cheatsheet revision to:
      - revisions.jsonl (one record per episode)
      - optional: cheatsheet_{episode_idx:06d}.txt (old/new snapshots)
    """

    def __init__(self, cfg: EditorCheatsheetLogConfig):
        self.cfg = cfg
        os.makedirs(cfg.log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(cfg.log_dir, "revisions.jsonl")
        self.snap_dir = os.path.join(cfg.log_dir, "snapshots")
        if cfg.save_full_text:
            os.makedirs(self.snap_dir, exist_ok=True)

    def _write_txt(self, name: str, text: str) -> str:
        path = os.path.join(self.snap_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text or "")
        return path

    def log_revision(
        self,
        *,
        dataset_fn: str,
        split_name: str,
        mode: str,
        sample: Dict[str, Any],
        models: Dict[str, str],
        eval_summary: Optional[Dict[str, Any]],
        curator_meta: Optional[Dict[str, Any]],
        cheatsheet_old: str,
        cheatsheet_new: str,
        episode_idx: int,
        cheatsheet_scope: str,
        task_key: str,
    ) -> None:
        record: Dict[str, Any] = {
            "run_name": self.cfg.run_name,
            "dataset_fn": dataset_fn,
            "split_name": split_name,
            "mode": mode,
            "episode_idx": episode_idx,
            "task": sample.get("task"),
            "task_id": sample.get("task_id"),
            "cheatsheet_scope": cheatsheet_scope,
            "task_key": task_key,
            "models": models,
            "eval_summary": eval_summary,
            "curator_meta": curator_meta,
        }

        if self.cfg.save_full_text:
            old_path = self._write_txt(
                f"cheatsheet_old_{episode_idx:06d}.txt", cheatsheet_old
            )
            new_path = self._write_txt(
                f"cheatsheet_new_{episode_idx:06d}.txt", cheatsheet_new
            )
            record["cheatsheet_old_path"] = old_path
            record["cheatsheet_new_path"] = new_path
        else:
            record["cheatsheet_old_preview"] = (cheatsheet_old or "")[:500]
            record["cheatsheet_new_preview"] = (cheatsheet_new or "")[:500]

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
