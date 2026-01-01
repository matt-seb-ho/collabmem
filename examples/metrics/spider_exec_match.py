# spider_exec_metric.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from collabllm.metric import BaseMetric, SingleTurnOrChatMetric
from collabmem.constants import LIC_DATA_PATH
from lic.tasks.database.eval_spider_exec import eval_exec_match


def _strip_sql_fences(text: str) -> str:
    if text is None:
        return ""
    t = text.strip()
    # Prefer fenced SQL block if present
    m = re.search(r"```sql\s*(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Otherwise strip any fence generically
    t = re.sub(r"```(\w+)?", "", t).strip()
    return t


def _normalize_sql(sql: str) -> str:
    # Match LiC: collapse whitespace
    return re.sub(r"\s+", " ", (sql or "")).strip()


def _best_effort_extract_sql(text: str) -> str:
    """
    LiC mostly executes the whole extracted_answer; but models may add brief preface.
    We try to locate a plausible SQL start (WITH/SELECT) and keep the rest.
    """
    t = _strip_sql_fences(text)
    t = t.strip()

    # Find first SQL keyword likely to start a full query
    m = re.search(r"\b(WITH|SELECT)\b", t, flags=re.IGNORECASE)
    if m:
        t = t[m.start() :].strip()

    # Keep up to last semicolon if present (optional)
    if ";" in t:
        # allow trailing comments after semicolon; take up to last semicolon
        t = t[: t.rfind(";") + 1].strip()

    return t


@SingleTurnOrChatMetric.register_metric("spider_exec")
class SpiderExecMatchMetric(BaseMetric):
    """
    Execution-based Spider metric (exec match).
    Requires local DBs under data/spider/databases/{db_id}/
    """

    def score(
        self,
        prompt: str,
        groundtruth: str,
        completion: str,
        messages: Optional[List[Dict[str, str]]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        metadata = metadata or {}

        db_id = metadata.get("db_id")
        ref_sql = metadata.get("reference_sql")

        if not db_id or not ref_sql:
            return 0.0

        # base_dir = "data/spider/databases/"
        # db_dir = os.path.join(base_dir, str(db_id))
        # LIC_DATA_PATH points to lic/data/sharded_*.json
        base_dir = LIC_DATA_PATH.parent / "spider/databases"
        db_dir = base_dir / str(db_id)

        if not os.path.exists(base_dir):
            raise FileNotFoundError(
                "data/spider/databases/ folder not found; please see data/spider/README.md for instructions"
            )
        if not os.path.exists(db_dir):
            # If specific db missing, treat as not scorable (or raise; LiC raises only on base dir)
            raise FileNotFoundError(f"Spider DB folder not found: {db_dir}")

        pred_sql = _best_effort_extract_sql(completion)
        pred_sql = _normalize_sql(pred_sql)
        ref_sql = _normalize_sql(ref_sql)

        try:
            ok = (
                eval_exec_match(
                    db_dir + "/",  # LiC passes trailing slash
                    pred_sql,
                    ref_sql,
                    plug_value=True,
                    keep_distinct=False,
                    progress_bar_for_each_datapoint=False,
                )
                == 1
            )
        except Exception:
            ok = False

        return 1.0 if ok else 0.0
