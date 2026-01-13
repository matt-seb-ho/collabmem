# lic/cheatsheet_memory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lic.model import generate
from lic.utils import date_str, extract_conversation
from lic.constants import PROMPT_FILE_PATHS


VANILLA_UPDATE_PROMPT = PROMPT_FILE_PATHS["dc_cu_vanilla"].read_text()
# LIC_UPDATE_PROMPT = PROMPT_FILE_PATHS["dc_cu_lic"].read_text()
LIC_UPDATE_PROMPT = PROMPT_FILE_PATHS["cheatsheet_v2"].read_text()


@dataclass
class CheatsheetConfig:
    # How the assistant sees the cheatsheet each episode
    inject_mode: str = "system"  # "system" recommended; could support "user"
    max_cheatsheet_chars: int = 12000  # safety guard

    # Reflection/update
    enable_updates: bool = True
    curator_model: str = "gpt-4o-mini"
    curator_temperature: float = 0.0
    curator_max_tokens: int = 2000

    # Extrinsic grounding toggles (ablatable)
    include_eval_label: bool = False  # (a) correctness label / eval summary
    include_full_spec_q: bool = False  # (b) single-turn fully specified question
    include_ground_truth: bool = False  # (c) ground truth output/answer

    # Prompt template
    cheatsheet_update_template: str = LIC_UPDATE_PROMPT


class CheatsheetMemory:
    def __init__(self, initial_cheatsheet: str, cfg: CheatsheetConfig):
        self.cheatsheet = initial_cheatsheet or "(empty)"
        self.cfg = cfg

    def build_assistant_preamble(self) -> str:
        cs = self.cheatsheet
        if len(cs) > self.cfg.max_cheatsheet_chars:
            cs = cs[-self.cfg.max_cheatsheet_chars :]  # simple truncation fallback
        return (
            "You have access to a persistent CHEATSHEET to improve performance across tasks.\n"
            "Use it as guidance and reusable snippets; avoid repeating past mistakes.\n\n"
            "CHEATSHEET:\n<<<\n"
            f"{cs}\n"
            ">>>\n"
        )

    @staticmethod
    def _safe_str(x: Any) -> str:
        """Cast anything reasonable to a string, preserving None as '(not provided)'."""
        if x is None:
            return "(not provided)"
        if isinstance(x, str):
            return x
        # For dict/list/etc., repr is often more informative than str
        try:
            return str(x)
        except Exception:
            return repr(x)

    @staticmethod
    def _replace_if_present(template: str, placeholder: str, value: str) -> str:
        """Replace placeholder if present; no-op otherwise."""
        if placeholder in template:
            return template.replace(placeholder, value)
        return template

    def _render_update_prompt(
        self,
        task_name: str,
        system_message: str,
        trace: List[Dict[str, Any]],
        eval_summary: Optional[Dict[str, Any]],
        full_spec_q: Optional[str],
        ground_truth_a: Any,
    ) -> str:
        # Conversation trajectory as primary reflection target
        conversation_txt = extract_conversation(trace, to_str=True)

        # Model answer: last assistant utterance is usually the final attempt
        model_answer = ""
        for msg in reversed(trace):
            if msg.get("role") == "assistant":
                model_answer = msg.get("content", "")
                break

        # Optional extrinsic grounding fields
        correctness_label_txt = "(not provided)"
        if self.cfg.include_eval_label and eval_summary is not None:
            correctness_label_txt = (
                "EVALUATION LABEL / SUMMARY:\n"
                f"- is_correct: {self._safe_str(eval_summary.get('is_correct'))}\n"
                f"- score: {self._safe_str(eval_summary.get('score'))}\n"
                f"- extracted_answer: {self._safe_str(eval_summary.get('extracted_answer'))}\n"
            )

        full_spec_txt = "(not provided)"
        if self.cfg.include_full_spec_q:
            full_spec_txt = self._safe_str(full_spec_q)

        gt_txt = "(not provided)"
        if self.cfg.include_ground_truth:
            gt_txt = self._safe_str(ground_truth_a)

        # Backwards-compat: some templates still expect [[QUESTION]] as a single blob.
        # We'll provide a compact episode header there, but also populate the new fields
        # if present in the template.
        question_blob = (
            f"TASK: {task_name}\n\n"
            f"SYSTEM PROMPT:\n{system_message}\n\n"
            f"CONVERSATION:\n{conversation_txt}\n"
        )

        tmpl = self.cfg.cheatsheet_update_template

        # Always fill legacy placeholders
        tmpl = tmpl.replace("[[PREVIOUS_CHEATSHEET]]", self.cheatsheet)
        tmpl = tmpl.replace("[[MODEL_ANSWER]]", model_answer)
        tmpl = tmpl.replace("[[QUESTION]]", question_blob)  # harmless if unused

        # Fill new LiC prompt placeholders if present
        tmpl = self._replace_if_present(tmpl, "[[CONVERSATION]]", conversation_txt)
        tmpl = self._replace_if_present(
            tmpl, "[[CORRECTNESS_LABEL]]", correctness_label_txt
        )
        tmpl = self._replace_if_present(
            tmpl, "[[FULL_TASK_SPEC_SINGLE_TURN]]", full_spec_txt
        )
        tmpl = self._replace_if_present(tmpl, "[[GROUND_TRUTH_OUTPUT]]", gt_txt)

        return tmpl

    def maybe_update(
        self,
        task_name: str,
        system_message: str,
        trace: List[Dict[str, Any]],
        eval_summary: Optional[Dict[str, Any]],
        full_spec_q: Optional[str] = None,
        ground_truth_a: Any = None,
    ) -> Dict[str, Any]:
        if not self.cfg.enable_updates:
            return {"updated": False, "old": self.cheatsheet, "new": self.cheatsheet}

        prompt = self._render_update_prompt(
            task_name=task_name,
            system_message=system_message,
            trace=trace,
            eval_summary=eval_summary,
            full_spec_q=full_spec_q,
            ground_truth_a=ground_truth_a,
        )

        resp = generate(
            [{"role": "user", "content": prompt}],
            model=self.cfg.curator_model,
            temperature=self.cfg.curator_temperature,
            return_metadata=True,
            max_tokens=self.cfg.curator_max_tokens + 8000,
        )

        curator_text = resp["message"].strip()

        # IMPORTANT: If you want exact DC behavior, call DC's extract_cheatsheet() here.
        new_cheatsheet = curator_text
        old = self.cheatsheet
        self.cheatsheet = new_cheatsheet

        return {
            "updated": True,
            "old": old,
            "new": new_cheatsheet,
            "curator_output": curator_text,
            "curator_cost_usd": resp.get("total_usd"),
            "timestamp": date_str(),
            # Keep optional meta slot if your logger expects it
            "curator_meta": {k: v for k, v in resp.items() if k != "message"},
        }
