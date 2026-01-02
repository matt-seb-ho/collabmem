# lic/cheatsheet_memory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lic.model import generate
from lic.utils import date_str, extract_conversation
from lic.constants import PROMPT_FILE_PATHS


VANILLA_UPDATE_PROMPT = PROMPT_FILE_PATHS["dc_cu_vanilla"].read_text()
LIC_UPDATE_PROMPT = PROMPT_FILE_PATHS["dc_cu_lic"].read_text


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

    # Whether to include extrinsic feedback in reflection prompt
    include_eval_feedback: bool = False

    # Prompt templates (you said you have them from the DC repo)
    # cheatsheet_update_template: str = ""  # must include placeholders below
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

    def _render_update_prompt(
        self,
        task_name: str,
        system_message: str,
        trace: List[Dict[str, Any]],
        eval_summary: Optional[Dict[str, Any]],
    ) -> str:
        # Build a compact “question” and “model_answer” representation
        # Use the conversation itself as the “interaction” to learn from.
        conversation_txt = extract_conversation(trace, to_str=True)

        model_answer = ""
        # last assistant msg (if present) is usually best “answer attempt”
        for msg in reversed(trace):
            if msg["role"] == "assistant":
                model_answer = msg["content"]
                break

        feedback_block = ""
        if self.cfg.include_eval_feedback and eval_summary is not None:
            feedback_block = (
                "\n\nEVALUATION FEEDBACK:\n"
                f"- is_correct: {eval_summary.get('is_correct')}\n"
                f"- score: {eval_summary.get('score')}\n"
                f"- extracted_answer: {eval_summary.get('extracted_answer')}\n"
            )

        # Required placeholders:
        # [[QUESTION]], [[MODEL_ANSWER]], [[PREVIOUS_CHEATSHEET]]
        # (You can put whatever you want in [[QUESTION]]; here it’s “episode context”.)
        question = (
            f"TASK: {task_name}\n\n"
            f"SYSTEM PROMPT:\n{system_message}\n\n"
            f"CONVERSATION:\n{conversation_txt}"
            f"{feedback_block}"
        )

        return (
            self.cfg.cheatsheet_update_template.replace("[[QUESTION]]", question)
            .replace("[[MODEL_ANSWER]]", model_answer)
            .replace("[[PREVIOUS_CHEATSHEET]]", self.cheatsheet)
        )

    def maybe_update(
        self,
        task_name: str,
        system_message: str,
        trace: List[Dict[str, Any]],
        eval_summary: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not self.cfg.enable_updates:
            return {"updated": False, "old": self.cheatsheet, "new": self.cheatsheet}

        prompt = self._render_update_prompt(
            task_name, system_message, trace, eval_summary
        )

        resp = generate(
            [{"role": "user", "content": prompt}],
            model=self.cfg.curator_model,
            temperature=self.cfg.curator_temperature,
            return_metadata=True,
            max_tokens=self.cfg.curator_max_tokens,
        )

        curator_text = resp["message"]

        # IMPORTANT: use the same extraction logic as DC if you want exact behavior.
        # If you already have DC's extract_cheatsheet(), import and call it here.
        new_cheatsheet = curator_text.strip()
        old = self.cheatsheet
        self.cheatsheet = new_cheatsheet

        return {
            "updated": True,
            "old": old,
            "new": new_cheatsheet,
            "curator_output": curator_text,
            "curator_cost_usd": resp.get("total_usd"),
            "timestamp": date_str(),
        }
