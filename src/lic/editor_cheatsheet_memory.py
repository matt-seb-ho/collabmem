# lic/editor_cheatsheet_memory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lic.cheatsheet_memory import CheatsheetConfig, CheatsheetMemory
from lic.utils import extract_conversation


class EditorCheatsheetMemory(CheatsheetMemory):
    """
    Same mechanics as CheatsheetMemory, but the update prompt is framed around:
      - the raw conversation
      - the editor's produced state (clean_context + lists)
      - (optional) downstream eval feedback (did the final answer succeed?)
    """

    def _render_update_prompt(
        self,
        task_name: str,
        system_message: str,
        trace: List[Dict[str, Any]],
        eval_summary: Optional[Dict[str, Any]],
        edited_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        conversation_txt = extract_conversation(trace, to_str=True)

        # "model_answer" for editor = what it produced
        model_answer = ""
        if edited_state is not None:
            # keep it compact-ish but still structured
            model_answer = (
                "EDITED_STATE_JSON:\n"
                f"{edited_state}\n\n"
                "CLEAN_CONTEXT:\n"
                f"{edited_state.get('clean_context', '')}"
            )
        else:
            model_answer = "(no edited_state logged)"

        feedback_block = ""
        if self.cfg.include_eval_feedback and eval_summary is not None:
            feedback_block = (
                "\n\nDOWNSTREAM EVALUATION FEEDBACK:\n"
                f"- is_correct: {eval_summary.get('is_correct')}\n"
                f"- score: {eval_summary.get('score')}\n"
                f"- extracted_answer: {eval_summary.get('extracted_answer')}\n"
            )

        question = (
            f"ROLE: CONTEXT_EDITOR\n"
            f"TASK: {task_name}\n\n"
            f"ORIGINAL SYSTEM PROMPT:\n{system_message}\n\n"
            f"CONVERSATION:\n{conversation_txt}"
            f"{feedback_block}"
        )

        return (
            self.cfg.cheatsheet_update_template.replace("[[QUESTION]]", question)
            .replace("[[MODEL_ANSWER]]", model_answer)
            .replace("[[PREVIOUS_CHEATSHEET]]", self.cheatsheet)
        )

    def maybe_update_editor(
        self,
        task_name: str,
        system_message: str,
        trace: List[Dict[str, Any]],
        eval_summary: Optional[Dict[str, Any]],
        edited_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not self.cfg.enable_updates:
            return {"updated": False, "old": self.cheatsheet, "new": self.cheatsheet}

        prompt = self._render_update_prompt(
            task_name=task_name,
            system_message=system_message,
            trace=trace,
            eval_summary=eval_summary,
            edited_state=edited_state,
        )

        # Use the inherited generate() call by delegating to parent logic:
        # (We can't call super().maybe_update because it calls its own _render_update_prompt signature.)
        from lic.model import generate
        from lic.utils import date_str

        resp = generate(
            [{"role": "user", "content": prompt}],
            model=self.cfg.curator_model,
            temperature=self.cfg.curator_temperature,
            return_metadata=True,
            max_tokens=self.cfg.curator_max_tokens + 8192, # TODO: don't hardcode more tokens for reasoning...
        )

        curator_text = resp["message"].strip()
        old = self.cheatsheet
        self.cheatsheet = curator_text

        return {
            "updated": True,
            "old": old,
            "new": self.cheatsheet,
            "curator_output": curator_text,
            "curator_cost_usd": resp.get("total_usd"),
            "timestamp": date_str(),
        }
