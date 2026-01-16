# lic/editor_cheatsheet_memory.py
from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from lic.model import generate
from lic.retry_utils import RetryConfig, retry_call
from lic.utils import date_str, extract_conversation

# DEFAULT_EDITOR_CHEATSHEET_UPDATE_TEMPLATE = """\

# prompts/editor_cheatsheet_update_prompt_v3.py
# Use this as the cheatsheet_update_template for the *context editor* curator.
# It is intentionally NOT overly prescriptive about *how* to edit; it specifies the
# objective and asks the curator to learn/update principles via self-reflection.

# EDITOR_CHEATSHEET_UPDATE_PROMPT_V3 = r"""
DEFAULT_EDITOR_CHEATSHEET_UPDATE_TEMPLATE = """\
# CHEATSHEET REFERENCE CURATOR (CONTEXT EDITOR / LOST-IN-CONVERSATION SETTING)

#### 1. Purpose and Goals
You are the Cheatsheet Curator for a **conversation context editor**.

The context editor is called *before each assistant turn* to produce a cleaned context that helps the assistant perform well when task specifications emerge across multiple turns.

In these settings, performance degrades because assistants may:
- answer before enough information is provided,
- cling to early assumptions even after later corrections,
- overweight prior assistant content relative to new user constraints,
- lose track of evolving requirements.

Your job is to maintain a continuously evolving **CHEATSHEET / PLAYBOOK** that helps the context editor facilitate correct downstream behavior in such multi-turn settings.

After each conversation trajectory, update the cheatsheet by extracting reusable strategies, failure modes, and editing patterns. Prefer generalizable lessons over episode-specific details.

---

#### 2. What “Success” Means for the Context Editor (Objective, Not a Fixed Policy)
The context editor is a *state reconstruction and stabilization* component.

After a multi-turn interaction, there is an implicit task specification that the assistant ideally should understand and act on.

When available, the dataset’s oracle single-turn specification:
[[FULL_TASK_SPEC_SINGLE_TURN]]

represents **what the assistant should have understood as the task after all interactions** (i.e., the fully accumulated and clarified task).

This oracle spec is not necessarily something the editor should copy verbatim, but it is a highly informative reference for reflection:
- Did the edited context support converging to the right understanding of the task?
- What information was missing or misrepresented in the edited context?
- What incorrect assumptions or outdated assistant-derived content should have been removed?
- What user constraints or corrections should have been emphasized?

The editor’s role can be decomposed into prioritized subtasks (use these as a lens during reflection, not as rigid rules):
1) **Consolidate user specifications into a coherent working spec** (accumulate constraints, definitions, formats, requirements)
2) **Remove incorrect / outdated / invalidated assumptions and derivations** (especially assistant-originated hypotheses that conflict with later user info)
3) **(Lower priority) Preserve valid prior assistant progress if it remains consistent**, to improve efficiency without harming correctness

The cheatsheet should help the editor do (1) and (2) reliably, and do (3) only when safe.

---

#### 3. Core Responsibilities (Curator)
As the Cheatsheet Curator, you must:

1) Curate and preserve knowledge
- Select and document only the most relevant, useful, actionable strategies.
- Preserve valuable older entries unless clearly superseded.
- Consolidate redundancies rather than deleting.

IMPORTANT: once you produce the NEW CHEATSHEET, anything not explicitly included will be lost. Copy forward any prior content you still want.

2) Maintain accuracy and scope
- Only include guidance that improves the context editor’s editing behavior.
- Avoid drifting into generic “how to solve tasks” advice (that belongs to the assistant, not the editor).

3) Encourage learning via self-reflection (do NOT hard-code a fixed policy)
- Do not treat any initial principles as immutable.
- Use the trajectory (and oracle spec if provided) to:
  - propose candidate rules,
  - strengthen them when supported by evidence,
  - weaken/scope/remove them when contradicted by evidence,
  - add new rules when novel patterns appear.

4) Track usefulness over time
- Each entry includes a usage count.
- Increment the count when the strategy appears to have helped (e.g., prevents a known trap, improves correctness, improves alignment to the accumulated spec).

---

#### 4. Cheatsheet Structure
Organize the cheatsheet into these sections:

1) EDITING OUTPUT TEMPLATES (How to write clean_context)
- Compact structures that help the assistant: e.g., “Working Spec + Open Questions + Constraints + Output Format”.

2) EDITING HEURISTICS / CHECKLISTS
- Step-by-step procedures the editor can apply before emitting clean_context.

3) FAILURE MODES / INTERACTION TRAPS
- Editor-specific mistakes and how to avoid them (e.g., anchoring to early assistant guesses, missing late constraints, collapsing uncertainty into guesses).

4) [OPTIONAL] TASK-SPECIFIC NOTES
- Only include if it clearly improves spec reconstruction for that task type (focus on spec completeness, not solution tactics).

5) USAGE COUNTERS
- Each memory item includes **Count: N**.

---

#### 5. Formatting Guidelines (Strict)
Use the following structure for each memory item:

<memory_item>
<title>
[Short name of strategy / template / trap-avoidance]
</title>
<description>
- What problem it addresses (editor-side).
- When it applies.
- What to include/exclude in edited context.
- If relevant, link to conversation points (e.g., Conv3-Turn5).
Optional but encouraged:
- Evidence: brief note of what in this episode supports it
- Scope: tasks where it applies (or unknown)
- Confidence: high/medium/low
</description>
<example>
[Short example of a clean_context snippet, rewrite rule, checklist step, or conflict-resolution pattern.]
</example>
</memory_item>
** Count: [N]

Avoid pasting long raw logs; summarize patterns and provide compact examples.

---

#### 6. Cheatsheet Template (Output Requirements)
Return ONLY the NEW CHEATSHEET, formatted exactly as:

<cheatsheet>

Version: [increment version number]

EDITING OUTPUT TEMPLATES
<memory_item> ... </memory_item>

EDITING HEURISTICS / CHECKLISTS
<memory_item> ... </memory_item>

FAILURE MODES / INTERACTION TRAPS
<memory_item> ... </memory_item>

[OPTIONAL] TASK-SPECIFIC NOTES
<memory_item> ... </memory_item>

</cheatsheet>

Target length: ~2000–2500 words.
All content must be inside <cheatsheet>.

-----
-----

## PREVIOUS CHEATSHEET
[[PREVIOUS_CHEATSHEET]]

-----
-----

## CONVERSATION TRAJECTORY (PRIMARY REFLECTION TARGET)
[[CONVERSATION]]

This is a multi-turn interaction; the correct task specification may only become clear by the end.

-----
-----

## OPTIONAL EXTRINSIC GROUNDING (MAY OR MAY NOT BE PRESENT)

(A) EVALUATION LABEL / SUMMARY
[[CORRECTNESS_LABEL]]

(B) SINGLE-TURN FULLY SPECIFIED TASK (WHAT THE ASSISTANT SHOULD HAVE UNDERSTOOD AFTER ALL TURNS)
[[FULL_TASK_SPEC_SINGLE_TURN]]

(C) GROUND TRUTH OUTPUT (IF AVAILABLE)
[[GROUND_TRUTH_OUTPUT]]
"""


@dataclass
class EditorCheatsheetConfig:
    # How the editor sees the cheatsheet each turn (we pass it as `playbook` into editor system messages)
    max_cheatsheet_chars: int = 12000  # safety guard

    # Update behavior
    enable_updates: bool = True
    curator_model: str = "gpt-4o-mini"
    curator_temperature: float = 0.0
    curator_max_tokens: int = 2000

    # Extrinsic grounding toggles (ablatable)
    include_eval_label: bool = False
    include_full_spec_q: bool = True
    include_ground_truth: bool = False

    retry_cfg: RetryConfig = RetryConfig(
        max_attempts=4, base_delay_s=0.8, max_delay_s=10.0, jitter=0.25
    )
    require_cheatsheet_wrapper: bool = (
        True  # if True, require <cheatsheet>...</cheatsheet>
    )
    min_chars: int = 50  # reject trivially short outputs

    # Prompt template for reflection/update
    cheatsheet_update_template: str = DEFAULT_EDITOR_CHEATSHEET_UPDATE_TEMPLATE


class EditorCheatsheetMemory:
    """
    Stores a persistent cheatsheet for the *context editor*.
    Driver calls maybe_update(...) after each episode; simulator passes build_editor_playbook()
    into edit_conversation_state(... playbook=...).
    """

    def __init__(self, initial_cheatsheet: str, cfg: EditorCheatsheetConfig):
        self.cheatsheet = initial_cheatsheet or "(empty)"
        self.cfg = cfg

    def build_editor_playbook(self) -> str:
        cs = self.cheatsheet
        if len(cs) > self.cfg.max_cheatsheet_chars:
            cs = cs[-self.cfg.max_cheatsheet_chars :]
        # Keep it as plain text; context_editor.py wraps it inside a system message.
        return cs

    @staticmethod
    def _safe_str(x: Any) -> str:
        if x is None:
            return "(not provided)"
        if isinstance(x, str):
            return x
        try:
            return str(x)
        except Exception:
            return repr(x)

    @staticmethod
    def _replace_if_present(template: str, placeholder: str, value: str) -> str:
        return (
            template.replace(placeholder, value)
            if placeholder in template
            else template
        )

    def _render_update_prompt(
        self,
        task_name: str,
        system_message: str,
        trace: List[Dict[str, Any]],
        eval_summary: Optional[Dict[str, Any]],
        full_spec_q: Optional[str],
        ground_truth_a: Any,
    ) -> str:
        conversation_txt = extract_conversation(trace, to_str=True)

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

        tmpl = self.cfg.cheatsheet_update_template

        tmpl = tmpl.replace("[[PREVIOUS_CHEATSHEET]]", self.cheatsheet)
        tmpl = tmpl.replace("[[TASK_NAME]]", task_name)
        tmpl = tmpl.replace("[[SYSTEM_MESSAGE]]", system_message)
        tmpl = tmpl.replace("[[CONVERSATION]]", conversation_txt)

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
        # frozen / disabled updates
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

        curator_attempts = 0

        def _call_curator():
            return generate(
                [{"role": "user", "content": prompt}],
                model=self.cfg.curator_model,
                temperature=self.cfg.curator_temperature,
                return_metadata=True,
                max_tokens=self.cfg.curator_max_tokens,
            )

        try:

            def _call_with_count():
                nonlocal curator_attempts
                curator_attempts += 1
                return _call_curator()

            resp = retry_call(
                _call_with_count,
                cfg=self.cfg.retry_cfg,
                retry_on=(TimeoutError, ConnectionError),
                retry_if=lambda e: "timeout" in repr(e).lower()
                or "rate" in repr(e).lower(),
            )

            curator_text = (resp.get("message") or "").strip()

            # Validate output; if invalid, do not update
            if not self._looks_like_valid_cheatsheet(
                curator_text,
                require_wrapper=self.cfg.require_cheatsheet_wrapper,
                min_chars=self.cfg.min_chars,
            ):
                return {
                    "updated": False,
                    "old": self.cheatsheet,
                    "new": self.cheatsheet,
                    "error": "invalid_curator_output",
                    "curator_attempts": curator_attempts,
                    "curator_output_preview": curator_text[:500],
                    "timestamp": date_str(),
                    "curator_meta": {k: v for k, v in resp.items() if k != "message"},
                }

            old = self.cheatsheet
            self.cheatsheet = curator_text

            return {
                "updated": True,
                "old": old,
                "new": self.cheatsheet,
                "curator_output": curator_text,
                "curator_cost_usd": resp.get("total_usd"),
                "curator_attempts": curator_attempts,
                "timestamp": date_str(),
                "curator_meta": {k: v for k, v in resp.items() if k != "message"},
            }

        except Exception as e:
            # Failure should not crash training/eval; keep old cheatsheet
            return {
                "updated": False,
                "old": self.cheatsheet,
                "new": self.cheatsheet,
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "curator_attempts": curator_attempts,
                "timestamp": date_str(),
            }

    @staticmethod
    def _looks_like_valid_cheatsheet(
        text: str, require_wrapper: bool, min_chars: int
    ) -> bool:
        if not text:
            return False
        t = text.strip()
        if len(t) < min_chars:
            return False
        if require_wrapper and ("<cheatsheet>" not in t or "</cheatsheet>" not in t):
            return False
        return True
