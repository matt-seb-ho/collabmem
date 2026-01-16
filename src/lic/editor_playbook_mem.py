# lic/editor_cheatsheet_memory.py
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lic.model import generate
from lic.retry_utils import RetryConfig, retry_call
from lic.utils import date_str
from lic.utils_editor_trace import extract_editor_triplets

# DEFAULT_EDITOR_CHEATSHEET_UPDATE_TEMPLATE = """\

# prompts/editor_cheatsheet_update_prompt_v3.py
# Use this as the cheatsheet_update_template for the *context editor* curator.
# It is intentionally NOT overly prescriptive about *how* to edit; it specifies the
# objective and asks the curator to learn/update principles via self-reflection.

# v3
DEFAULT_EDITOR_CHEATSHEET_UPDATE_TEMPLATE = """\
# CHEATSHEET / PLAYBOOK CURATOR INSTRUCTIONS

- You are the curator of a continuously evolving **cheatsheet / playbook** for a *conversation context editor*.
- The context editor is invoked before each assistant turn and is responsible for producing a cleaned context that helps the assistant operate correctly when task specifications are revealed gradually across multiple turns.
- Multi-turn conversations are prone to "lost in conversation" failures, where early or incorrect content (e.g. early assumptions, premature solutions) interferes with later understanding.
- Your job is to extract **reusable editor-side strategies, pitfalls, and patterns** that help reconstruct the user’s task specification faithfully and suppress misleading conversational artifacts.

---

## 1. What the Context Editor Is Optimizing For (Priority Order)

The editor’s role is **not** to solve the task.

Its role is to maintain the best possible *representation of the user’s task specification so far*, under partial and evolving information.

The editor’s priorities are:

1) **Reconstruct the task specification from user turns**
   - Accumulate constraints, definitions, requirements, formats, and revisions.
   - Maintain a coherent, up-to-date picture of "what the user is asking for."
   - Explicitly track what is still unknown or underspecified.

2) **Suppress interference from misleading prior content**
   - Remove or demote incorrect, outdated, or speculative assumptions.
   - Remove results derived from outdated information/assumptions.
   - Especially guard against assistant-originated hypotheses that are later contradicted or never confirmed by the user.

3) **Optionally preserve prior assistant progress**
   - Only when it is clearly consistent with the reconstructed specification.
   - From an efficiency standpoint, we prefer to keep prior work if it is safe to do so, but this is strictly secondary to ensuring clean context/mitigating interference.
   - If interference suppression (2) conflicts with progress salvage, **drop salvage**.
     False positives here are costly because they anchor the assistant incorrectly.

---

## 2. What to Reflect On (Most Important Signal)

### Full Task Specification

In realistic situations, users often specify tasks gradually across multiple turns (adding details, revising constraints, correcting earlier statements, etc.).

In this exercise, we have access to the full task specification
This represents:
> The complete task specification that the context editor *should have inferred*
> after seeing all user turns in the multi-turn interaction.

This full task spec is the **main reflection target**.

Use it to ask:
- What information from the conversation was essential to recover this spec?
- What was missing, diluted, or distorted in the edited context?
- What earlier assumptions or assistant-derived content should have been removed?
- What editing behavior would have caused the final clean_context to closely approximate this spec?

**Important caveat:**
- The full task spec is the *ideal target only at the end* of the conversation, once all shards have been revealed.
- Earlier context edits are intermediate steps:
  - they should preserve known specs,
  - mark unknowns clearly,
  - and steer the assistant toward asking for or preparing for missing information.

Your cheatsheet should therefore encode strategies that help the editor *progressively converge* toward this target across turns.

---

## 3. Scope Discipline

Only include guidance that improves **context editing behavior**.

Good content:
- how to aggregate user constraints across turns
- how to represent partial specs and open questions
- how to detect and remove stale or invalid assumptions
- how to decide whether assistant progress is safe to keep
- patterns for writing clean_context that emphasize user intent

Do NOT include:
- advice on how the assistant should solve the task
- domain-specific solution strategies or reasoning steps
- “the answer is likely X”-style guidance

If a pattern is task-type-specific, include it only if it affects
**spec reconstruction**, not task execution.

---

## 4. Cheatsheet Structure (Deliberately Flexible)

Organize content in whatever structure best preserves useful editor knowledge.
Common (but optional) groupings include:
- Editing moves / heuristics
- Rewrite or consolidation patterns
- Failure modes / traps
- Gates/decision rules for keeping vs dropping assistant content
- Minimal before→after examples

---

## 5. Memory Item Guidelines (Lightweight, Not Rigid)

We recommend keeping organizing the cheatsheet into "memory items" or bullet points.
- Each memory item should capture **one reusable lesson** about context editing.
- You can rewrite, revise, recombine or reorganize items however you like.
- The benefit of itemization is that we can track "usage counts" for each item over time, helping identify which lessons are most impactful.

You may include any of the following when helpful:
- what situation triggers it
- what the editor should do
- what to avoid
- why it matters (brief evidence or rationale)
- a compact example or rewrite pattern

Do NOT force all fields to appear in every item.

Each item must include:
- a clear title
- an **inline usage count**, incremented when supported by this episode

Other fields (e.g. rationale, example, when to use) are optional.

We recommend using yaml formatting, e.g.:
```
- title: "Explicitly track unknowns"
  usage_count: 5
  when_to_use: "When the user has not yet specified a key detail"
```

---

## 6. Output Requirements

Return ONLY the updated cheatsheet, wrapped exactly as:

<cheatsheet>
version: [increment]

[cheatsheet contents]

</cheatsheet>

Anything not included will be lost. Copy forward prior material you still want.

Target length: flexible. Prefer concise, high-signal entries.

---

## 7. Inputs (Labeled for Clarity)

### PREVIOUS CHEATSHEET
[[PREVIOUS_CHEATSHEET]]

### CONVERSATION TRAJECTORY
(Multi-turn interaction from which the editor must reconstruct the task.)
[[CONVERSATION]]

### FULL TASK SPECIFICATION (PRIMARY REFLECTION TARGET)
[[FULL_TASK_SPEC_SINGLE_TURN]]

### OPTIONAL AUXILIARY SIGNALS
- Evaluation label / summary: [[CORRECTNESS_LABEL]]
- Ground truth output (if available): [[GROUND_TRUTH_OUTPUT]]
"""


@dataclass
class EditorCheatsheetConfig:
    # How the editor sees the cheatsheet each turn (we pass it as `playbook` into editor system messages)
    max_cheatsheet_chars: int = 16000  # safety guard

    # Update behavior
    enable_updates: bool = True
    curator_model: str = "gpt-4o-mini"
    curator_temperature: float = 0.0
    curator_max_tokens: int = 2000

    # Extrinsic grounding toggles (ablatable)
    include_eval_label: bool = False
    include_full_spec_q: bool = True
    include_ground_truth: bool = False

    # retry_cfg: RetryConfig = RetryConfig(
    #     max_attempts=4, base_delay_s=0.8, max_delay_s=10.0, jitter=0.25
    # )
    retry_cfg: RetryConfig = field(
        default_factory=lambda: RetryConfig(
            max_attempts=4, base_delay_s=0.8, max_delay_s=10.0, jitter=0.25
        )
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
        conversation_txt = extract_editor_triplets(trace, to_str=True)

        # Optional auxiliary signals
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

        # Required in v3 template
        tmpl = tmpl.replace("[[PREVIOUS_CHEATSHEET]]", self.cheatsheet)
        tmpl = tmpl.replace("[[CONVERSATION]]", conversation_txt)

        # Robust to template drift: only replace if present
        tmpl = self._replace_if_present(
            tmpl, "[[TASK_NAME]]", self._safe_str(task_name)
        )
        tmpl = self._replace_if_present(
            tmpl, "[[SYSTEM_MESSAGE]]", self._safe_str(system_message)
        )
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
