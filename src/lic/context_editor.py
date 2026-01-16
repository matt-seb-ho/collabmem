# lic/context_editor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from lic.model import generate
from lic.utils import extract_conversation

BASE_EDITOR_SYSTEM_PROMPT = """\
You are a conversation state editor.

## Background
In realistic interactions, users often specify tasks across multiple turns. They may add details, revise constraints, or correct earlier statements. Assistants sometimes respond prematurely or make assumptions that later turn out to be incorrect or outdated. These early assistant messages can interfere with later turns by biasing the conversation or causing new user information to be overlooked.

## Task
Given the conversation so far, produce a cleaned context for the *next* assistant turn that supports correct task completion.

The cleaned context should reflect the relevant conversation history, emphasize user specifications, and reduce the influence of incorrect, outdated, or misleading prior assistant content.

## Output
Return valid JSON with the following keys:
```json
{
  "scratch": string,
  "clean_context": string
}
```
- scratch: your scratchpad for reasoning through the task-- brief internal reasoning about what information should be carried forward and what should not.
- clean_context: a compact working context for the next assistant turn.
"""


@dataclass(frozen=True)
class EditorSchema:
    """
    Allows the playbook/reflection to evolve what the editor returns WITHOUT changing
    the base system prompt.

    - required_keys: keys we will insist exist in the JSON
    - extra_keys_hint: optional guidance text asking for additional keys/structure
    """

    required_keys: Tuple[str, ...] = ("scratch", "clean_context")
    extra_keys_hint: Optional[str] = None

    def system_message(self) -> str:
        msg = (
            "Schema instruction for the editor:\n"
            f"- Required JSON keys: {list(self.required_keys)}\n"
            "- Output MUST be valid JSON.\n"
        )
        if self.extra_keys_hint:
            msg += f"- Optional keys requested (if helpful): {self.extra_keys_hint}\n"
        return msg


def _safe_json_loads(text: str) -> Dict[str, Any]:
    import json

    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError(f"Context editor did not return JSON. Raw:\n{text}")


def _compose_editor_messages(
    messages: List[Dict[str, Any]],
    schema: EditorSchema,
    playbook: Optional[str] = None,
    reflection: Optional[str] = None,
    extra_instructions: Optional[str] = None,
) -> List[Dict[str, str]]:
    system_msgs: List[Dict[str, str]] = [
        {"role": "system", "content": BASE_EDITOR_SYSTEM_PROMPT},
        # Critical: schema is a *separate* system message, easy to evolve over time.
        {"role": "system", "content": schema.system_message()},
    ]

    if playbook:
        system_msgs.append(
            {
                "role": "system",
                "content": (
                    "Editor playbook (may revise how you edit and/or what extra keys you include):\n"
                    f"{playbook}"
                ),
            }
        )

    if reflection:
        system_msgs.append(
            {
                "role": "system",
                "content": (
                    "Editor reflection / lessons learned (apply if useful):\n"
                    f"{reflection}"
                ),
            }
        )

    if extra_instructions:
        system_msgs.append(
            {"role": "system", "content": f"Additional guidance:\n{extra_instructions}"}
        )

    user_msg = {
        "role": "user",
        "content": (
            "Conversation so far (JSON-like message list):\n\n"
            f"{messages}\n\n"
            "Return the edited state JSON now."
        ),
    }

    return system_msgs + [user_msg]


def edit_conversation_state(
    trace: Any,
    editor_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 1200,
    # Evolving knobs:
    schema: Optional[EditorSchema] = None,
    playbook: Optional[str] = None,
    reflection: Optional[str] = None,
    extra_instructions: Optional[str] = None,
    response_format_json: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      edited_state (dict): parsed JSON from the editor model
      editor_obj (dict): raw generate() return_metadata object

    The base prompt stays minimal. Behavior/structure can be updated via:
      - schema (required keys, optional hints)
      - playbook/reflection messages (how to edit; optional additional keys)
    """
    if schema is None:
        schema = EditorSchema()

    messages = extract_conversation(trace, to_str=False)

    editor_messages = _compose_editor_messages(
        messages=messages,
        schema=schema,
        playbook=playbook,
        reflection=reflection,
        extra_instructions=extra_instructions,
    )

    gen_kwargs: Dict[str, Any] = dict(
        model=editor_model,
        temperature=temperature,
        return_metadata=True,
        max_tokens=max_tokens,
    )
    if response_format_json:
        gen_kwargs["response_format"] = {"type": "json_object"}

    editor_obj = generate(editor_messages, **gen_kwargs)
    text = editor_obj["message"]

    edited_state = _safe_json_loads(text)

    # Enforce minimal contract: required keys must exist (fill missing with "")
    for k in schema.required_keys:
        if k not in edited_state:
            edited_state[k] = ""

    return edited_state, editor_obj


def build_assistant_input_from_edited_state(
    original_system_message: str,
    last_user_message: str,
    edited_state: Dict[str, Any],
    # Default: only expose clean_context to downstream assistant.
    # expose_scratch: bool = False,
) -> List[Dict[str, str]]:
    clean_context = edited_state.get("clean_context", "")

    # In most setups you do NOT want to show scratch to the assistant.
    # if expose_scratch:
    #     scratch = edited_state.get("scratch", "")
    #     edited_system = (
    #         "Edited conversation state (produced by a context editor).\n\n"
    #         "Scratch (editor planning):\n"
    #         f"{scratch}\n\n"
    #         "Clean context (use this):\n"
    #         f"{clean_context}"
    #     )
    # else:
    #     edited_system = (
    #         "Edited conversation state (produced by a context editor). "
    #         "Use it as a bias-reduced snapshot of what the user has provided.\n\n"
    #         f"{clean_context}"
    #     )

    edited_system = f"Current Conversation State (summarized from previous interactions in this session):\n\n{clean_context}"

    return [
        {"role": "system", "content": original_system_message},
        {"role": "system", "content": edited_system},
        {"role": "user", "content": last_user_message},
    ]
