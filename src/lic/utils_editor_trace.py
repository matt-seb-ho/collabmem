# lic/utils_editor_trace.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def extract_editor_triplets(
    simulation_trace: List[Dict[str, Any]],
    *,
    to_str: bool = False,
    include_system: bool = False,
    clean_context_key: str = "clean_context",
    max_clean_context_chars: int = 16000,
    max_msg_chars: int = 16000,
) -> Any:
    """
    Extract a compact transcript for editor reflection:

      [user] ...
      [edited_context] ...   # ONLY edited_state[clean_context_key] from the context-editor log
      [assistant] ...

    Repeats for each assistant turn.

    Assumptions:
    - The simulator logs a 'role=log' message with content.type == 'context-editor'
      *before* the assistant message for that turn.
    - That log contains content.edited_state, which contains 'clean_context'.

    If an editor log is missing for a user->assistant turn (e.g., editor disabled or failed),
    edited_context will be "(missing)".
    """
    # Optionally capture first system message for context
    system_msg = None
    if include_system:
        for m in simulation_trace:
            if m.get("role") == "system":
                system_msg = m.get("content", None)
                break

    triplets: List[Dict[str, str]] = []
    pending_user: Optional[str] = None
    pending_clean_context: Optional[str] = None

    def _truncate(s: str, n: int) -> str:
        if s is None:
            return ""
        s = str(s)
        return s if len(s) <= n else (s[:n] + "...(truncated)")

    for msg in simulation_trace:
        role = msg.get("role")

        if role == "user":
            pending_user = _truncate(msg.get("content", ""), max_msg_chars)
            pending_clean_context = None  # reset for this turn
            continue

        if role == "log":
            c = msg.get("content", {}) or {}
            if c.get("type") == "context-editor":
                edited_state = c.get("edited_state", {}) or {}
                clean = edited_state.get(clean_context_key, "")
                pending_clean_context = _truncate(clean, max_clean_context_chars)
            continue

        if role == "assistant":
            # We form a triplet when we see the assistant message following a user turn.
            if pending_user is None:
                # Could be a preamble assistant message; ignore.
                continue

            assistant_text = _truncate(msg.get("content", ""), max_msg_chars)
            clean = (
                pending_clean_context
                if pending_clean_context is not None
                else "(missing)"
            )

            triplets.append(
                {
                    "user": pending_user,
                    "edited_context": clean,
                    "assistant": assistant_text,
                }
            )

            # reset for next turn
            pending_user = None
            pending_clean_context = None

    if not to_str:
        out: Dict[str, Any] = {"turns": triplets}
        if include_system and system_msg is not None:
            out["system"] = system_msg
        return out

    # to_str formatting
    blocks: List[str] = []
    if include_system and system_msg is not None:
        blocks.append(f"[system] {system_msg}")

    for i, t in enumerate(triplets, start=1):
        blocks.append(
            "\n\n".join(
                [
                    f"=== TURN {i} ===",
                    f"[user] {t['user']}",
                    f"[edited_context] {t['edited_context']}",
                    f"[assistant] {t['assistant']}",
                ]
            )
        )

    return "\n\n".join(blocks)
