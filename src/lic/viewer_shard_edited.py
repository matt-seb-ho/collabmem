import json
from collections import defaultdict
from datetime import datetime

import streamlit as st


# -------------------------
# Utilities
# -------------------------
def format_timestamp(timestamp_str):
    if not timestamp_str:
        return "No timestamp"
    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%H:%M:%S")
    except ValueError:
        return "Invalid timestamp"


def load_conversations_from_uploaded_jsonl(uploaded_file):
    """
    uploaded_file: streamlit UploadedFile
    returns list[dict]
    """
    conversations = []
    raw = uploaded_file.getvalue().decode("utf-8", errors="replace").splitlines()
    for line in raw:
        line = line.strip()
        if not line:
            continue
        try:
            conversations.append(json.loads(line))
        except json.JSONDecodeError:
            # Skip malformed lines rather than failing entire viewer.
            continue
    return conversations


def load_dataset_from_uploaded_json(uploaded_file):
    """
    Dataset file is the original JSON (list of samples).
    We index by task_id for quick lookup.
    """
    data = json.loads(uploaded_file.getvalue().decode("utf-8", errors="replace"))
    by_task_id = {str(s.get("task_id")): s for s in data}
    return by_task_id


def group_by_field(items, field, default="(missing)"):
    grouped = defaultdict(list)
    for it in items:
        grouped[str(it.get(field, default))].append(it)
    return grouped


def safe_get(d, path, default=None):
    """
    path: list of keys
    """
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# -------------------------
# UI rendering
# -------------------------
def render_trace(conversation, dataset_by_task_id=None):
    conv_id = conversation.get("conv_id", "(missing)")
    st.title(f"Conversation Viewer ({conv_id})")

    # Sidebar metadata
    st.sidebar.header("Conversation Info")
    st.sidebar.write(
        f"Task: {conversation.get('task', conversation.get('task_name', 'N/A'))}"
    )
    st.sidebar.write(f"Task ID: {conversation.get('task_id', 'N/A')}")
    st.sidebar.write(
        f"Sample ID: {conversation.get('sample_id', conversation.get('task_id', 'N/A'))}"
    )
    st.sidebar.write(f"Conv Type: {conversation.get('conv_type', 'N/A')}")
    st.sidebar.write(f"Assistant Model: {conversation.get('assistant_model', 'N/A')}")
    st.sidebar.write(f"System Model: {conversation.get('system_model', 'N/A')}")
    st.sidebar.write(f"User Model: {conversation.get('user_model', 'N/A')}")
    st.sidebar.write(
        f"Solved / is_correct: {conversation.get('is_correct', conversation.get('solved', 'N/A'))}"
    )
    st.sidebar.write(f"Score: {conversation.get('score', 'N/A')}")

    trace = conversation.get("trace", [])

    # Render trace
    for turn in trace:
        role = turn.get("role", "")
        timestamp = format_timestamp(turn.get("timestamp", ""))

        if role == "system":
            # Usually the initial task prompt; show collapsed
            msg = turn.get("content", "")
            with st.expander(f"System message ({timestamp})", expanded=False):
                st.markdown(msg)

        elif role == "user":
            msg = turn.get("content", "")
            st.chat_message("user").write(f"{msg}\n\n*{timestamp}*")

        elif role == "assistant":
            msg = turn.get("content", "")
            st.chat_message("assistant").write(f"{msg}\n\n*{timestamp}*")

        elif role == "log":
            content = turn.get("content", {})
            log_type = content.get("type", "")

            # --- New: context editor logs ---
            if log_type == "context-editor":
                edited_state = content.get("edited_state", {})
                clean_context = edited_state.get("clean_context", "")

                discarded = edited_state.get("discarded_assumptions", [])
                open_q = edited_state.get("open_questions", [])
                facts = edited_state.get("confirmed_facts", [])
                constraints = edited_state.get("constraints", [])

                st.chat_message("system").write(
                    f"🧹 Context editor applied\n\n*{timestamp}*"
                )

                with st.expander("Edited state (details)", expanded=False):
                    if clean_context:
                        st.subheader("clean_context")
                        st.markdown(clean_context)

                    cols = st.columns(2)
                    with cols[0]:
                        st.subheader("confirmed_facts")
                        if facts:
                            for x in facts:
                                st.markdown(f"- {x}")
                        else:
                            st.caption("(none)")

                        st.subheader("constraints")
                        if constraints:
                            for x in constraints:
                                st.markdown(f"- {x}")
                        else:
                            st.caption("(none)")

                    with cols[1]:
                        st.subheader("open_questions")
                        if open_q:
                            for x in open_q:
                                st.markdown(f"- {x}")
                        else:
                            st.caption("(none)")

                        st.subheader("discarded_assumptions")
                        if discarded:
                            for x in discarded:
                                st.markdown(f"- {x}")
                        else:
                            st.caption("(none)")

                    # Optional: show what assistant actually saw (preview)
                    assistant_input_preview = content.get("assistant_input_preview")
                    if assistant_input_preview:
                        with st.expander(
                            "Assistant input preview (raw messages)", expanded=False
                        ):
                            st.json(assistant_input_preview)

            # --- Existing: system verification ---
            elif log_type == "system-verification":
                resp_type = safe_get(
                    content, ["response", "response_type"], default="(missing)"
                )
                st.chat_message("system").write(
                    f"🔎 System verification: **{resp_type}**\n\n*{timestamp}*"
                )

            # --- Existing: answer evaluation ---
            elif log_type == "answer-evaluation":
                exact_answer = content.get("exact_answer", "")
                is_correct = content.get("is_correct", None)
                score = content.get("score", None)

                block = f"```\n{exact_answer}\n```"

                if isinstance(is_correct, (bool, int)):
                    st.chat_message("system").write(
                        f"{block}\n\nAnswer evaluation: {'✅ Correct' if is_correct else '❌ Incorrect'}\n\n*{timestamp}*"
                    )
                elif isinstance(score, float):
                    st.chat_message("system").write(
                        f"{block}\n\nAnswer evaluation score: **{score}**\n\n*{timestamp}*"
                    )
                else:
                    st.chat_message("system").write(
                        f"{block}\n\nAnswer evaluation logged.\n\n*{timestamp}*"
                    )

                # Optionally show full evaluation payload
                with st.expander("Evaluation payload", expanded=False):
                    st.json(content.get("evaluation_return", {}))

            # --- Shard reveal ---
            elif log_type == "shard_revealed":
                shard_id = content.get("shard_id", None)
                st.chat_message("system").write(
                    f"🧩 Shard revealed: **{shard_id}**\n\n*{timestamp}*"
                )

            else:
                # Unknown log types: show compact
                with st.expander(
                    f"Log: {log_type or '(unknown)'} ({timestamp})", expanded=False
                ):
                    st.json(content)

    # Reference answer / sample (optional)
    task_id = str(conversation.get("task_id", ""))
    if dataset_by_task_id and task_id in dataset_by_task_id:
        sample = dataset_by_task_id[task_id]
        with st.expander("Reference sample (from dataset)", expanded=False):
            st.json(sample)

        # If sample has "Answer", show it prominently
        if "Answer" in sample:
            st.chat_message("system").write(f"📌 Reference Answer: {sample['Answer']}")


def main():
    st.set_page_config(page_title="LiC Sharded-Edited Viewer", layout="wide")

    st.sidebar.title("LiC Viewer (Sharded + Context Editor)")

    uploaded_logs = st.sidebar.file_uploader(
        "Upload one or more .jsonl log files",
        type=["jsonl"],
        accept_multiple_files=True,
    )

    uploaded_dataset = st.sidebar.file_uploader(
        "Optional: Upload dataset JSON (e.g., sharded_instructions_600.json) for reference samples",
        type=["json"],
        accept_multiple_files=False,
    )

    if not uploaded_logs:
        st.info("Upload at least one .jsonl log file to begin.")
        return

    # Load conversations
    conversations = []
    for f in uploaded_logs:
        conversations.extend(load_conversations_from_uploaded_jsonl(f))

    if not conversations:
        st.error("No valid conversations found in uploaded files.")
        return

    # Load dataset if provided
    dataset_by_task_id = None
    if uploaded_dataset is not None:
        try:
            dataset_by_task_id = load_dataset_from_uploaded_json(uploaded_dataset)
        except Exception as e:
            st.sidebar.error(f"Failed to parse dataset JSON: {e}")
            dataset_by_task_id = None

    # Build filter options
    conv_types = sorted({str(c.get("conv_type", "(missing)")) for c in conversations})
    assistant_models = sorted(
        {str(c.get("assistant_model", "(missing)")) for c in conversations}
    )
    tasks = sorted(
        {str(c.get("task", c.get("task_name", "(missing)"))) for c in conversations}
    )

    # Sidebar filters
    st.sidebar.header("Filters")

    selected_conv_type = st.sidebar.selectbox(
        "conv_type", options=["(any)"] + conv_types
    )
    selected_task = st.sidebar.selectbox("task", options=["(any)"] + tasks)
    selected_model = st.sidebar.selectbox(
        "assistant_model", options=["(any)"] + assistant_models
    )

    filter_incorrect = st.sidebar.checkbox(
        "Show only incorrect (is_correct == False)", value=False
    )

    # Apply filters
    filtered = conversations
    if selected_conv_type != "(any)":
        filtered = [
            c
            for c in filtered
            if str(c.get("conv_type", "(missing)")) == selected_conv_type
        ]
    if selected_task != "(any)":
        filtered = [
            c
            for c in filtered
            if str(c.get("task", c.get("task_name", "(missing)"))) == selected_task
        ]
    if selected_model != "(any)":
        filtered = [
            c
            for c in filtered
            if str(c.get("assistant_model", "(missing)")) == selected_model
        ]
    if filter_incorrect:
        filtered = [c for c in filtered if c.get("is_correct", True) is False]

    st.sidebar.write(f"Matching conversations: **{len(filtered)}**")

    if not filtered:
        st.warning("No conversations match the selected filters.")
        return

    # Group by task_id then choose
    grouped_by_task_id = group_by_field(filtered, "task_id", default="(missing)")
    task_id_options = sorted(grouped_by_task_id.keys())
    selected_task_id = st.sidebar.selectbox("task_id", options=task_id_options)

    candidates = grouped_by_task_id[selected_task_id]
    conv_id_options = sorted({str(c.get("conv_id", "(missing)")) for c in candidates})
    selected_conv_id = st.sidebar.selectbox("conv_id", options=conv_id_options)

    # Pick conversation
    selected = [
        c for c in candidates if str(c.get("conv_id", "(missing)")) == selected_conv_id
    ]
    if not selected:
        st.error("Selected conversation not found (unexpected).")
        return

    conversation = selected[0]
    render_trace(conversation, dataset_by_task_id=dataset_by_task_id)


if __name__ == "__main__":
    main()
