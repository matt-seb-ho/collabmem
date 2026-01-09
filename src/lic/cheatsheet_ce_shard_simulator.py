# lic/cheatsheet_ce_shard_simulator.py
# from lic.context_editor import (
#     build_assistant_input_from_edited_state,
#     edit_conversation_state,
# )
from lic.cheatsheet_context_editor import (
    build_assistant_input_from_edited_state,
    edit_conversation_state,
)
from lic.model import generate, is_reasoning_model
from lic.system_agent import SystemAgent
from lic.tasks import get_task
from lic.user_agent import UserAgent
from lic.utils import date_str, extract_conversation, print_colored
from lic.utils_log import log_conversation

FINALIZATION_USER_MESSAGE = (
    "That's all the information I have. Please provide your final answer now."
)


class ConversationSimulatorShardedEdited:
    def __init__(
        self,
        sample,
        assistant_model="gpt-4o-mini",
        system_model="gpt-4o-mini",
        user_model="gpt-4o-mini",
        assistant_temperature=1.0,
        user_temperature=1.0,
        dataset_fn=None,
        log_folder="logs",
        reasoning_cls_override=None,
        # editor params
        editor_model="gpt-4o-mini",
        editor_temperature=0.0,
        editor_max_tokens=1200,
        enable_editor=True,
        log_editor_artifacts=True,
        # finalization behavior
        enable_finalization_nudge=True,
        # NEW: cheatsheet preambles
        assistant_preamble: str | None = None,
        editor_preamble: str | None = None,
    ):
        self.task_name = sample["task"]
        self.task = get_task(self.task_name)
        self.dataset_fn = dataset_fn
        self.sample = sample

        self.assistant_model = assistant_model
        self.system_model = system_model
        self.user_model = user_model

        self.user_agent = UserAgent(self.task, user_model)
        self.system_agent = SystemAgent(self.task_name, system_model, self.sample)

        self.log_folder = log_folder
        self.system_message = self.task.generate_system_prompt(self.sample)
        self.answer_description = self.task.get_answer_description()

        self.run_with_custom_temperature = (
            assistant_temperature != 1.0 or user_temperature != 1.0
        )
        self.assistant_temperature = assistant_temperature
        self.user_temperature = user_temperature

        self.reasoning_cls_override = reasoning_cls_override

        # editor knobs
        self.editor_model = editor_model
        self.editor_temperature = editor_temperature
        self.editor_max_tokens = editor_max_tokens
        self.enable_editor = enable_editor
        self.log_editor_artifacts = log_editor_artifacts

        # finalization nudge
        self.enable_finalization_nudge = enable_finalization_nudge
        self._finalization_prompted = False

        # NEW
        self.assistant_preamble = assistant_preamble
        self.editor_preamble = editor_preamble

        self.trace = [
            {"role": "system", "content": self.system_message, "timestamp": date_str()}
        ]

    def get_num_turns(self, participant="assistant"):
        return sum(1 for msg in self.trace if msg["role"] == participant)

    def _get_last_system_verification(self):
        for msg in reversed(self.trace):
            if msg.get("role") == "log":
                content = msg.get("content", {})
                if content.get("type") == "system-verification":
                    return content.get("response")
        return None

    def _revealed_shard_ids(self, shards):
        return set(
            [
                msg["content"]["shard_id"]
                for msg in self.trace
                if msg.get("role") == "log"
                and msg.get("content", {}).get("type") == "shard_revealed"
            ]
        )

    def run(self, verbose=False, save_log=True):
        _is_reasoning_model = is_reasoning_model(
            self.assistant_model, override=self.reasoning_cls_override
        )
        max_assistant_tokens = 10000 if _is_reasoning_model else 1000

        is_completed, is_correct, score = False, False, None
        shards = self.sample["shards"]

        while not is_completed:
            revealed_shard_ids = self._revealed_shard_ids(shards)
            all_shards_revealed = len(revealed_shard_ids) == len(shards)

            use_synthetic_user = False
            if all_shards_revealed:
                if not self.enable_finalization_nudge or self._finalization_prompted:
                    if verbose:
                        print_colored(
                            f"[log] all shards revealed ({len(revealed_shard_ids)} / {len(shards)}); stopping",
                            "blue",
                        )
                    break

                last_verification = self._get_last_system_verification()
                last_type = (
                    last_verification.get("response_type")
                    if isinstance(last_verification, dict)
                    else None
                )

                if last_type in ["clarification", "discussion"]:
                    use_synthetic_user = True
                    self._finalization_prompted = True
                else:
                    if verbose:
                        print_colored(
                            f"[log] all shards revealed ({len(revealed_shard_ids)} / {len(shards)}); stopping",
                            "blue",
                        )
                    break

            is_last_shard_turn = len(revealed_shard_ids) == len(shards) - 1
            is_last_turn = True if use_synthetic_user else is_last_shard_turn

            # 1) user response
            if use_synthetic_user:
                user_response = FINALIZATION_USER_MESSAGE
                shard_revealed_id = -1
                cost_usd = 0.0
            else:
                user_response, shard_revealed_id, cost_usd = (
                    self.user_agent.generate_response(
                        self.trace, self.sample, temperature=self.user_temperature
                    )
                )

            self.trace.append(
                {
                    "role": "user",
                    "content": user_response,
                    "timestamp": date_str(),
                    "cost_usd": cost_usd,
                }
            )
            if verbose:
                print_colored(f"[user] {user_response}", "green")

            if (not use_synthetic_user) and shard_revealed_id != -1:
                self.trace.append(
                    {
                        "role": "log",
                        "content": {
                            "type": "shard_revealed",
                            "shard_id": shard_revealed_id,
                        },
                        "timestamp": date_str(),
                    }
                )
                if verbose:
                    print_colored(f"[log] shard revealed: {shard_revealed_id}", "blue")

            # 2) assistant response with editor
            assistant_input = None
            editor_cost = 0.0
            edited_state = None

            if self.enable_editor:
                edited_state, editor_obj = edit_conversation_state(
                    trace=self.trace,
                    editor_model=self.editor_model,
                    temperature=self.editor_temperature,
                    max_tokens=self.editor_max_tokens,
                    editor_preamble=self.editor_preamble,  # NEW
                )
                editor_cost = editor_obj.get("total_usd", 0.0)

                assistant_input = build_assistant_input_from_edited_state(
                    original_system_message=self.system_message,
                    last_user_message=user_response,
                    edited_state=edited_state,
                    assistant_preamble=self.assistant_preamble,  # NEW
                )

                if self.log_editor_artifacts:
                    self.trace.append(
                        {
                            "role": "log",
                            "content": {
                                "type": "context-editor",
                                "edited_state": edited_state,
                                "assistant_input_preview": assistant_input,
                            },
                            "timestamp": date_str(),
                            "cost_usd": editor_cost,
                        }
                    )
                    if verbose:
                        print_colored("[log] context editor applied", "blue")
            else:
                assistant_input = extract_conversation(self.trace, to_str=False)

            assistant_response_obj = generate(
                assistant_input,
                model=self.assistant_model,
                temperature=self.assistant_temperature,
                return_metadata=True,
                max_tokens=max_assistant_tokens,
            )
            assistant_response = assistant_response_obj["message"]
            self.trace.append(
                {
                    "role": "assistant",
                    "content": assistant_response,
                    "timestamp": date_str(),
                    "cost_usd": assistant_response_obj["total_usd"],
                }
            )
            if verbose:
                print_colored(f"[assistant] {assistant_response}", "red")

            # 3) system verification
            system_verification_response, verification_cost_usd = (
                self.system_agent.verify_system_response(self.trace)
            )
            self.trace.append(
                {
                    "role": "log",
                    "content": {
                        "type": "system-verification",
                        "response": system_verification_response,
                    },
                    "timestamp": date_str(),
                    "cost_usd": verification_cost_usd,
                }
            )
            if verbose:
                print_colored(
                    f"[log] system verification: {system_verification_response}", "blue"
                )

            if system_verification_response["response_type"] == "answer_attempt":
                extracted_answer = self.system_agent.extract_answer(self.trace)
                is_correct, score = None, None

                if self.task_name == "summary" and not is_last_turn:
                    evaluation_return = {"score": 0.0}
                    score = 0.0
                else:
                    evaluation_return = self.task.evaluator_function(
                        extracted_answer, self.sample
                    )
                    is_correct = evaluation_return.get("is_correct", None)
                    score = evaluation_return.get("score", None)

                if score == 1.0 and not is_correct:
                    is_correct = True

                self.trace.append(
                    {
                        "role": "log",
                        "content": {
                            "type": "answer-evaluation",
                            "exact_answer": extracted_answer,
                            "is_correct": is_correct,
                            "score": score,
                            "evaluation_return": evaluation_return,
                        },
                        "timestamp": date_str(),
                    }
                )

                if is_correct:
                    is_completed = True
                    self.trace.append(
                        {
                            "role": "log",
                            "content": {
                                "type": "conversation-completed",
                                "is_correct": is_correct,
                            },
                            "timestamp": date_str(),
                        }
                    )

            elif system_verification_response["response_type"] in [
                "clarification",
                "discussion",
            ]:
                continue

        if save_log:
            conv_type = "sharded-edited"
            if self.run_with_custom_temperature:
                conv_type = f"sharded-edited-at{self.assistant_temperature}-ut{self.user_temperature}"

            log_conversation(
                conv_type,
                self.task.get_task_name(),
                self.sample["task_id"],
                self.dataset_fn,
                self.assistant_model,
                self.system_model,
                self.user_model,
                self.trace,
                is_correct,
                score,
                log_folder=self.log_folder,
            )

        return is_correct, score
