import json
import random

from lic.model import generate, is_reasoning_model
from lic.system_agent import SystemAgent
from lic.tasks import get_task
from lic.user_agent import UserAgent
from lic.utils import date_str, extract_conversation, print_colored
from lic.utils_log import log_conversation


class ConversationSimulatorSharded:
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
        assistant_preamble: str | None = None,
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
        self.assistant_preamble = assistant_preamble

        self.run_with_custom_temperature = (
            assistant_temperature != 1.0 or user_temperature != 1.0
        )
        self.assistant_temperature = assistant_temperature
        self.user_temperature = user_temperature

        self.trace = [
            {"role": "system", "content": self.system_message, "timestamp": date_str()}
        ]
        if assistant_preamble is not None:
            self.trace.append(
                {
                    "role": "assistant",
                    "content": assistant_preamble,
                    "timestamp": date_str(),
                }
            )

        self.reasoning_cls_override = reasoning_cls_override

    def get_num_turns(self, participant="assistant"):
        return sum(1 for msg in self.trace if msg["role"] == participant)

    def run(self, verbose=False, save_log=True):
        _is_reasoning_model = is_reasoning_model(
            self.assistant_model, override=self.reasoning_cls_override
        )
        max_assistant_tokens = 10000 if _is_reasoning_model else 1000
        is_completed, is_correct, score = False, False, None

        shards = self.sample["shards"]

        while not is_completed:
            revealed_shard_ids = set(
                [
                    msg["content"]["shard_id"]
                    for msg in self.trace
                    if msg["role"] == "log"
                    and msg["content"]["type"] == "shard_revealed"
                ]
            )
            all_shards_revealed = len(revealed_shard_ids) == len(shards)
            if all_shards_revealed:
                if verbose:
                    print_colored(
                        f"[log] all shards revealed ({revealed_shard_ids} / {len(shards)})",
                        "blue",
                    )
                break  # no need to keep going, nothing else to reveal

            is_last_turn = len(revealed_shard_ids) == len(shards) - 1

            # 1. get a user response
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

            if shard_revealed_id != -1:
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

            # 2. get the assistant's response
            assistant_response_obj = generate(
                extract_conversation(self.trace, to_str=False),
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

            # 3. evaluate if the last turn in the conversation asks a clarification question, or proposes a solution.
            system_verification_response, verification_cost_usd = (
                self.system_agent.verify_system_response(self.trace)
            )

            # log it
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
                # 4. extract the answer
                extracted_answer = self.system_agent.extract_answer(self.trace)
                is_correct, score = None, None
                if self.task_name == "summary" and not is_last_turn:
                    # we skip eval, and only run the evaluator on the last turn
                    evaluation_return = {"score": 0.0}
                    score = 0.0
                else:
                    evaluation_return = self.task.evaluator_function(
                        extracted_answer, self.sample
                    )

                    assert (
                        type(evaluation_return) is dict
                        and (
                            "score" in evaluation_return
                            or "is_correct" in evaluation_return
                        )
                    ), "Evaluator function should return a dictionary with 'score' or 'is_correct' key"
                    is_correct = evaluation_return.get("is_correct", None)
                    score = evaluation_return.get("score", None)

                if score == 1.0 and not is_correct:
                    is_correct = True

                # let's log that
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
                if verbose:
                    print_colored(
                        f"[log] answer evaluation:\n```{extracted_answer}\n```\n({'correct' if is_correct else 'incorrect'}; score: {score})",
                        "blue",
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
                    if verbose:
                        print_colored(
                            f"[log] conversation completed: {is_correct}; score: {score}",
                            "blue",
                        )
                # For now, if the system last assistant response is incorrect;
            elif system_verification_response["response_type"] in [
                "clarification",
                "discussion",
            ]:
                continue  # end of the turn
        if save_log:
            conv_type = "sharded"
            if self.run_with_custom_temperature:
                conv_type = (
                    f"sharded-at{self.assistant_temperature}-ut{self.user_temperature}"
                )
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


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="math")
    parser.add_argument("--assistant_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--system_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--user_model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    dataset_fn = "data/sharded_instructions_600.json"
    with open(dataset_fn, "r") as f:
        data = json.load(f)

    data = [d for d in data if d["task"] == args.task]

    sample = random.choice(data)

    conversation_simulator = ConversationSimulatorSharded(
        sample=sample,
        assistant_model=args.assistant_model,
        system_model=args.system_model,
        user_model=args.user_model,
        dataset_fn=dataset_fn,
    )
    conversation_simulator.run(verbose=True, save_log=True)
