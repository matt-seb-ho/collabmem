import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import tqdm

from lic.model_openai import generate
from lic.system_agent import SystemAgent
from lic.tasks import get_task
from lic.utils import date_str, extract_conversation
from lic.utils_log import log_conversation

recap_message = """Just to recapitulate, the entire task is:

[[SAMPLE_TEXT]]

Now please complete the task correctly taking all the information into account."""


class RecapSimulator:
    def __init__(self, task_name):
        self.task_name = task_name
        self.task = get_task(task_name)
        self.dataset_fn = self.task.get_dataset_file()

    def run_recap_sample(self, sharded_log, conv_type, save_log=True):
        sharded_conv_id = sharded_log["conv_id"]

        sharded_trace = sharded_log["trace"]
        sharded_sample = self.task.get_sample(sharded_log["task_id"])

        assistant_model, system_model, user_model = (
            sharded_log["assistant_model"],
            sharded_log["system_model"],
            sharded_log["user_model"],
        )

        system_agent = SystemAgent(self.task_name, system_model, sample=sharded_sample)

        recap_trace = sharded_trace.copy()

        # 1. prep the user recap message
        sample_text = (
            self.task.populate_fully_specific_prompt(sharded_sample)
            if conv_type == "recap-full"
            else self.task.populate_concat_prompt(sharded_sample)
        )
        recap_message_populated = recap_message.replace("[[SAMPLE_TEXT]]", sample_text)
        recap_trace.append(
            {
                "role": "user",
                "content": recap_message_populated,
                "timestamp": date_str(),
                "cost_usd": 0.0,
            }
        )

        assistant_conversation = extract_conversation(
            recap_trace, to_str=False, skip_system=False
        )

        # 2. get the assistant's response
        max_tokens = 3000 if "o1-" in assistant_model else 1000
        recap_response_obj = generate(
            assistant_conversation,
            model=assistant_model,
            temperature=1.0,
            return_metadata=True,
            max_tokens=max_tokens,
        )
        recap_response = recap_response_obj["message"]
        recap_trace.append(
            {
                "role": "assistant",
                "content": recap_response,
                "timestamp": date_str(),
                "cost_usd": recap_response_obj["total_usd"],
            }
        )

        # 3. verify the assistant's response
        system_verification_response, system_verification_cost_usd = (
            system_agent.verify_system_response(recap_trace)
        )
        recap_trace.append(
            {
                "role": "log",
                "content": {
                    "type": "system-verification",
                    "response": system_verification_response,
                },
                "timestamp": date_str(),
                "cost_usd": system_verification_cost_usd,
            }
        )

        # 4. extract the answer
        is_correct = False
        score = None
        if system_verification_response["response_type"] == "answer_attempt":
            extracted_answer = system_agent.extract_answer(recap_trace)
            evaluation_return = self.task.evaluator_function(
                extracted_answer, sharded_sample
            )
            assert (
                type(evaluation_return) is dict
                and ("score" in evaluation_return or "is_correct" in evaluation_return)
            ), "Evaluator function should return a dictionary with 'score' or 'is_correct' key"
            is_correct = evaluation_return.get("is_correct", None)
            score = evaluation_return.get("score", None)
            recap_trace.append(
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
                recap_trace.append(
                    {
                        "role": "log",
                        "content": {
                            "type": "conversation-completed",
                            "is_correct": is_correct,
                        },
                        "timestamp": date_str(),
                    }
                )

        if save_log:
            additional_info = {
                "source_conv_id": sharded_conv_id,
                "source_is_correct": sharded_log["is_correct"],
            }
            log_conversation(
                conv_type,
                self.task.get_task_name(),
                sharded_sample["task_id"],
                self.dataset_fn,
                assistant_model,
                system_model,
                user_model,
                recap_trace,
                is_correct=is_correct,
                score=score,
                additional_info=additional_info,
            )

        return is_correct, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="database")
    parser.add_argument("--models", type=str, nargs="+")
    parser.add_argument("--N_workers", type=int, default=5)
    args = parser.parse_args()

    task_name = args.task
    todo_convs = []

    samples = get_task(task_name).get_samples()
    task_ids = set([d["task_id"] for d in samples])

    for recap_type in ["concat"]:
        conv_type = f"recap-{recap_type}"

        if not os.path.exists(f"logs/{task_name}/{conv_type}"):
            os.makedirs(f"logs/{task_name}/{conv_type}")

        sharded_conv_ids = []
        conv_id2log = {}
        # Get task_id-based counter of number of logs for this task and conv type
        task_id_counts = []
        for fn in os.listdir(f"logs/{task_name}/sharded"):
            with open(f"logs/{task_name}/sharded/{fn}", "r") as f:
                for line in f:
                    log = json.loads(line)
                    if (
                        log["dataset_fn"] == f"sharded_{args.task}.json"
                        and log["task_id"] in task_ids
                    ):
                        task_id_counts.append(log["task_id"])

        for fn in os.listdir(f"logs/{task_name}/recap-{recap_type}"):
            with open(f"logs/{task_name}/recap-{recap_type}/{fn}", "r") as f:
                for line in f:
                    log = json.loads(line)
                    if (
                        log["dataset_fn"] != f"sharded_{args.task}.json"
                        or log["task_id"] not in task_ids
                    ):
                        continue

                    model_name = (
                        log["assistant_model"][2:]
                        if log["assistant_model"].startswith("t-")
                        else log["assistant_model"]
                    )
                    if args.models is None or model_name in args.models:
                        sharded_conv_ids.append(log["conv_id"])
                        conv_id2log[log["conv_id"]] = log

        already_completed_conv_ids = set([])
        for fn in os.listdir(f"logs/{task_name}/{conv_type}"):
            with open(f"logs/{task_name}/{conv_type}/{fn}", "r") as f:
                for line in f:
                    conv = json.loads(line)
                    if (
                        conv["dataset_fn"] != f"sharded_{args.task}.json"
                        or conv["task_id"] not in task_ids
                    ):
                        continue

                    model_name = (
                        conv["assistant_model"][2:]
                        if conv["assistant_model"].startswith("t-")
                        else conv["assistant_model"]
                    )
                    if args.models is None or model_name in args.models:
                        already_completed_conv_ids.add(conv["source_conv_id"])

        # now we need to find the ones in sharded_conv_ids that are not in already_completed_conv_ids
        for conv_id in sharded_conv_ids:
            if conv_id not in already_completed_conv_ids:
                model_name = (
                    conv_id2log[conv_id]["assistant_model"][2:]
                    if conv_id2log[conv_id]["assistant_model"].startswith("t-")
                    else conv_id2log[conv_id]["assistant_model"]
                )
                if args.models is None or model_name in args.models:
                    todo_convs.append(
                        {"conv": conv_id2log[conv_id], "conv_type": conv_type}
                    )

        print(
            f"[{conv_type}] Found {len(sharded_conv_ids)} sharded conversations, {len(already_completed_conv_ids)} already completed; todo: {len(todo_convs)}"
        )

    recap_simulator = RecapSimulator(task_name=task_name)

    with ThreadPoolExecutor(max_workers=args.N_workers) as executor:
        list(
            tqdm.tqdm(
                executor.map(
                    lambda conv: recap_simulator.run_recap_sample(
                        conv["conv"], conv["conv_type"]
                    ),
                    todo_convs,
                ),
                total=len(todo_convs),
            )
        )
