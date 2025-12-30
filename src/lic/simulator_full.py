import random

from utils_log import log_conversation
from system_agent import SystemAgent
from model_openai import generate
from tasks import get_task
from utils import date_str


class ConversationSimulatorFull:
    def __init__(self, sample, assistant_model, system_model, run_concat=False, run_shuffle_concat=False, temperature=1.0, dataset_fn=None, log_folder=None):
        self.task_name = sample["task"]
        self.task = get_task(self.task_name)
        self.dataset_fn = dataset_fn
        self.sample = sample
        self.assistant_model = assistant_model
        self.system_model = system_model
        self.run_concat = run_concat
        self.run_shuffle_concat = run_shuffle_concat
        self.log_folder = log_folder
        self.run_custom_temperature = temperature != 1.0
        self.temperature = temperature

        self.system_message = self.task.generate_system_prompt(self.sample)
        self.system_agent = SystemAgent(self.task_name, self.system_model, self.sample)

    def run(self, verbose=False, save_log=True):
        if self.run_shuffle_concat and self.run_concat:
            raise ValueError("Cannot set both run_concat and run_shuffle_concat to True")

        if self.run_shuffle_concat:
            conv_type = "shuffle-concat"

            random.shuffle(self.sample["shards"])

            input_prompt = self.task.populate_concat_prompt(self.sample)
        elif self.run_concat:
            conv_type = "concat"
            input_prompt = self.task.populate_concat_prompt(self.sample)
        else:
            conv_type = "full"
            input_prompt = self.task.populate_fully_specific_prompt(self.sample)

        if verbose:
            print(f"\033[92m[system] {input_prompt}\033[0m")

        # custom output dir for different temperatures to not mix up
        if self.run_custom_temperature:
            conv_type = f"{conv_type}-t{self.temperature}"

        is_reasoning_model = "o1" in self.assistant_model or "o3" in self.assistant_model or "deepseek-r1" in self.assistant_model

        max_tokens = 16000 if is_reasoning_model else 1000
        assistant_response_obj = generate([{"role": "system", "content": self.system_message},{"role": "user", "content": input_prompt}], model=self.assistant_model, return_metadata=True, temperature=self.temperature, max_tokens=max_tokens)

        assistant_response = assistant_response_obj["message"]
        if verbose:
            print(f"\033[91m[assistant] {assistant_response}\033[0m")

        trace = [{"role": "system", "content": self.system_message}, {"role": "user", "content": input_prompt}, {"role": "assistant", "content": assistant_response, "cost_usd": assistant_response_obj["total_usd"]}]

        extracted_answer = self.system_agent.extract_answer(trace)

        evaluation_return = self.task.evaluator_function(extracted_answer, self.sample)
        assert type(evaluation_return) is dict and ("score" in evaluation_return or "is_correct" in evaluation_return), "Evaluator function should return a dictionary with 'score' or 'is_correct' key"
        score = evaluation_return.get("score", None)
        is_correct = score == 1.0

        trace.append({"role": "log", "content": {"type": "answer-evaluation", "exact_answer": extracted_answer, "is_correct": is_correct, "score": score, "evaluation_return": evaluation_return}, "timestamp": date_str()})

        if verbose:
            print('==================================================')
            icon = "\033[92m✔\033[0m" if is_correct else "\033[91m✘\033[0m"
            print(f"{icon} {extracted_answer} (score: {score})")

        if save_log:
            log_conversation(conv_type, self.task_name, self.sample["task_id"], self.dataset_fn, assistant_model=self.assistant_model, system_model="NA", user_model="NA", trace=trace, is_correct=is_correct, score=score, log_folder=self.log_folder)
        return is_correct, score



if __name__ == "__main__":
    import json, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--assistant_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--system_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--task", type=str, default="math")
    parser.add_argument("--run_concat", action="store_true")
    parser.add_argument("--run_shuffle_concat", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.run_concat and args.run_shuffle_concat:
        raise ValueError("Cannot set both run_concat and run_shuffle_concat to True")

    with open("data/sharded_instructions_600.json", "r") as f:
        data = json.load(f)

    data = [d for d in data if (d["task"] == args.task or args.task == "all")]

    sample = random.choice(data)

    conversation_simulator = ConversationSimulatorFull(sample, args.assistant_model, args.system_model, args.run_concat, args.run_shuffle_concat)
    conversation_simulator.run(verbose=args.verbose, save_log=False)
