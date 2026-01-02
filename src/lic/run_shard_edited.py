import argparse
import json
import multiprocessing
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import tqdm

from lic.sim_shard_edited import ConversationSimulatorShardedEdited
from lic.utils_log import get_run_counts


def run_simulation(todo, args):
    dataset_fn = todo["dataset_fn"]
    try:
        conversation_simulator = ConversationSimulatorShardedEdited(
            todo["sample"],
            assistant_model=todo["assistant_model"],
            system_model=todo["system_model"],
            user_model=todo["user_model"],
            assistant_temperature=todo["assistant_temperature"],
            user_temperature=todo["user_temperature"],
            dataset_fn=dataset_fn,
            log_folder=args.log_folder,
            reasoning_cls_override=todo.get("reasoning_cls_override", None),
            # editor knobs
            editor_model=todo["editor_model"],
            editor_temperature=todo["editor_temperature"],
            editor_max_tokens=todo["editor_max_tokens"],
            enable_editor=todo["enable_editor"],
            log_editor_artifacts=todo["log_editor_artifacts"],
        )

        conversation_simulator.run(verbose=args.verbose, save_log=True)

    except Exception as e:
        import traceback

        error_msg = traceback.format_exc()
        tqdm.tqdm.write(
            f"\033[91m [Error on {todo['sample']['task_id']}; {todo['assistant_model']}; {todo['conv_type']}]:\n{error_msg}\033[0m"
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()

    # dataset / selection
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="data/sharded_instructions_600.json",
        help="Dataset file to use",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["math"],
        help='Tasks to run (e.g. "code", "database", "actions", "math", "data2text", "summary", "translation")',
    )
    parser.add_argument(
        "--problem_limit",
        type=int,
        default=-1,
        help="Limit number of problems to run per selected task (-1 for no limit)",
    )

    # models
    parser.add_argument(
        "--assistant_models",
        nargs="+",
        default=["gpt-4o-mini"],
        help="List of assistant models to run",
    )
    parser.add_argument(
        "--system_model",
        type=str,
        default="gpt-4o-mini",
        help="System model to use (for verification etc.)",
    )
    parser.add_argument(
        "--user_model",
        type=str,
        default="gpt-4o-mini",
        help="User model to use",
    )

    # editor
    parser.add_argument(
        "--editor_model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for context editing between turns",
    )
    parser.add_argument(
        "--editor_temperature",
        type=float,
        default=0.0,
        help="Temperature for the editor model (recommend 0.0)",
    )
    parser.add_argument(
        "--editor_max_tokens",
        type=int,
        default=1200,
        help="Max tokens for the editor output",
    )
    parser.add_argument(
        "--disable_editor",
        action="store_true",
        help="Disable the editor (falls back to standard sharded behavior, but logs as sharded-edited anyway)",
    )
    parser.add_argument(
        "--no_log_editor_artifacts",
        action="store_true",
        help="Do not store editor outputs in the trace logs",
    )

    # run counts / parallelism / logging
    parser.add_argument(
        "--N_runs",
        type=int,
        default=1,
        help="Number of sharded-edited runs per sample per assistant model",
    )
    parser.add_argument(
        "--N_workers",
        type=int,
        default=1,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--log_folder",
        type=str,
        default="logs",
        help="Log folder to use",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    # temperatures
    parser.add_argument(
        "--assistant_temperature",
        type=float,
        default=1.0,
        help="Temperature for assistant model",
    )
    parser.add_argument(
        "--user_temperature",
        type=float,
        default=1.0,
        help="Temperature for user model",
    )

    # reasoning override passthrough (kept compatible with your original script)
    parser.add_argument(
        "--reasoning_cls_override",
        type=json.loads,
        default="{}",
        help="JSON dict of reasoning classification override.",
    )

    args = parser.parse_args()

    # Windows dataset path fix like your original script
    dataset_fn = args.dataset_file
    if dataset_fn.startswith(".\\"):
        dataset_fn = dataset_fn[2:]
    dataset_fn = dataset_fn.replace("\\", "/")

    with open(dataset_fn, "r") as f:
        samples = json.load(f)

    # filter by task
    samples = [s for s in samples if s["task"] in args.tasks]

    # optional problem limit per task
    if args.problem_limit > 0:
        per_task_counts = Counter()
        selected_samples = []
        for sample in samples:
            if per_task_counts[sample["task"]] < args.problem_limit:
                selected_samples.append(sample)
                per_task_counts[sample["task"]] += 1
        samples = selected_samples

    print(f"Loaded {len(samples)} samples after filtering")

    random.shuffle(samples)

    # build conv_type naming similar to original
    sharded_extra = (
        f"-at{args.assistant_temperature}-ut{args.user_temperature}"
        if args.assistant_temperature != 1.0 or args.user_temperature != 1.0
        else ""
    )
    editor_extra = f"-em{args.editor_model}"
    conv_type = f"sharded-edited{sharded_extra}{editor_extra}"

    # compute run counts to avoid rerunning completed ones
    # (same behavior as your original script: get_run_counts keyed by task_id)
    all_todos = []
    for assistant_model in args.assistant_models:
        run_counts = Counter()
        for task in sorted(set([s["task"] for s in samples])):
            run_counts.update(
                get_run_counts(
                    conv_type,
                    task,
                    assistant_model,
                    dataset_fn,
                    log_folder=args.log_folder,
                )
            )
        print(
            f"[{assistant_model}] Existing run counts: {sum(run_counts.values())} total"
        )

        for sample in samples:
            missing = args.N_runs - run_counts[sample["task_id"]]
            if missing <= 0:
                continue

            for _ in range(missing):
                all_todos.append(
                    {
                        "sample": sample,
                        "assistant_model": assistant_model,
                        "conv_type": conv_type,
                        "system_model": args.system_model,
                        "user_model": args.user_model,
                        "dataset_fn": dataset_fn,
                        "assistant_temperature": args.assistant_temperature,
                        "user_temperature": args.user_temperature,
                        "reasoning_cls_override": args.reasoning_cls_override,
                        # editor config
                        "editor_model": args.editor_model,
                        "editor_temperature": args.editor_temperature,
                        "editor_max_tokens": args.editor_max_tokens,
                        "enable_editor": (not args.disable_editor),
                        "log_editor_artifacts": (not args.no_log_editor_artifacts),
                    }
                )

    random.shuffle(all_todos)

    print(f"Running {len(all_todos)} conversations")
    print("Assistant model counts:", Counter([t["assistant_model"] for t in all_todos]))
    print("Conv type:", conv_type)

    if len(all_todos) == 0:
        print("Nothing to run (all requested runs already exist in logs).")
    else:
        with ThreadPoolExecutor(max_workers=args.N_workers) as executor:
            list(
                tqdm.tqdm(
                    executor.map(lambda todo: run_simulation(todo, args), all_todos),
                    total=len(all_todos),
                )
            )
