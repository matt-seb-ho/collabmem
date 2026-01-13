# run_cheatsheet_baseline.py
import argparse
import json
import os
import random
from collections import Counter

import tqdm

from lic.cheatsheet_logger import CheatsheetLogConfig, CheatsheetLogger
from lic.cheatsheet_memory import CheatsheetConfig, CheatsheetMemory
from lic.cheatsheet_sim_sharded import ConversationSimulatorSharded
from lic.constants import PROMPT_FILE_PATHS


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_samples(
    dataset_fn: str, tasks: list[str], problem_limit: int, seed: int, shuffle: bool
):
    with open(dataset_fn, "r", encoding="utf-8") as f:
        samples = json.load(f)
    samples = [s for s in samples if s["task"] in tasks]
    if problem_limit > 0:
        # limit per task
        out = []
        per_task = Counter()
        for s in samples:
            if per_task[s["task"]] < problem_limit:
                out.append(s)
                per_task[s["task"]] += 1
        samples = out
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)
    return samples


def find_last_eval(trace):
    for msg in reversed(trace):
        if msg["role"] == "log" and msg["content"].get("type") == "answer-evaluation":
            return {
                "extracted_answer": msg["content"].get("exact_answer"),
                "score": msg["content"].get("score"),
                "is_correct": msg["content"].get("is_correct"),
            }
    return {"extracted_answer": None, "score": None, "is_correct": None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_file", type=str, required=True, help="Train or eval dataset JSON"
    )
    p.add_argument(
        "--split_name",
        type=str,
        default="train",
        help="Name for logging (train/eval/mini_eval)",
    )
    p.add_argument("--tasks", nargs="+", default=["math"])
    p.add_argument("--models", nargs="+", default=["gpt-4o-mini"])

    p.add_argument("--assistant_temperature", type=float, default=1.0)
    p.add_argument("--user_temperature", type=float, default=1.0)
    p.add_argument("--system_model", type=str, default="gpt-4o-mini")
    p.add_argument("--user_model", type=str, default="gpt-4o-mini")

    p.add_argument(
        "--mode", type=str, choices=["vanilla", "warmup", "frozen"], required=True
    )
    p.add_argument("--curator_model", type=str, default="gpt-4o-mini")
    p.add_argument(
        "--cheatsheet_update_prompt_path",
        type=str,
        required=False,
        default=PROMPT_FILE_PATHS["cheatsheet_v2"],
        help="Path to cheatsheet update template (required unless mode=frozen)",
    )
    p.add_argument(
        "--init_cheatsheet_path",
        type=str,
        default=None,
        help="Start cheatsheet from file",
    )
    p.add_argument("--save_final_cheatsheet_path", type=str, default=None)

    p.add_argument("--log_folder", type=str, default="logs_cheatsheet")
    p.add_argument("--run_name", type=str, default=None)

    p.add_argument("--problem_limit", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_shuffle", action="store_true")

    # if you use reasoning overrides in LiC
    p.add_argument("--reasoning_cls_override", type=json.loads, default="{}")

    p.add_argument(
        "--cheatsheet_scope",
        choices=["global", "per_task"],
        default="per_task",
        help="Whether cheatsheet is shared across tasks or separate per task",
    )

    # NEW: Ablation flags for extrinsic grounding in reflection prompt
    p.add_argument(
        "--include_eval_label",
        action="store_true",
        help="Include evaluation correctness label/summary as reflection fodder (a)",
    )
    p.add_argument(
        "--include_full_spec_q",
        action="store_true",
        help="Include the oracle single-turn fully specified question sample['full_spec_q'] (b)",
    )
    p.add_argument(
        "--include_ground_truth",
        action="store_true",
        help="Include ground truth output sample['ground_truth_a'] (c)",
    )

    args = p.parse_args()

    if args.mode != "frozen" and not args.cheatsheet_update_prompt_path:
        raise ValueError(
            "--cheatsheet_update_prompt_path is required unless --mode=frozen"
        )

    cheatsheet_update_template = ""
    if args.cheatsheet_update_prompt_path:
        cheatsheet_update_template = read_text(args.cheatsheet_update_prompt_path)

    initial_cheatsheet = "(empty)"
    if args.init_cheatsheet_path:
        initial_cheatsheet = read_text(args.init_cheatsheet_path)

    if args.run_name is None:
        args.run_name = f"dc_{args.mode}_{'_'.join(args.tasks)}"

    logger = CheatsheetLogger(
        CheatsheetLogConfig(
            log_dir=os.path.join(args.log_folder, args.run_name),
            run_name=args.run_name,
            save_full_text=True,
        )
    )

    samples = load_samples(
        args.dataset_file,
        args.tasks,
        args.problem_limit,
        args.seed,
        shuffle=(not args.no_shuffle),
    )
    print(
        f"Loaded {len(samples)} samples from {args.dataset_file} (tasks={args.tasks})"
    )

    for assistant_model in args.models:
        cfg = CheatsheetConfig(
            enable_updates=(args.mode != "frozen"),
            curator_model=args.curator_model,
            curator_temperature=(
                1.0 if args.curator_model.startswith("gpt-5") else 0.0
            ),
            curator_max_tokens=2000,
            cheatsheet_update_template=cheatsheet_update_template,
            # NEW toggles (ablatable, independent of mode)
            include_eval_label=args.include_eval_label,
            include_full_spec_q=args.include_full_spec_q,
            include_ground_truth=args.include_ground_truth,
        )

        if args.cheatsheet_scope == "global":
            memories = {"__global__": CheatsheetMemory(initial_cheatsheet, cfg)}
        else:
            memories = {
                task: CheatsheetMemory(initial_cheatsheet, cfg) for task in args.tasks
            }

        correct = 0
        total = 0

        pbar = tqdm.tqdm(
            enumerate(samples),
            total=len(samples),
            desc=f"[{assistant_model}] {args.mode} ({args.split_name})",
            dynamic_ncols=True,
        )

        for episode_idx, sample in pbar:
            task_key = (
                "__global__" if args.cheatsheet_scope == "global" else sample["task"]
            )
            memory = memories[task_key]
            preamble = memory.build_assistant_preamble()

            sim = ConversationSimulatorSharded(
                sample=sample,
                assistant_model=assistant_model,
                system_model=args.system_model,
                user_model=args.user_model,
                assistant_temperature=args.assistant_temperature,
                user_temperature=args.user_temperature,
                dataset_fn=args.dataset_file,
                log_folder=os.path.join(args.log_folder, args.run_name, "lic_logs"),
                reasoning_cls_override=args.reasoning_cls_override,
                assistant_preamble=preamble,
            )

            is_correct, score = sim.run(verbose=False, save_log=True)
            trace = sim.trace
            eval_summary = find_last_eval(trace)

            # Reflection fodder from dataset (may be None)
            full_spec_q = sample.get("full_spec_q")
            ground_truth_a = sample.get("ground_truth_a")

            # update cheatsheet (or not)
            old_cheatsheet = memory.cheatsheet
            upd = memory.maybe_update(
                task_name=sample["task"],
                system_message=sim.system_message,
                trace=trace,
                eval_summary=eval_summary,
                full_spec_q=full_spec_q,
                ground_truth_a=ground_truth_a,
            )

            # log revision
            logger.log_revision(
                dataset_fn=args.dataset_file,
                split_name=args.split_name,
                mode=args.mode,
                sample=sample,
                models={
                    "assistant": assistant_model,
                    "system": args.system_model,
                    "user": args.user_model,
                    "curator": args.curator_model,
                },
                eval_summary=eval_summary,
                curator_meta=upd.get("curator_meta"),
                cheatsheet_old=upd.get("old", old_cheatsheet),
                cheatsheet_new=upd.get("new", memory.cheatsheet),
                episode_idx=episode_idx,
                cheatsheet_scope=args.cheatsheet_scope,
            )

            total += 1
            if is_correct or (score == 1.0):
                correct += 1

            pbar.set_postfix(
                {
                    "task": sample["task"],
                    "acc": f"{correct}/{total}",
                }
            )

        print(
            f"[{assistant_model}] DONE split={args.split_name} mode={args.mode} "
            f"acc={correct}/{total} = {correct / total:.3f}"
        )

        if args.save_final_cheatsheet_path:
            base, ext = os.path.splitext(args.save_final_cheatsheet_path)
            ext = ext or ".txt"
            os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

            if args.cheatsheet_scope == "global":
                out_path = f"{base}_{assistant_model}{ext}"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(memories["__global__"].cheatsheet)
                print(f"Saved final cheatsheet to {out_path}")
            else:
                for task, mem in memories.items():
                    out_path = f"{base}_{assistant_model}_{task}{ext}"
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(mem.cheatsheet)
                    print(f"Saved final cheatsheet for {task} to {out_path}")


if __name__ == "__main__":
    main()
