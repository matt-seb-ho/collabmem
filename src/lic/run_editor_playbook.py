# run_editor_cheatsheet.py
import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import tqdm

from lic.editor_playbook_logger import (
    EditorCheatsheetLogConfig,
    EditorCheatsheetLogger,
)
from lic.editor_playbook_mem import (
    DEFAULT_EDITOR_CHEATSHEET_UPDATE_TEMPLATE,
    EditorCheatsheetConfig,
    EditorCheatsheetMemory,
)
from lic.sim_shard_edited import ConversationSimulatorShardedEdited


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
        if (
            msg.get("role") == "log"
            and msg.get("content", {}).get("type") == "answer-evaluation"
        ):
            c = msg["content"]
            return {
                "extracted_answer": c.get("exact_answer"),
                "score": c.get("score"),
                "is_correct": c.get("is_correct"),
            }
    return {"extracted_answer": None, "score": None, "is_correct": None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_file", type=str, required=True)
    p.add_argument("--split_name", type=str, default="train")
    p.add_argument("--tasks", nargs="+", default=["math"])

    # Models used in the sharded-edited simulation
    p.add_argument("--assistant_models", nargs="+", default=["gpt-4o-mini"])
    p.add_argument("--assistant_temperature", type=float, default=1.0)
    p.add_argument("--user_temperature", type=float, default=1.0)
    p.add_argument("--system_model", type=str, default="gpt-4o-mini")
    p.add_argument("--user_model", type=str, default="gpt-4o-mini")

    # Editor model (the thing we are improving)
    p.add_argument("--editor_model", type=str, default="gpt-4o-mini")
    p.add_argument("--editor_temperature", type=float, default=1.0)
    p.add_argument("--editor_max_tokens", type=int, default=1200)
    p.add_argument("--disable_editor", action="store_true")
    p.add_argument("--no_log_editor_artifacts", action="store_true")

    # Cheatsheet / curator
    p.add_argument(
        "--mode", type=str, choices=["warmup", "vanilla", "frozen"], required=True
    )
    p.add_argument("--curator_model", type=str, default="gpt-4o-mini")
    p.add_argument("--curator_temperature", type=float, default=1.0)
    p.add_argument("--curator_max_tokens", type=int, default=2000)

    p.add_argument(
        "--cheatsheet_update_prompt_path",
        type=str,
        default=None,
        help="Optional override prompt template file. If omitted, uses built-in default.",
    )
    p.add_argument("--init_cheatsheet_path", type=str, default=None)
    p.add_argument("--save_final_cheatsheet_path", type=str, default=None)

    p.add_argument(
        "--cheatsheet_scope", choices=["global", "per_task"], default="per_task"
    )

    # Extrinsic grounding toggles
    p.add_argument("--include_eval_label", action="store_true")
    p.add_argument("--include_full_spec_q", action="store_true")
    p.add_argument("--include_ground_truth", action="store_true")

    # Logging
    p.add_argument("--log_folder", type=str, default="logs_editor_cheatsheet")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--problem_limit", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_shuffle", action="store_true")

    # LiC compatibility
    p.add_argument("--reasoning_cls_override", type=json.loads, default="{}")

    args = p.parse_args()

    update_template = None
    if args.cheatsheet_update_prompt_path:
        update_template = read_text(args.cheatsheet_update_prompt_path)

    initial_cheatsheet = "(empty)"
    per_task_init_cheatsheets = None
    if args.init_cheatsheet_path:
        # check if extension is json
        if Path(args.init_cheatsheet_path).suffix == ".json":
            with open(args.init_cheatsheet_path, "r", encoding="utf-8") as f:
                init_cheatsheet_data = json.load(f)
            per_task_init_cheatsheets = {
                k: Path(v).read_text() for k, v in init_cheatsheet_data.items()
            }
        initial_cheatsheet = read_text(args.init_cheatsheet_path)

    if args.run_name is None:
        args.run_name = f"editor_cs_{args.mode}_{'_'.join(args.tasks)}"

    logger = EditorCheatsheetLogger(
        EditorCheatsheetLogConfig(
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

    # Build config for editor-cheatsheet updates
    cfg = EditorCheatsheetConfig(
        enable_updates=(args.mode != "frozen"),
        curator_model=args.curator_model,
        curator_temperature=args.curator_temperature,
        curator_max_tokens=args.curator_max_tokens,
        include_eval_label=args.include_eval_label,
        include_full_spec_q=args.include_full_spec_q,
        include_ground_truth=args.include_ground_truth,
        cheatsheet_update_template=update_template
        or DEFAULT_EDITOR_CHEATSHEET_UPDATE_TEMPLATE,
    )

    # Initialize memories
    if args.cheatsheet_scope == "global":
        memories = {"__global__": EditorCheatsheetMemory(initial_cheatsheet, cfg)}
    else:
        if per_task_init_cheatsheets:
            memories = {
                task: EditorCheatsheetMemory(
                    per_task_init_cheatsheets.get(task, "(empty)"), cfg
                )
                for task in args.tasks
            }
        else:
            memories = {
                task: EditorCheatsheetMemory(initial_cheatsheet, cfg) for task in args.tasks
            }

    for assistant_model in args.assistant_models:
        correct = 0
        total = 0

        pbar = tqdm.tqdm(
            enumerate(samples),
            total=len(samples),
            desc=f"[{assistant_model}] editor-cheatsheet {args.mode} ({args.split_name})",
            dynamic_ncols=True,
        )

        for episode_idx, sample in pbar:
            task_key = (
                "__global__" if args.cheatsheet_scope == "global" else sample["task"]
            )
            mem = memories[task_key]

            # Frozen mode: do not depend on sequential updates (parallelizable externally).
            # Warmup/vanilla: per-episode sequential updates.
            editor_playbook = mem.build_editor_playbook()

            sim = ConversationSimulatorShardedEdited(
                sample=sample,
                assistant_model=assistant_model,
                system_model=args.system_model,
                user_model=args.user_model,
                assistant_temperature=args.assistant_temperature,
                user_temperature=args.user_temperature,
                dataset_fn=args.dataset_file,
                log_folder=os.path.join(args.log_folder, args.run_name, "lic_logs"),
                reasoning_cls_override=args.reasoning_cls_override,
                # editor knobs
                editor_model=args.editor_model,
                editor_temperature=args.editor_temperature,
                editor_max_tokens=args.editor_max_tokens,
                enable_editor=(not args.disable_editor),
                log_editor_artifacts=(not args.no_log_editor_artifacts),
                # NEW: inject playbook into editor
                editor_playbook=editor_playbook,
            )

            is_correct, score = sim.run(verbose=False, save_log=True)
            trace = sim.trace
            eval_summary = find_last_eval(trace)

            # Dataset-provided oracle fields (may be None)
            full_spec_q = sample.get("full_spec_q")
            ground_truth_a = sample.get("ground_truth_a")

            old_cheatsheet = mem.cheatsheet

            upd = mem.maybe_update(
                task_name=sample["task"],
                system_message=sim.system_message,
                trace=trace,
                eval_summary=eval_summary,
                full_spec_q=full_spec_q,
                ground_truth_a=ground_truth_a,
            )

            logger.log_revision(
                dataset_fn=args.dataset_file,
                split_name=args.split_name,
                mode=args.mode,
                sample=sample,
                models={
                    "assistant": assistant_model,
                    "system": args.system_model,
                    "user": args.user_model,
                    "editor": args.editor_model,
                    "curator": args.curator_model,
                },
                eval_summary=eval_summary,
                # curator_meta=upd.get("curator_meta"),
                curator_meta=upd.get("curator_meta")
                or {"error": upd.get("error"), "attempts": upd.get("curator_attempts")},
                cheatsheet_old=upd.get("old", old_cheatsheet),
                cheatsheet_new=upd.get("new", mem.cheatsheet),
                episode_idx=episode_idx,
                cheatsheet_scope=args.cheatsheet_scope,
                task_key=task_key,
            )

            total += 1
            if is_correct or (score == 1.0):
                correct += 1

            pbar.set_postfix({"task": sample["task"], "acc": f"{correct}/{total}"})

        print(
            f"[{assistant_model}] DONE split={args.split_name} mode={args.mode} acc={correct}/{total} = {correct / total:.3f}"
        )

        if args.save_final_cheatsheet_path:
            base, ext = os.path.splitext(args.save_final_cheatsheet_path)
            ext = ext or ".txt"
            os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

            if args.cheatsheet_scope == "global":
                out_path = f"{base}_{assistant_model}{ext}"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(memories["__global__"].cheatsheet)
                print(f"Saved final editor cheatsheet to {out_path}")
            else:
                for task, mem2 in memories.items():
                    out_path = f"{base}_{assistant_model}_{task}{ext}"
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(mem2.cheatsheet)
                    print(f"Saved final editor cheatsheet for {task} to {out_path}")


if __name__ == "__main__":
    main()
