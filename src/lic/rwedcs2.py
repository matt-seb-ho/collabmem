# run_edited_with_dual_cheatsheets.py
import argparse
import json
import os
import random
import traceback
from collections import Counter

import tqdm

from lic.cheatsheet_ce_shard_simulator import ConversationSimulatorShardedEdited
from lic.cheatsheet_logger import CheatsheetLogConfig, CheatsheetLogger
from lic.cheatsheet_memory import CheatsheetConfig, CheatsheetMemory
from lic.constants import PROMPT_FILE_PATHS
from lic.editor_cheatsheet_memory import EditorCheatsheetMemory


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def safe_tqdm_write(msg: str) -> None:
    # tqdm-safe printing; fallback to print if tqdm isn't ready
    try:
        tqdm.tqdm.write(msg)
    except Exception:
        print(msg)


def format_episode_id(sample: dict, episode_idx: int) -> str:
    tid = sample.get("task_id", sample.get("id", "unknown_task_id"))
    task = sample.get("task", "unknown_task")
    return f"ep={episode_idx} task_id={tid} task={task}"


def load_samples(
    dataset_fn: str, tasks: list[str], problem_limit: int, seed: int, shuffle: bool
):
    with open(dataset_fn, "r", encoding="utf-8") as f:
        samples = json.load(f)

    samples = [s for s in samples if s.get("task") in tasks]

    if problem_limit > 0:
        out = []
        per_task = Counter()
        for s in samples:
            t = s.get("task")
            if t is None:
                continue
            if per_task[t] < problem_limit:
                out.append(s)
                per_task[t] += 1
        samples = out

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)

    return samples


def find_last_eval(trace):
    # Be defensive about trace structure
    try:
        for msg in reversed(trace or []):
            if msg.get("role") == "log" and isinstance(msg.get("content"), dict):
                if msg["content"].get("type") == "answer-evaluation":
                    return {
                        "extracted_answer": msg["content"].get("exact_answer"),
                        "score": msg["content"].get("score"),
                        "is_correct": msg["content"].get("is_correct"),
                    }
    except Exception:
        pass

    return {"extracted_answer": None, "score": None, "is_correct": None}


def find_last_edited_state(trace):
    try:
        for msg in reversed(trace or []):
            if msg.get("role") == "log" and isinstance(msg.get("content"), dict):
                if msg["content"].get("type") == "context-editor":
                    return msg["content"].get("edited_state")
    except Exception:
        pass
    return None


def make_memories(
    initial_text: str, cfg: CheatsheetConfig, tasks: list[str], scope: str, cls
):
    if scope == "global":
        return {"__global__": cls(initial_text, cfg)}
    return {t: cls(initial_text, cfg) for t in tasks}


def pick_memory(memories: dict, scope: str, task_name: str):
    return memories["__global__"] if scope == "global" else memories[task_name]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_file", type=str, required=True)
    p.add_argument("--split_name", type=str, default="train")
    p.add_argument("--tasks", nargs="+", default=["math"])
    p.add_argument("--models", nargs="+", default=["gpt-4o-mini"])

    # temps / models
    p.add_argument("--assistant_temperature", type=float, default=1.0)
    p.add_argument("--user_temperature", type=float, default=1.0)
    p.add_argument("--system_model", type=str, default="gpt-4o-mini")
    p.add_argument("--user_model", type=str, default="gpt-4o-mini")

    # context editor knobs
    p.add_argument("--editor_model", type=str, default="gpt-4o-mini")
    p.add_argument("--editor_temperature", type=float, default=1.0)
    p.add_argument("--editor_max_tokens", type=int, default=8192)
    p.add_argument("--disable_editor", action="store_true")
    p.add_argument("--no_log_editor_artifacts", action="store_true")

    # cheatsheet update modes (solver + editor can be controlled independently)
    p.add_argument(
        "--solver_mode",
        type=str,
        choices=["vanilla", "warmup", "frozen"],
        required=True,
    )
    p.add_argument(
        "--editor_mode",
        type=str,
        choices=["vanilla", "warmup", "frozen"],
        required=True,
    )

    p.add_argument("--curator_model", type=str, default="gpt-4o-mini")

    p.add_argument(
        "--solver_update_prompt_path",
        type=str,
        default=PROMPT_FILE_PATHS["dc_cu_lic"],
        help="Cheatsheet update template for SOLVER",
    )
    p.add_argument(
        "--editor_update_prompt_path",
        type=str,
        default=PROMPT_FILE_PATHS["editor_cheatsheet"],
        help=(
            "Cheatsheet update template for EDITOR "
            "(can be same as solver, but recommended to make a tailored one later)"
        ),
    )

    p.add_argument("--init_solver_cheatsheet_path", type=str, default=None)
    p.add_argument("--init_editor_cheatsheet_path", type=str, default=None)

    p.add_argument("--save_final_solver_cheatsheet_path", type=str, default=None)
    p.add_argument("--save_final_editor_cheatsheet_path", type=str, default=None)

    # scopes
    p.add_argument(
        "--solver_cheatsheet_scope", choices=["global", "per_task"], default="per_task"
    )
    p.add_argument(
        "--editor_cheatsheet_scope", choices=["global", "per_task"], default="per_task"
    )

    # logging
    p.add_argument("--log_folder", type=str, default="logs_dual")
    p.add_argument("--run_name", type=str, default=None)

    p.add_argument("--problem_limit", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_shuffle", action="store_true")

    # reasoning override passthrough
    p.add_argument("--reasoning_cls_override", type=json.loads, default="{}")
    p.add_argument("--disable_solver_cheatsheet", action="store_true")
    p.add_argument("--disable_editor_cheatsheet", action="store_true")

    args = p.parse_args()

    # Normalize windows-style dataset path like your other runner
    dataset_fn = args.dataset_file
    if dataset_fn.startswith(".\\"):
        dataset_fn = dataset_fn[2:]
    dataset_fn = dataset_fn.replace("\\", "/")
    args.dataset_file = dataset_fn

    # Load templates / initial cheatsheets (defensive)
    try:
        solver_template = (
            read_text(args.solver_update_prompt_path)
            if args.solver_update_prompt_path
            else ""
        )
    except Exception:
        safe_tqdm_write(
            f"\033[91m[ERROR reading solver_update_prompt_path={args.solver_update_prompt_path}]\n"
            f"{traceback.format_exc()}\033[0m"
        )
        solver_template = ""

    try:
        editor_template = (
            read_text(args.editor_update_prompt_path)
            if args.editor_update_prompt_path
            else ""
        )
    except Exception:
        safe_tqdm_write(
            f"\033[91m[ERROR reading editor_update_prompt_path={args.editor_update_prompt_path}]\n"
            f"{traceback.format_exc()}\033[0m"
        )
        editor_template = ""

    try:
        init_solver = (
            read_text(args.init_solver_cheatsheet_path)
            if args.init_solver_cheatsheet_path
            else "(empty)"
        )
    except Exception:
        safe_tqdm_write(
            f"\033[91m[ERROR reading init_solver_cheatsheet_path={args.init_solver_cheatsheet_path}]\n"
            f"{traceback.format_exc()}\033[0m"
        )
        init_solver = "(empty)"

    try:
        init_editor = (
            read_text(args.init_editor_cheatsheet_path)
            if args.init_editor_cheatsheet_path
            else "(empty)"
        )
    except Exception:
        safe_tqdm_write(
            f"\033[91m[ERROR reading init_editor_cheatsheet_path={args.init_editor_cheatsheet_path}]\n"
            f"{traceback.format_exc()}\033[0m"
        )
        init_editor = "(empty)"

    if args.run_name is None:
        args.run_name = (
            f"dual_dc_solver-{args.solver_mode}_editor-{args.editor_mode}_"
            f"{'_'.join(args.tasks)}"
        )

    # two separate loggers so you can analyze revisions independently
    solver_logger = CheatsheetLogger(
        CheatsheetLogConfig(
            log_dir=os.path.join(args.log_folder, args.run_name, "solver_cheatsheet"),
            run_name=args.run_name + "_solver",
            save_full_text=False,
        )
    )
    editor_logger = CheatsheetLogger(
        CheatsheetLogConfig(
            log_dir=os.path.join(args.log_folder, args.run_name, "editor_cheatsheet"),
            run_name=args.run_name + "_editor",
            save_full_text=True,
        )
    )

    # Load dataset samples
    try:
        samples = load_samples(
            args.dataset_file,
            args.tasks,
            args.problem_limit,
            args.seed,
            shuffle=(not args.no_shuffle),
        )
    except Exception:
        raise RuntimeError(
            f"Failed to load dataset_file={args.dataset_file}\n{traceback.format_exc()}"
        )

    print(
        f"Loaded {len(samples)} samples from {args.dataset_file} (tasks={args.tasks})"
    )

    for assistant_model in args.models:
        # SOLVER cheatsheet config
        solver_cfg = CheatsheetConfig(
            enable_updates=(args.solver_mode != "frozen"),
            include_eval_feedback=(args.solver_mode == "warmup"),
            curator_model=args.curator_model,
            curator_temperature=(
                1.0 if args.curator_model.startswith("gpt-5") else 0.0
            ),
            curator_max_tokens=2000,
            cheatsheet_update_template=solver_template,
        )
        # EDITOR cheatsheet config
        editor_cfg = CheatsheetConfig(
            enable_updates=(args.editor_mode != "frozen"),
            include_eval_feedback=(args.editor_mode == "warmup"),
            curator_model=args.curator_model,
            curator_temperature=(
                1.0 if args.curator_model.startswith("gpt-5") else 0.0
            ),
            curator_max_tokens=2000,
            cheatsheet_update_template=editor_template,
        )

        solver_memories = make_memories(
            init_solver,
            solver_cfg,
            args.tasks,
            args.solver_cheatsheet_scope,
            CheatsheetMemory,
        )
        editor_memories = make_memories(
            init_editor,
            editor_cfg,
            args.tasks,
            args.editor_cheatsheet_scope,
            EditorCheatsheetMemory,
        )

        correct = 0
        total = 0

        pbar = tqdm.tqdm(
            enumerate(samples),
            total=len(samples),
            desc=f"[{assistant_model}] ed_cs ({args.split_name})",
            dynamic_ncols=True,
        )

        for episode_idx, sample in pbar:
            # Everything per-episode is wrapped so we can always proceed/skip
            task_name = sample.get("task", "unknown_task")

            try:
                solver_mem = pick_memory(
                    solver_memories, args.solver_cheatsheet_scope, task_name
                )
                editor_mem = pick_memory(
                    editor_memories, args.editor_cheatsheet_scope, task_name
                )
            except Exception:
                safe_tqdm_write(
                    f"\033[91m[ERROR pick_memory] {format_episode_id(sample, episode_idx)} "
                    f"model={assistant_model}\n{traceback.format_exc()}\033[0m"
                )
                continue

            assistant_preamble = None
            editor_preamble = None

            if not args.disable_solver_cheatsheet:
                try:
                    assistant_preamble = solver_mem.build_assistant_preamble()
                except Exception:
                    safe_tqdm_write(
                        f"\033[91m[WARN solver preamble] {format_episode_id(sample, episode_idx)} "
                        f"model={assistant_model}\n{traceback.format_exc()}\033[0m"
                    )
                    assistant_preamble = None

            if not args.disable_editor_cheatsheet:
                try:
                    editor_preamble = editor_mem.build_assistant_preamble()
                except Exception:
                    safe_tqdm_write(
                        f"\033[91m[WARN editor preamble] {format_episode_id(sample, episode_idx)} "
                        f"model={assistant_model}\n{traceback.format_exc()}\033[0m"
                    )
                    editor_preamble = None

            # Construct simulator (can fail if inputs invalid)
            try:
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
                    editor_model=args.editor_model,
                    editor_temperature=args.editor_temperature,
                    editor_max_tokens=args.editor_max_tokens,
                    enable_editor=(not args.disable_editor),
                    log_editor_artifacts=(not args.no_log_editor_artifacts),
                    assistant_preamble=assistant_preamble,
                    editor_preamble=editor_preamble,
                )
            except Exception:
                safe_tqdm_write(
                    f"\033[91m[ERROR sim ctor] {format_episode_id(sample, episode_idx)} "
                    f"model={assistant_model}\n{traceback.format_exc()}\033[0m"
                )
                continue

            # Run simulator (most common failure point)
            try:
                is_correct, score = sim.run(verbose=False, save_log=True)
            except Exception:
                safe_tqdm_write(
                    f"\033[91m[ERROR sim.run] {format_episode_id(sample, episode_idx)} "
                    f"model={assistant_model}\n{traceback.format_exc()}\033[0m"
                )
                continue

            trace = getattr(sim, "trace", None) or []
            eval_summary = find_last_eval(trace)
            last_edited_state = find_last_edited_state(trace)

            # Update SOLVER cheatsheet (guarded)
            if not args.disable_solver_cheatsheet:
                try:
                    old_solver = getattr(solver_mem, "cheatsheet", "(missing)")
                    upd_solver = solver_mem.maybe_update(
                        task_name=task_name,
                        system_message=getattr(sim, "system_message", None),
                        trace=trace,
                        eval_summary=eval_summary,
                    )

                    # Log revision (guarded)
                    try:
                        solver_logger.log_revision(
                            dataset_fn=args.dataset_file,
                            split_name=args.split_name,
                            mode=args.solver_mode,
                            sample=sample,
                            models={
                                "assistant": assistant_model,
                                "system": args.system_model,
                                "user": args.user_model,
                                "curator": args.curator_model,
                            },
                            eval_summary=eval_summary,
                            curator_meta=(
                                upd_solver.get("curator_meta")
                                if isinstance(upd_solver, dict)
                                else None
                            ),
                            cheatsheet_old=(
                                upd_solver.get("old", old_solver)
                                if isinstance(upd_solver, dict)
                                else old_solver
                            ),
                            cheatsheet_new=(
                                upd_solver.get("new", solver_mem.cheatsheet)
                                if isinstance(upd_solver, dict)
                                else solver_mem.cheatsheet
                            ),
                            episode_idx=episode_idx,
                            cheatsheet_scope=args.solver_cheatsheet_scope,
                        )
                    except Exception:
                        safe_tqdm_write(
                            f"\033[91m[WARN solver log_revision] {format_episode_id(sample, episode_idx)} "
                            f"model={assistant_model}\n{traceback.format_exc()}\033[0m"
                        )

                except Exception:
                    safe_tqdm_write(
                        f"\033[91m[ERROR solver update] {format_episode_id(sample, episode_idx)} "
                        f"model={assistant_model}\n{traceback.format_exc()}\033[0m"
                    )

            # Update EDITOR cheatsheet (guarded)
            try:
                old_editor = getattr(editor_mem, "cheatsheet", "(missing)")
                upd_editor = editor_mem.maybe_update_editor(
                    task_name=task_name,
                    system_message=getattr(sim, "system_message", None),
                    trace=trace,
                    eval_summary=eval_summary,
                    edited_state=last_edited_state,
                )

                try:
                    editor_logger.log_revision(
                        dataset_fn=args.dataset_file,
                        split_name=args.split_name,
                        mode=args.editor_mode,
                        sample=sample,
                        models={
                            "assistant": assistant_model,
                            "system": args.system_model,
                            "user": args.user_model,
                            "curator": args.curator_model,
                        },
                        eval_summary=eval_summary,
                        curator_meta=(
                            upd_editor.get("curator_meta")
                            if isinstance(upd_editor, dict)
                            else None
                        ),
                        cheatsheet_old=(
                            upd_editor.get("old", old_editor)
                            if isinstance(upd_editor, dict)
                            else old_editor
                        ),
                        cheatsheet_new=(
                            upd_editor.get("new", editor_mem.cheatsheet)
                            if isinstance(upd_editor, dict)
                            else editor_mem.cheatsheet
                        ),
                        episode_idx=episode_idx,
                        cheatsheet_scope=args.editor_cheatsheet_scope,
                    )
                except Exception:
                    safe_tqdm_write(
                        f"\033[91m[WARN editor log_revision] {format_episode_id(sample, episode_idx)} "
                        f"model={assistant_model}\n{traceback.format_exc()}\033[0m"
                    )

            except Exception:
                safe_tqdm_write(
                    f"\033[91m[ERROR editor update] {format_episode_id(sample, episode_idx)} "
                    f"model={assistant_model}\n{traceback.format_exc()}\033[0m"
                )

            # Accuracy bookkeeping (guarded)
            total += 1
            try:
                if is_correct or (score == 1.0):
                    correct += 1
            except Exception:
                pass

            try:
                pbar.set_postfix({"task": task_name, "acc": f"{correct}/{total}"})
            except Exception:
                pass

        print(
            f"[{assistant_model}] DONE split={args.split_name} acc={correct}/{total} = "
            f"{(correct / total) if total else 0.0:.3f}"
        )

        # Save finals (guarded so saving can't kill the run)
        try:
            if args.save_final_solver_cheatsheet_path:
                base, ext = os.path.splitext(args.save_final_solver_cheatsheet_path)
                ext = ext or ".txt"
                os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

                if args.solver_cheatsheet_scope == "global":
                    out_path = f"{base}_{assistant_model}{ext}"
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(solver_memories["__global__"].cheatsheet)
                    print(f"Saved final solver cheatsheet to {out_path}")
                else:
                    for task, mem in solver_memories.items():
                        out_path = f"{base}_{assistant_model}_{task}{ext}"
                        with open(out_path, "w", encoding="utf-8") as f:
                            f.write(mem.cheatsheet)
                        print(f"Saved final solver cheatsheet for {task} to {out_path}")
        except Exception:
            safe_tqdm_write(
                f"\033[91m[ERROR save_final_solver_cheatsheet] model={assistant_model}\n"
                f"{traceback.format_exc()}\033[0m"
            )

        try:
            if args.save_final_editor_cheatsheet_path:
                base, ext = os.path.splitext(args.save_final_editor_cheatsheet_path)
                ext = ext or ".txt"
                os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

                if args.editor_cheatsheet_scope == "global":
                    out_path = f"{base}_{assistant_model}{ext}"
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(editor_memories["__global__"].cheatsheet)
                    print(f"Saved final editor cheatsheet to {out_path}")
                else:
                    for task, mem in editor_memories.items():
                        out_path = f"{base}_{assistant_model}_{task}{ext}"
                        with open(out_path, "w", encoding="utf-8") as f:
                            f.write(mem.cheatsheet)
                        print(f"Saved final editor cheatsheet for {task} to {out_path}")
        except Exception:
            safe_tqdm_write(
                f"\033[91m[ERROR save_final_editor_cheatsheet] model={assistant_model}\n"
                f"{traceback.format_exc()}\033[0m"
            )


if __name__ == "__main__":
    main()
