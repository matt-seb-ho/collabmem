# lic/run_cheatsheet_baseline.py
import json
import random
from collections import Counter

from lic.cheatsheet_memory import CheatsheetConfig, CheatsheetMemory

# from lic.simulator_sharded import ConversationSimulatorSharded
from lic.cheatsheet_sim_sharded import ConversationSimulatorSharded


def find_last_eval(trace):
    # Pull the most recent answer-evaluation log (if any)
    extracted_answer = None
    score = None
    is_correct = None
    for msg in reversed(trace):
        if msg["role"] == "log" and msg["content"].get("type") == "answer-evaluation":
            extracted_answer = msg["content"].get("exact_answer")
            score = msg["content"].get("score")
            is_correct = msg["content"].get("is_correct")
            break
    return {
        "extracted_answer": extracted_answer,
        "score": score,
        "is_correct": is_correct,
    }


def run_stateful_dc(
    samples,
    dataset_fn,
    assistant_model,
    system_model,
    user_model,
    cheatsheet_init,
    cheatsheet_update_template,
    curator_model,
    mode,  # "vanilla" | "warmup" | "frozen"
    assistant_temperature=1.0,
    user_temperature=1.0,
    log_folder="logs",
    reasoning_cls_override=None,
    seed=0,
    shuffle=True,
):
    cfg = CheatsheetConfig(
        enable_updates=(mode != "frozen"),
        include_eval_feedback=(mode == "warmup"),
        curator_model=curator_model,
        curator_temperature=0.0,
        curator_max_tokens=2000,
        cheatsheet_update_template=cheatsheet_update_template,
    )
    memory = CheatsheetMemory(cheatsheet_init, cfg)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)

    results = []
    for sample in samples:
        preamble = memory.build_assistant_preamble()

        sim = ConversationSimulatorSharded(
            sample=sample,
            assistant_model=assistant_model,
            system_model=system_model,
            user_model=user_model,
            assistant_temperature=assistant_temperature,
            user_temperature=user_temperature,
            dataset_fn=dataset_fn,
            log_folder=log_folder,
            reasoning_cls_override=reasoning_cls_override,
            assistant_preamble=preamble,
        )
        is_correct, score = sim.run(verbose=False, save_log=True)
        trace = sim.trace

        eval_summary = find_last_eval(trace)

        upd = memory.maybe_update(
            task_name=sample["task"],
            system_message=sim.system_message,
            trace=trace,
            eval_summary=eval_summary,
        )

        # log cheatsheet update into trace for auditability
        trace.append(
            {
                "role": "log",
                "content": {
                    "type": "cheatsheet_update",
                    "mode": mode,
                    "curator_model": curator_model,
                    "updated": upd["updated"],
                    "curator_cost_usd": upd.get("curator_cost_usd"),
                    "cheatsheet_old": upd.get("old"),
                    "cheatsheet_new": upd.get("new"),
                },
                "timestamp": upd.get("timestamp"),
            }
        )

        results.append(
            {
                "task_id": sample["task_id"],
                "task": sample["task"],
                "is_correct": is_correct,
                "score": score,
            }
        )

    return results, memory.cheatsheet
