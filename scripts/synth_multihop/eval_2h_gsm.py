import asyncio
import json

from collabmem.compose_q import (
    BaselinePrediction,
    ComposedExample,
    extract_final_numeric_from_model_output,
)
from llmplus import GenerationConfig, LLMClient, Provider

from collabmem.constants import REPO_ROOT

BASELINE_OUT_FILE = REPO_ROOT / "outputs/gsm8k_baseline_predictions.json"
COMPOSED_OUT_FILE = REPO_ROOT / "outputs/composed_gsm8k_problems.json"
COMPOSED_EVAL_OUT_FILE = REPO_ROOT / "outputs/composed_gsm8k_eval_artifacts.json"

GPT5_MINI = "gpt-5-mini-2025-08-07"

GSM_PROMPT_TEMPLATE = """\
You are a math word problem solver.
Solve the following problem step by step, then give the final numeric answer.
Output your final numeric answer inside <answer> </answer> tags.
Write only your numeric answer in the tags without unit labels (if the problem does not request specific units, use the same units as in the question).

Problem:
{question}
"""

LM_CHECK_Q1_TEMPLATE = """\
# Instructions
We provided this 2-part math word problem. 
- Part 1's solution we refer to as "X" is used in Part 2. 
- Please read the student's solution and determine if they correctly solved Part 1 to find X.
- Output <answer>correct</answer> if the student's solution for X is correct, otherwise output <answer>incorrect</answer>.
- Ignore Part 2 for this evaluation.

# Your Inputs
## Problem:
{question}

## Correct Value of X
X = {x}

## Student's Solution:
{solution}
"""

# setup OAI client
client = LLMClient(
    provider=Provider.OPENAI,
    dotenv_path=(REPO_ROOT / ".env"),
    cache_dir=(REPO_ROOT / ".llm_cache"),
)


async def a_main():
    # =========================
    # Load composed problems
    # =========================
    with open(COMPOSED_OUT_FILE, "r") as f:
        data = json.load(f)
    composed_examples = [
        ComposedExample(**ce) if ce is not None else None
        for ce in data["composed_examples"]
    ]

    # create prompts
    prompts = [
        GSM_PROMPT_TEMPLATE.format(question=ce.composed_question)
        if ce is not None
        else None
        for ce in composed_examples
    ]

    # =========================
    # Generate predictions
    # =========================
    gen_cfg = GenerationConfig(max_tokens=4096)
    preds = await client.async_batch_generate(
        prompts=prompts,
        model=GPT5_MINI,
        gen_cfg=gen_cfg,
    )

    # =========================
    # Parse outputs, composed (overall) correctness, bookkeeping
    # =========================
    total_non_null = len([ce for ce in composed_examples if ce is not None])
    overall_correct = 0

    # We’ll index everything by the position in composed_examples
    valid_indices = []  # indices where ce and model response are present
    model_responses = {}  # idx -> raw model text
    q2_overall_correct = {}  # idx -> bool, "q2 q correctness" (overall composed correctness)
    q1_id2composed_correct = {}  # retained for your original conditional stat

    # For Q1 checking
    q1_check_prompts = []
    q1_ids_for_check = []

    for idx, (ce, pred) in enumerate(zip(composed_examples, preds)):
        if ce is None or not pred or pred[0] is None:
            continue

        resp = pred[0]
        model_responses[idx] = resp
        valid_indices.append(idx)

        final_numeric = extract_final_numeric_from_model_output(resp)
        is_overall_correct = abs(final_numeric - ce.composed_answer_value) < 1e-6
        q2_overall_correct[idx] = is_overall_correct

        if is_overall_correct:
            overall_correct += 1
            # we keep your original per-q1 mapping for conditional accuracy
            q1_id2composed_correct[ce.id1] = True

        # Prepare LM check for Q1 correctness (composed trace)
        q1_ids_for_check.append(ce.id1)
        q1_check_prompt = LM_CHECK_Q1_TEMPLATE.format(
            question=ce.composed_question,
            x=ce.answer1_value,
            solution=resp,
        )
        q1_check_prompts.append(q1_check_prompt)

    accuracy = overall_correct / total_non_null if total_non_null > 0 else 0.0
    print(f"Composed GSM8K accuracy: {accuracy:.4f}")

    # =========================
    # Check first subproblem correctness in the composed trace
    # =========================
    q1_check_preds = await client.async_batch_generate(
        prompts=q1_check_prompts,
        model=GPT5_MINI,
        gen_cfg=gen_cfg,
    )

    q1_correct = 0
    q2_correct_given_correct_q1 = 0

    # idx in composed_examples -> bool: correctness of Q1 inside the composed solution
    q1_composed_correct = {}

    for local_idx, (q1_id, pred) in enumerate(zip(q1_ids_for_check, q1_check_preds)):
        pred_answer = pred[0]
        if pred_answer is None:
            continue
        is_q1_correct = "<answer>correct</answer>" in pred_answer
        # map back to global index
        global_idx = valid_indices[local_idx]
        q1_composed_correct[global_idx] = is_q1_correct

        if is_q1_correct:
            q1_correct += 1
            if q1_id2composed_correct.get(q1_id, False):
                q2_correct_given_correct_q1 += 1

    q1_accuracy = q1_correct / len(q1_check_prompts) if q1_check_prompts else 0.0
    print(f"First subproblem accuracy: {q1_accuracy:.4f}")
    q2_conditional_accuracy = (
        q2_correct_given_correct_q1 / q1_correct if q1_correct > 0 else 0.0
    )
    print(
        f"Second subproblem accuracy given correct first subproblem: {q2_conditional_accuracy:.4f}"
    )

    # =========================
    # Load baseline solo predictions
    # =========================
    with open(BASELINE_OUT_FILE, "r") as f:
        solo_data = json.load(f)
    baseline_preds = {}
    for pred in solo_data["predictions"]:
        pred_obj = BaselinePrediction(**pred)
        baseline_preds[pred_obj.example_id] = pred_obj

    # for original print of Q1 solo accuracy (on the subset we checked)
    q1_solo_correct = 0
    for q1_id in q1_ids_for_check:
        q1_solo_pred = baseline_preds[q1_id]
        if abs(q1_solo_pred.pred_answer - q1_solo_pred.gold_answer) < 1e-6:
            q1_solo_correct += 1
    q1_solo_accuracy = (
        q1_solo_correct / len(q1_ids_for_check) if q1_ids_for_check else 0.0
    )
    print(f"First subproblem solo accuracy: {q1_solo_accuracy:.4f}")

    # =========================
    # Build final per-composed-problem artifact
    # =========================
    artifacts = []

    for idx in valid_indices:
        ce = composed_examples[idx]
        if ce is None:
            continue

        # Solo correctness (baseline; Q1 and Q2)
        # Assumes baseline contains both id1 and id2.
        q1_solo = baseline_preds.get(ce.id1, None)
        q2_solo = baseline_preds.get(ce.id2, None)

        q1_solo_correct_bool = (
            abs(q1_solo.pred_answer - q1_solo.gold_answer) < 1e-6
            if q1_solo is not None
            else False
        )
        q2_solo_correct_bool = (
            abs(q2_solo.pred_answer - q2_solo.gold_answer) < 1e-6
            if q2_solo is not None
            else False
        )

        artifact_entry = {
            # Elements from ComposedExample
            "composed_question": ce.composed_question,
            "q1_id": ce.id1,
            "q2_id": ce.id2,
            "q1_answer": ce.answer1_value,
            "composed_answer": ce.composed_answer_value,
            # Predictions / evals
            "model_response": model_responses.get(idx, None),
            # solo correctness (baseline)
            "q1_solo_correct": q1_solo_correct_bool,
            "q2_solo_correct": q2_solo_correct_bool,
            # correctness in composed setting
            "q1_composed_correct": q1_composed_correct.get(idx, False),
            # "q2 q correctness" (overall composed correctness)
            "q2_composed_correct": q2_overall_correct.get(idx, False),
        }

        artifacts.append(artifact_entry)

    # =========================
    # Save final artifact file
    # =========================
    out_payload = {
        "composed_accuracy": accuracy,
        "q1_accuracy": q1_accuracy,
        "q2_conditional_accuracy": q2_conditional_accuracy,
        "q1_solo_accuracy": q1_solo_accuracy,
        "examples": artifacts,
    }

    with open(COMPOSED_EVAL_OUT_FILE, "w") as f:
        json.dump(out_payload, f, indent=2)

    print(f"Saved composed evaluation artifacts to: {COMPOSED_EVAL_OUT_FILE}")


if __name__ == "__main__":
    asyncio.run(a_main())
