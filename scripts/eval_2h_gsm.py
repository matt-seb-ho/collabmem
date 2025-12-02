import asyncio
import json

from llmplus import GenerationConfig, LLMClient, Provider

from collabmem.compose_q import (
    BaselinePrediction,
    ComposedExample,
    extract_final_numeric_from_model_output,
)
from collabmem.constants import REPO_ROOT

BASELINE_OUT_FILE = REPO_ROOT / "outputs/gsm8k_baseline_predictions.json"
COMPOSED_OUT_FILE = REPO_ROOT / "outputs/composed_gsm8k_problems.json"
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
    # read in composed problems
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

    # generate predictions
    gen_cfg = GenerationConfig(max_tokens=4096)
    preds = await client.async_batch_generate(
        prompts=prompts,
        model=GPT5_MINI,
        gen_cfg=gen_cfg,
    )

    # parse outputs and check answers
    correct = 0
    q1_id2composed_correct = {}
    for ce, pred in zip(composed_examples, preds):
        if not pred:
            continue
        pred_answer = pred[0]
        if pred_answer is None:
            continue
        final_numeric = extract_final_numeric_from_model_output(pred_answer)
        if abs(final_numeric - ce.composed_answer_value) < 1e-6:
            correct += 1
            q1_id2composed_correct[ce.id1] = True

    accuracy = correct / len([ce for ce in composed_examples if ce is not None])
    print(f"Composed GSM8K accuracy: {accuracy:.4f}")

    # check first subproblem solutions
    q1_check_prompts = []
    q1_ids = []
    for ce, pred in zip(composed_examples, preds):
        if ce is None or pred[0] is None:
            continue
        q1_ids.append(ce.id1)
        q1_check_prompt = LM_CHECK_Q1_TEMPLATE.format(
            question=ce.composed_question,
            x=ce.answer1_value,
            solution=pred[0],
        )
        q1_check_prompts.append(q1_check_prompt)
    q1_check_preds = await client.async_batch_generate(
        prompts=q1_check_prompts,
        model=GPT5_MINI,
        gen_cfg=gen_cfg,
    )

    # check first subproblem accuracy
    q1_correct = 0
    q2_correct_given_correct_q1 = 0
    for q1_id, pred in zip(q1_ids, q1_check_preds):
        pred_answer = pred[0]
        if pred_answer is None:
            continue
        if "<answer>correct</answer>" in pred_answer:
            q1_correct += 1
            if q1_id2composed_correct.get(q1_id, False):
                q2_correct_given_correct_q1 += 1
    q1_accuracy = q1_correct / len(q1_check_prompts)
    print(f"First subproblem accuracy: {q1_accuracy:.4f}")
    q2_conditional_accuracy = q2_correct_given_correct_q1 / q1_correct
    print(
        f"Second subproblem accuracy given correct first subproblem: {q2_conditional_accuracy:.4f}"
    )

    # check first subproblem solo accuracy
    with open(BASELINE_OUT_FILE, "r") as f:
        solo_data = json.load(f)
    baseline_preds = {}
    for pred in solo_data["predictions"]:
        pred_obj = BaselinePrediction(**pred)
        baseline_preds[pred_obj.example_id] = pred_obj

    q1_solo_correct = 0
    for q1_id in q1_ids:
        q1_solo_pred = baseline_preds[q1_id]
        if abs(q1_solo_pred.pred_answer - q1_solo_pred.gold_answer) < 1e-6:
            q1_solo_correct += 1
    q1_solo_accuracy = q1_solo_correct / len(q1_ids)
    print(f"First subproblem solo accuracy: {q1_solo_accuracy:.4f}")


if __name__ == "__main__":
    asyncio.run(a_main())
