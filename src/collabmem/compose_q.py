"""
compositional_dataset.py

Pipeline to construct a 2-hop compositional math dataset from
a base set of (question, answer) pairs, using an LLM client with
`async_batch_generate`.

This code is general: it works with arbitrary question/answer pairs
that have numeric answers, not just GSM8K.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from llmplus import GenerationConfig, LLMClient

from collabmem.execute_code import RunResult, run_snippet

# ---------- Data structures ----------


@dataclass
class QAExample:
    """Single question-answer example from any dataset."""

    id: int
    question: str
    answer_text: str  # original answer text (e.g. CoT)
    answer_value: Optional[float]  # parsed numeric final answer, if any


@dataclass
class BaselinePrediction:
    example_id: int
    gold_answer: Optional[float]
    pred_answer: Optional[float]
    raw_model_output: str


@dataclass
class CodeRepresentation:
    example_id: int
    question: str
    answer_value: Optional[float]
    code: str  # model-generated code representation
    raw_model_output: str  # raw LLM output that produced `code`
    execution_output: Optional[RunResult] = None  # filled in later
    correct: Optional[bool] = None  # filled in later


@dataclass
class ComposedExample:
    """Single composed (Q1, Q2) example."""

    id1: int
    id2: int
    question1: str
    question2: str
    answer1_value: Optional[float]
    answer2_value_original: Optional[float]

    composed_question: str  # natural language, like the example you gave
    composed_answer_value: Optional[
        float
    ]  # numeric final answer for the composed problem
    composed_code: str  # code representation for the composed problem
    # raw_model_output: str | None = None # raw LLM output for this composition step

    # Arbitrary metadata (e.g., prompts, etc.)
    meta: dict[str, Any] = field(default_factory=dict)


# ---------- Utilities ----------

_NUMERIC_REGEX = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def extract_final_numeric_from_answer_text(answer_text: str) -> Optional[float]:
    """
    Heuristic parser to extract the final numeric answer from a free-form
    answer string. Works reasonably well on GSM8K-style answers, but is
    intentionally generic.

    Strategy:
    1. Look for GSM8K-style "#### <number>".
    2. Otherwise, take the last numeric token in the string.
    """
    if answer_text is None:
        return None

    # Try GSM8K "#### <number>"
    m = re.search(r"####\s*(" + _NUMERIC_REGEX.pattern + r")", answer_text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass

    # Fallback: last numeric token in the string
    nums = _NUMERIC_REGEX.findall(answer_text.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def extract_final_numeric_from_model_output(text: str) -> Optional[float]:
    """
    Similar heuristic for model outputs: look for 'ANSWER:' first, then
    take the last numeric token.
    """
    if text is None:
        return None

    # prefer <answer>...</answer> tags if present
    answer_section = text
    pattern = r"<answer>\s*(" + _NUMERIC_REGEX.pattern + r")\s*</answer>"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    # m = re.search(r"ANSWER\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        answer_section = m.group(1)

    nums = _NUMERIC_REGEX.findall(answer_section.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


# ---------- LLM client helpers ----------


async def _async_batch_generate_simple(
    client: Any,
    prompts: list[str],
    gen_cfg: Any,
    model: str,
) -> list[str]:
    """
    Wrapper around your client's async_batch_generate.

    Expects:
      results: list[list[str]]  # outer: prompts, inner: generations per prompt
    Returns:
      list[str] of the first generation per prompt.
    """
    results = await client.async_batch_generate(
        prompts=list(prompts),
        gen_cfg=gen_cfg,
        model=model,
    )
    # results: list[list[str]]
    first_gens: list[str] = []
    for gens in results:
        if not gens:
            first_gens.append("")
        else:
            first_gens.append(gens[0])
    return first_gens


def batch_generate_simple(
    client: Any,
    prompts: list[str],
    gen_cfg: Any,
    model: str = "gpt-5-mini-2025-08-07",
) -> list[str]:
    """
    Synchronous convenience wrapper that runs the async version via asyncio.run().
    If you're already inside an event loop, call `_async_batch_generate_simple`
    directly instead.
    """
    return asyncio.run(_async_batch_generate_simple(client, prompts, gen_cfg, model))


# ---------- Step 1: Generic subsampling ----------


def subsample_qa_examples(
    questions: list[str],
    answers: list[str],
    n: int,
    seed: int = 0,
) -> list[QAExample]:
    """
    Subsample n question/answer pairs from arbitrary lists using a fixed seed.

    Returns a list of QAExample with numeric answer_value parsed.
    """
    assert len(questions) == len(answers), "questions and answers must align"
    total = len(questions)
    idxs = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    idxs = idxs[:n]

    examples: list[QAExample] = []
    for orig_idx in idxs:
        q = questions[orig_idx]
        a = answers[orig_idx]
        val = extract_final_numeric_from_answer_text(a)
        examples.append(
            QAExample(id=orig_idx, question=q, answer_text=a, answer_value=val)
        )
    return examples


# ---------- Step 2: Baseline evaluation ----------


def build_baseline_prompts(examples: list[QAExample]) -> list[str]:
    """Create prompts for baseline solving of the original (non-compositional) problems."""
    prompts: list[str] = []
    template = (
        "You are a math word problem solver.\n"
        "Solve the following problem step by step, then give the final numeric answer.\n"
        "Output your final numeric answer inside <answer> </answer> tags.\n"
        "Write only your numeric answer in the tags without unit labels (if the problem does not request specific units, use the same units as in the question).\n\n"
        "Problem:\n{question}\n"
    )
    for ex in examples:
        prompts.append(template.format(question=ex.question))
    return prompts


def evaluate_baseline_accuracy(
    client: Any,
    gen_cfg: Any,
    examples: list[QAExample],
    model: str = "gpt-5-mini-2025-08-07",
    # output_file: Path | None = "baseline_predictions.json",
    output_file: Path | None = None,
) -> tuple[list[BaselinePrediction], float]:
    """
    Run the model on the base questions to get baseline accuracy.

    Returns:
      - list of BaselinePrediction
      - accuracy (fraction of examples where parsed prediction == parsed gold)
    """
    prompts = build_baseline_prompts(examples)
    outputs = batch_generate_simple(client, prompts, gen_cfg, model=model)
    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(
                {
                    "prompts": prompts,
                    "outputs": outputs,
                },
                f,
            )
            print(f"Saved baseline prompts and outputs to {output_file}")

    preds: list[BaselinePrediction] = []
    correct = 0
    total = 0

    for ex, out in zip(examples, outputs):
        pred_val = extract_final_numeric_from_model_output(out)
        preds.append(
            BaselinePrediction(
                example_id=ex.id,
                gold_answer=ex.answer_value,
                pred_answer=pred_val,
                raw_model_output=out,
            )
        )
        if ex.answer_value is not None and pred_val is not None:
            total += 1
            if abs(ex.answer_value - pred_val) < 1e-6:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return preds, accuracy


# ---------- Step 3: Code representation generation ----------


def build_code_representation_prompts(examples: list[QAExample]) -> list[str]:
    """
    Prompts for converting a problem into a Python code representation.

    The expected model behavior:
      - Emit ONLY Python code, defining a function `def solve(X=None):`
      - If the problem uses an external value X (e.g., for composition),
        the function should accept an argument X (but we don't force it here).
    """
    prompts: list[str] = []
    template = (
        "You are converting math word problems into executable Python code.\n"
        "Given the problem and its known correct final numeric answer, write a single\n"
        "self-contained Python function of the form:\n\n"
        "def solve():\n"
        "    ...\n"
        "    return answer\n\n"
        "The function should compute the final numeric answer using only built-in\n"
        "arithmetic and variables. Do not import any libraries, and do not print.\n"
        "Define problem values as variables in the first part of the function.\n"
        "Problem:\n{question}\n\n"
        "Known correct final numeric answer: {answer_value}\n\n"
        "Now output ONLY the Python function code, nothing else."
    )
    for ex in examples:
        prompts.append(
            template.format(
                question=ex.question,
                answer_value=(
                    "unknown" if ex.answer_value is None else str(ex.answer_value)
                ),
            )
        )
    return prompts


CODE_REPR_VALIDATION_TEMPLATE = """\
# expecting a function here
{code}

result = {func_name}()
print(result)
"""


def generate_code_representations(
    client: Any,
    gen_cfg: Any,
    examples: list[QAExample],
    model: str = "gpt-5-mini-2025-08-07",
) -> list[CodeRepresentation]:
    """
    Use the LLM to generate code representations for each example.

    Returns a list of CodeRepresentation aligned with input examples.
    """
    prompts = build_code_representation_prompts(examples)
    outputs = batch_generate_simple(client, prompts, gen_cfg, model=model)

    code_reps: list[CodeRepresentation] = []
    for ex, out in zip(examples, outputs):
        # validate by running the code
        try:
            formatted_code = CODE_REPR_VALIDATION_TEMPLATE.format(
                code=out, func_name="solve"
            )
            run_res = run_snippet(formatted_code)
            # first check if it ran successfully
            correct = False
            if run_res.returncode == 0:
                # expect single print of numeric answer
                val = float(run_res.stdout.strip())
                if ex.answer_value is not None and abs(val - ex.answer_value) < 1e-6:
                    correct = True
        except Exception as e:
            print(f"Code execution failed for example {ex.id}: {e}")
            run_res = None
            correct = False

        code_reps.append(
            CodeRepresentation(
                example_id=ex.id,
                question=ex.question,
                answer_value=ex.answer_value,
                code=out.strip(),
                raw_model_output=out,
                execution_output=run_res,
                correct=correct,
            )
        )

    return code_reps


# ---------- Step 4: Pairing and compositional construction ----------


def pair_examples_sequentially(
    examples: list[QAExample],
) -> list[tuple[QAExample, QAExample]]:
    """
    Simple pairing strategy:
      (ex0, ex1), (ex2, ex3), ...
    """
    pairs: list[tuple[QAExample, QAExample]] = []
    for i in range(0, len(examples) - 1, 2):
        pairs.append((examples[i], examples[i + 1]))
    return pairs


COMPOSITION_PROMPT_TEMPLATE = """\
# Introduction
We are trying to compose two math word problems together, Q1, Q2.
- The idea is that Q1 produces a numeric answer X and then we replace one of the constant values in Q2 with X
- That way, the student has to solve Q1 first to get X, then use X to solve Q2.
- This obviously would change the answer to Q2, so we need to compute the new final numeric answer.
- Luckily we have Q2's solution expressed in code, so your job is just to modify that code to use X instead of the original constant.

# Composed Problem Example
Let X be the answer to the Q1:
Q1: There are 27 unicorns left in the world. One third of them are in the Scottish Highlands. Two thirds of the Scottish
unicorns are female. How many female Scottish unicorns are there?
Solve it and use the value of X to solve Q2. Explain your answer step by step.
Q2: Zack’s locker is half as big as Timothy’s locker. Peter’s locker is 1/4 as big as Zack’s locker. If Peter’s locker is X cubic
inches, how big is Timothy’s locker in cubic inches?

# Instructions
Given:
- Q1's text question 
- Q1's numerical answer (X) 
- Q2's text question
- Q2's solution code
Your tasks are:
1. rewrite Q2's solution code to use the literal constant value of X (we'll run the code to get the new correct answer)
2. write the new compositional question (in the style of the example) 

# Output Format
Write the revised code in <revised_code> </revised_code> tags (make sure to name the function `solve_revised`). Then write the new composed question in <composed_question> </composed_question> tags.

```
<revised_code>
def solve_revised():
    ...
</revised_code>
<composed_question>
Let X be the answer to Q1:
Q1: <text of Q1>
Solve it and use the value of X to solve Q2. Explain your answer step by step
Q2: <modified text of Q2 that uses X>
</composed_question>
```

# Your Turn
Q1 question: {q1}
Q1 answer value: {a1}
Q2 question: {q2}
Q2 solution code:
```python
{code2}
```"""


def compose_problems(
    q1_q2_code: list[tuple[QAExample, QAExample, CodeRepresentation]],
    client: LLMClient,
    gen_cfg: GenerationConfig,
    model: str = "gpt-5-mini-2025-08-07",
) -> list[ComposedExample | None]:
    prompts = []
    for q1, q2, code2 in q1_q2_code:
        prompt = COMPOSITION_PROMPT_TEMPLATE.format(
            q1=q1.question,
            a1=q1.answer_value,
            q2=q2.question,
            code2=code2.code,
        )
        prompts.append(prompt)

    outputs = batch_generate_simple(client, prompts, gen_cfg, model=model)
    parsed = []
    for out in outputs:
        # Extract revised code
        code_match = re.search(
            r"<revised_code>\s*(.*?)\s*</revised_code>",
            out,
            flags=re.DOTALL | re.IGNORECASE,
        )
        revised_code = code_match.group(1).strip() if code_match else ""

        # Extract composed question
        q_match = re.search(
            r"<composed_question>\s*(.*?)\s*</composed_question>",
            out,
            flags=re.DOTALL | re.IGNORECASE,
        )
        composed_question = q_match.group(1).strip() if q_match else ""
        parsed.append((revised_code, composed_question))

    # next is to execute the revised code to get the final answer
    composed_examples: list[ComposedExample | None] = []
    for (q1, q2, _), (revised_code, composed_question) in zip(q1_q2_code, parsed):
        if revised_code == "" or composed_question == "":
            # failed to parse
            composed_examples.append(None)
            continue

        execution_succeeeded = False
        output_val = None
        try:
            formatted_code = CODE_REPR_VALIDATION_TEMPLATE.format(
                code=revised_code, func_name="solve_revised"
            )
            run_res = run_snippet(formatted_code)
            if run_res.returncode == 0:
                execution_succeeeded = True
                output_val = float(run_res.stdout.strip())
        except Exception as e:
            print(f"Code execution failed for composed example: {e}")
            run_res = None
            execution_succeeeded = False

        if execution_succeeeded:
            composed_example = ComposedExample(
                id1=q1.id,
                id2=q2.id,
                question1=q1.question,
                question2=q2.question,
                answer1_value=q1.answer_value,
                answer2_value_original=q2.answer_value,
                composed_question=composed_question,
                composed_answer_value=output_val,
                composed_code=revised_code,
            )
            composed_examples.append(composed_example)
        else:
            # have to skip this one
            composed_examples.append(None)

    return composed_examples


# ---------- Step 5: Utility to convert to HF-style dicts ----------


def to_hf_dicts_for_composed_dataset(
    composed_examples: list[ComposedExample],
) -> list[dict[str, Any]]:
    """
    Convert ComposedExample into a simple list of dicts suitable for
    constructing a HuggingFace Dataset.

    Example fields:
      - composed_question
      - composed_answer_value
      - question1
      - question2
      - answer1_value
      - answer2_value_original
      - composed_code
    """
    rows: list[dict[str, Any]] = []
    for ex in composed_examples:
        row = {
            "composed_question": ex.composed_question,
            "composed_answer_value": ex.composed_answer_value,
            "question1": ex.question1,
            "question2": ex.question2,
            "answer1_value": ex.answer1_value,
            "answer2_value_original": ex.answer2_value_original,
            "composed_code": ex.composed_code,
            # You can add more metadata if you want:
            "meta": ex.meta,
        }
        rows.append(row)
    return rows
