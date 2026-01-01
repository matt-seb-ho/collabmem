import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from collabmem.compose_q import CodeRepresentation, QAExample, batch_generate_simple
from collabmem.execute_code import RunResult, run_snippet

# Reuse your existing COMPOSITION_PROMPT_TEMPLATE, run_snippet, CODE_REPR_VALIDATION_TEMPLATE,
# batch_generate_simple, LLMClient, GenerationConfig, QAExample, CodeRepresentation
from llmplus import GenerationConfig, LLMClient

CODE_REPR_VALIDATION_TEMPLATE = """\
# expecting a function here
{code}

result = {func_name}()
print(result)
"""

MULTIHOP_COMPOSITION_PROMPT_TEMPLATE = """\
# Overview
You are constructing a *multi-hop* math word problem.  
At this hop, you are given:
- A current problem **Q1**, which already has a **known numeric answer X**.
  *Q1 may already be a composition of previous problems.*  
- A new problem **Q2** whose solution is defined by code.

Your job is to compose Q1 and Q2 into a new multi-hop problem.

---

# Your Tasks

### **1. Rewrite Q2’s solution code**
You must:
- Replace exactly one constant value in Q2’s solution code with the numeric value **X**.
- Define a function called **`solve_revised`** that returns the new final numeric answer.

The returned value of `solve_revised()` will be executed to check correctness,  
so the code must:
- Be valid Python
- Use the literal numeric value of X
- Contain no explanations or comments inside `<revised_code>`

---

### **2. Generate the new composed question**
Write a new multi-hop question in this format:
```
Let X_i be the answer to Q_i:
Solve Q_i and use the value of X_i to solve Q_{{i+1}}. Explain your answer step by step.
Q1: [text of Q1]
Q2: [text of Q2, partially expressed in terms of X_1]
Q3: [text of Q3, partially expressed in terms of X_2]
````
Notes:
- use if the previous problem uses "X1", you use "X2" and so on (increment the X value, assume "X" without a value is "X0", but ideally start at "X0")

---
# Output Format (strict)

Your output MUST contain both sections:

<revised_code>
def solve_revised():
    ...
</revised_code>

<composed_question>
Let X_i be the answer to Q_i:
...
</composed_question>

Do NOT include any content outside these tags.

---

# Inputs for This Hop

Q1 text:
{q1}

Q1 numeric answer (X):
{a1}

Q2 text:
{q2}

Q2 solution code:
```python
{code2}
````

# **Produce your output below:**
"""

MULTIHOP_COMPOSITION_PROMPT_TEMPLATE_2 = """\
# Overview
You are constructing a *multi-hop* math word problem.  
At this hop, you are given:
- the answer X to the previous problem in the chain
- the current problem Q to be revised and added to the chain
    - its problem text
    - its current solution to Q expressed as code

Your job is to edit the problem text of Q such that it depends on the answer X from the previous problem.
Secondly, you must edit the solution code to replace exactly one constant value with the numeric value X,
the new final answer should be computed and returned by a function called `solve_revised`.

# Your Tasks
1. **Rewrite Q2’s solution code**
You must:
- Replace exactly one constant value in Q2’s solution code with the numeric value **X**.
- Define a function called **`solve_revised`** that returns the new final numeric answer.

The returned value of `solve_revised()` will be executed to check correctness,  
so the code must:
- Be valid Python
- Use the literal numeric value of X

2. **Generate the new subquestion for the overall composed question**
The overall multi-hop question should be in this format:
```
Let X_i be the answer to Q_i:
Solve Q_i and use the value of X_i to solve Q_{{i+1}}. Explain your answer step by step.
Q1: [text of Q1]
Q2: [text of Q2, partially expressed in terms of X_1]
Q3: [text of Q3, partially expressed in terms of X_2]
````
You are NOT writing this full multi-hop quesiton.
- Rather, we are only giving you one hop at a time
- You receive X_i and the unedited text of Q_{{i+1}}
- You must rewrite the text of Q_{{i+1}} so that it depends on X_i
- Your output WILL be inserted into the overall multi-hop question as Q_{{i+1}}

# Output Format (strict)
Your output MUST contain both sections:

<revised_code>
def solve_revised():
    ...
</revised_code>

<rewritten_subquestion>

</rewritten_subquestion>

Do NOT include any content outside these tags.

# Inputs for This Hop
Q_i numeric answer X_i:
{prev_a}

Q_{{i+1}} original text:
{q_text}

Q_{{i+1}} original solution code:
```python
{q_code}
```

# **Produce your output below:**
"""


@dataclass
class MultiHopComposedExample:
    # All original question IDs in order (Q1, Q2, ..., Qn)
    ids: List[str]

    # Original question texts in order
    questions: List[str]

    # Original question code solutions in order
    original_codes: List[str]

    # Original single-hop numeric answers (if you want them)
    answer_values_original: List[float]

    # Final composed multi-hop question text
    composed_question: str

    # Final numeric answer after all hops
    composed_answer_value: float

    # Final composed code (the last solve_revised that returns the final answer)
    composed_code: str

    # All intermediate answers generated during composition (one per hop)
    intermediate_answers: List[float]


def compose_multi_hop_problems(
    chains: List[Tuple[QAExample, List[Tuple[QAExample, CodeRepresentation]]]],
    client: LLMClient,
    gen_cfg: GenerationConfig,
    model: str = "gpt-5-mini-2025-08-07",
    debug: bool = False,
) -> List[Optional[MultiHopComposedExample]]:
    """
    Generalizes `compose_problems` to arbitrary n-hop.

    Args:
        chains:
            A list of chains. Each chain is:
              (q1, [(q2, code2), (q3, code3), ..., (q_n, code_n)])
            where:
              - q1 is the first question with a known numeric answer (q1.answer_value).
              - Each (q_k, code_k) is a QAExample + its solution code that we will
                compose with the current composed problem.
        client, gen_cfg, model:
            Same as in your existing `compose_problems`.

    Returns:
        List of MultiHopComposedExample or None (if some hop failed) for each chain.
    """

    num_chains = len(chains)
    if num_chains == 0:
        return []

    # Per-chain state that evolves over hops
    chain_state = []
    max_hops = 0

    for q1, rest in chains:
        # rest is a list of (q_k, code_k) for k>=2
        max_hops = max(max_hops, len(rest))
        chain_state.append(
            {
                "current_question_text": q1.question,  # starts as Q1's text
                "current_answer_value": q1.answer_value,  # X for the first composition
                "current_code": None,
                "questions": [q1.question],
                "ids": [q1.id],
                "answer_values_original": [q1.answer_value],
                "intermediate_answers": [
                    q1.answer_value
                ],  # Track all new intermediate answers
                "rest": rest,  # [(q2, code2), (q3, code3), ...]
                "failed": False,
                "final_composed_question": None,
                "final_composed_answer": None,
                "final_composed_code": None,
            }
        )

    # Process hop by hop: hop_idx = 0 corresponds to composing with q2, etc.
    for hop_idx in range(max_hops):
        # Build prompts for all chains that:
        # - have not failed so far
        # - have this hop available (len(rest) > hop_idx)
        prompts = []
        active_chain_indices = []

        for chain_idx, state in enumerate(chain_state):
            if state["failed"]:
                continue

            rest = state["rest"]
            if hop_idx >= len(rest):
                # This chain has no more hops; skip
                continue

            a_prev = state["current_answer_value"]

            q_k, code_k = rest[hop_idx]

            prompt = MULTIHOP_COMPOSITION_PROMPT_TEMPLATE.format(
                q1=q_prev_text,
                a1=a_prev,
                q2=q_k.question,
                code2=code_k.code,
            )
            prompts.append(prompt)
            active_chain_indices.append(chain_idx)

            # For bookkeeping
            state["questions"].append(q_k.question)
            state["ids"].append(q_k.id)
            state["answer_values_original"].append(q_k.answer_value)

        if not prompts:
            # No more hops to process for any chain
            break

        # Batch-generate for this hop across all active chains
        outputs = batch_generate_simple(client, prompts, gen_cfg, model=model)

        # Parse, execute, and update state for each active chain
        for out, chain_idx in zip(outputs, active_chain_indices):
            state = chain_state[chain_idx]

            # If a previous hop marked failure, skip
            if state["failed"]:
                continue

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

            if revised_code == "" or composed_question == "":
                # Parsing failed
                state["failed"] = True
                continue

            # Execute the revised code to obtain the new X
            execution_succeeded = False
            output_val = None
            try:
                formatted_code = CODE_REPR_VALIDATION_TEMPLATE.format(
                    code=revised_code, func_name="solve_revised"
                )
                run_res = run_snippet(formatted_code)
                if run_res.returncode == 0:
                    execution_succeeded = True
                    output_val = float(run_res.stdout.strip())
            except Exception as e:
                print(
                    f"[multi-hop] Code execution failed (chain {chain_idx}, hop {hop_idx}): {e}"
                )
                execution_succeeded = False

            if not execution_succeeded:
                state["failed"] = True
                continue

            # Update chain state for the next hop
            state["current_question_text"] = composed_question
            state["current_answer_value"] = output_val
            state["intermediate_answers"].append(
                output_val
            )  # Record intermediate answer
            state["final_composed_question"] = composed_question
            state["final_composed_answer"] = output_val
            state["final_composed_code"] = revised_code

    # save chain_state info for debugging
    if debug:
        with open("nhop_compose_debug.json", "w") as f:
            json.dump(chain_state, f, indent=2)

    # Build final outputs
    results: List[Optional[MultiHopComposedExample]] = []

    for state in chain_state:
        if state["failed"] or state["final_composed_question"] is None:
            results.append(None)
            continue

        comp = MultiHopComposedExample(
            ids=state["ids"],
            questions=state["questions"],
            answer_values_original=state["answer_values_original"],
            composed_question=state["final_composed_question"],
            composed_answer_value=state["final_composed_answer"],
            composed_code=state["final_composed_code"],
            intermediate_answers=state["intermediate_answers"],
        )
        results.append(comp)

    return results
