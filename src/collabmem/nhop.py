import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from llmplus import GenerationConfig, LLMClient

from collabmem.compose_q import CodeRepresentation, QAExample, batch_generate_simple
from collabmem.execute_code import RunResult, run_snippet

CODE_REPR_VALIDATION_TEMPLATE = """\
# expecting a function here
{code}

result = {func_name}()
print(result)
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

Let X_i be the answer to Q_i:
Solve Q_i and use the value of X_i to solve Q_{{i+1}}. Explain your answer step by step.
Q1: [text of Q1]
Q2: [text of Q2, partially expressed in terms of X_1]
Q3: [text of Q3, partially expressed in terms of X_2]

You are NOT writing this full multi-hop quesiton.
- Rather, we are only giving you one hop at a time
- You receive X_i and the unedited text of Q_{{i+1}}
- You must rewrite the text of Q_{{i+1}} so that it depends on X_i
- Your output WILL be inserted into the overall multi-hop question as Q_{{i+1}}
- DO NOT REVEAL THE VALUE OF X_i IN THE QUESTION TEXT
  - This would defeat the purpose of multi-hop
  - Instead, refer to it symbolically as "X_i"

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

Current hop's value for i: {i}
In other words, rewrite Q_{{i+1}} to depend on X_{i}.

# **Produce your output below:**
"""


@dataclass
class MultiHopComposedExample:
    # All original question IDs in order (Q1, Q2, ..., Qn)
    ids: List[str]

    # Original question texts in order (unmodified)
    questions: List[str]

    # Original solution code strings in order; Q1 may be None
    original_codes: List[Optional[str]]

    # Original single-hop numeric answers (one per original question)
    answer_values_original: List[float]

    # Final composed multi-hop question text (header + Q1..Qn)
    composed_question: str

    # Final numeric answer after all hops (output of last solve_revised)
    composed_answer_value: float

    # Final composed code (the last solve_revised that returns the final answer)
    composed_code: str

    # All intermediate answers generated during composition:
    # [Q1 solo answer, X_2, X_3, ..., X_n]
    intermediate_answers: List[float]

    # Rewritten subquestions (Q2..Qn) in order
    rewritten_subquestions: List[str]

    # All intermediate solve_revised() implementations (one per hop, for Q2..Qn)
    revised_codes: List[str]


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

    chain_state = []
    max_hops = 0

    for q1, rest in chains:
        max_hops = max(max_hops, len(rest))

        state = {
            # Metadata for originals
            "ids": [q1.id],
            "questions": [q1.question],
            "original_codes": [None],  # we typically don't have code for Q1
            "answer_values_original": [q1.answer_value],
            # Per-hop composition state
            "rest": rest,  # [(q2, code2), (q3, code3), ...]
            "current_answer_value": q1.answer_value,  # X_i at this hop
            "intermediate_answers": [q1.answer_value],  # store X_1, X_2, ...
            # Outputs we accumulate
            "rewritten_subquestions": [],  # Q2..Qn
            "revised_codes": [],  # all solve_revised definitions, one per hop
            "failed": False,
            "final_composed_answer": q1.answer_value,  # will be updated
            "final_revised_code": None,
        }

        chain_state.append(state)

    # Process hop by hop: hop_idx = 0 corresponds to composing with q2, etc.
    for hop_idx in range(max_hops):
        prompts: List[str] = []
        active_chain_indices: List[int] = []

        for chain_idx, state in enumerate(chain_state):
            if state["failed"]:
                continue

            rest: List[Tuple[QAExample, CodeRepresentation]] = state["rest"]
            if hop_idx >= len(rest):
                # This chain has no more hops
                continue

            prev_a = state["current_answer_value"]
            q_k, code_k = rest[hop_idx]

            # Build prompt for this hop using only the previous numeric answer
            # and the current (unmodified) Q_{i+1} text + code.
            prompt = MULTIHOP_COMPOSITION_PROMPT_TEMPLATE_2.format(
                prev_a=prev_a,
                q_text=q_k.question,
                q_code=code_k.code,
                i=hop_idx + 1,
            )
            prompts.append(prompt)
            active_chain_indices.append(chain_idx)

            # Bookkeeping for originals (Q2..Qn)
            state["ids"].append(q_k.id)
            state["questions"].append(q_k.question)
            state["original_codes"].append(code_k.code)
            state["answer_values_original"].append(q_k.answer_value)

        if not prompts:
            # No more hops to process for any chain
            break

        # Batch-generate for this hop across all active chains
        outputs = batch_generate_simple(client, prompts, gen_cfg, model=model)

        # Parse, execute, and update state for each active chain
        for out, chain_idx in zip(outputs, active_chain_indices):
            state = chain_state[chain_idx]

            if state["failed"]:
                continue

            # Extract revised code block
            code_match = re.search(
                r"<revised_code>\s*(.*?)\s*</revised_code>",
                out,
                flags=re.DOTALL | re.IGNORECASE,
            )
            revised_code = code_match.group(1).strip() if code_match else ""

            # Extract rewritten subquestion for this hop (Q_{i+1})
            q_match = re.search(
                r"<rewritten_subquestion>\s*(.*?)\s*</rewritten_subquestion>",
                out,
                flags=re.DOTALL | re.IGNORECASE,
            )
            rewritten_subq = q_match.group(1).strip() if q_match else ""

            if not revised_code or not rewritten_subq:
                state["failed"] = True
                continue

            # Execute the revised code to obtain the next X (X_{i+1})
            execution_succeeded = False
            output_val: Optional[float] = None
            try:
                formatted_code = CODE_REPR_VALIDATION_TEMPLATE.format(
                    code=revised_code,
                    func_name="solve_revised",
                )
                run_res: RunResult = run_snippet(formatted_code)
                if run_res.returncode == 0:
                    # Expect final line / only line to be the numeric answer
                    output_val = float(run_res.stdout.strip())
                    execution_succeeded = True
            except Exception as e:
                print(
                    f"[multi-hop] Code execution failed (chain {chain_idx}, hop {hop_idx}): {e}"
                )
                execution_succeeded = False

            if not execution_succeeded or output_val is None:
                state["failed"] = True
                continue

            # Update state for next hop
            state["rewritten_subquestions"].append(rewritten_subq)
            state["revised_codes"].append(revised_code)

            state["current_answer_value"] = output_val
            state["intermediate_answers"].append(output_val)
            state["final_composed_answer"] = output_val
            state["final_revised_code"] = revised_code

    # Optional: save a JSON-debuggable view (without QAExample/CodeRepresentation objects)
    if debug:
        debug_view = []
        for s in chain_state:
            debug_view.append(
                {
                    "ids": s["ids"],
                    "questions": s["questions"],
                    "original_codes_present": [
                        c is not None for c in s["original_codes"]
                    ],
                    "answer_values_original": s["answer_values_original"],
                    "rewritten_subquestions": s["rewritten_subquestions"],
                    "revised_codes_count": len(s["revised_codes"]),
                    "intermediate_answers": s["intermediate_answers"],
                    "final_composed_answer": s["final_composed_answer"],
                    "failed": s["failed"],
                }
            )
        with open("nhop_compose_debug.json", "w") as f:
            json.dump(debug_view, f, indent=2)

    # Build final outputs
    results: List[Optional[MultiHopComposedExample]] = []

    for state in chain_state:
        # If we never successfully did any hop, or something failed, return None
        if state["failed"] or len(state["revised_codes"]) == 0:
            results.append(None)
            continue

        # Build final composed multi-hop question text from template:
        #   Let X_i be the answer to Q_i:
        #   Solve Q_i and use the value of X_i to solve Q_{i+1}. Explain your answer step by step.
        #   Q1: [original Q1]
        #   Q2: [rewritten Q2]
        #   ...
        header_lines = [
            "Let X_i be the answer to Q_i:",
            "Solve Q_i and use the value of X_i to solve Q_{i+1}. Explain your answer step by step.",
        ]

        full_lines: List[str] = header_lines[:]
        # Q1: original text
        full_lines.append(f"Q1: {state['questions'][0]}")

        # Q2..Qn: rewritten subquestions
        for idx, subq in enumerate(state["rewritten_subquestions"], start=2):
            full_lines.append(f"Q{idx}: {subq}")

        composed_question_text = "\n".join(full_lines)

        comp = MultiHopComposedExample(
            ids=state["ids"],
            questions=state["questions"],
            original_codes=state["original_codes"],
            answer_values_original=state["answer_values_original"],
            composed_question=composed_question_text,
            composed_answer_value=state["final_composed_answer"],
            composed_code=state["final_revised_code"],
            intermediate_answers=state["intermediate_answers"],
            rewritten_subquestions=state["rewritten_subquestions"],
            revised_codes=state["revised_codes"],
        )
        results.append(comp)

    return results
