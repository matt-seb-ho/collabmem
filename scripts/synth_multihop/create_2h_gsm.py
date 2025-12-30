import json
from dataclasses import asdict

from llmplus import GenerationConfig, LLMClient, Provider

from collabmem.compose_q import (
    QAExample,
    compose_problems,
    generate_code_representations,
)
from collabmem.constants import REPO_ROOT

INIT_OUT_FILE = REPO_ROOT / "outputs/gsm8k_baseline_predictions.json"
GPT5_MINI = "gpt-5-mini-2025-08-07"


def main():
    # load selected problems file
    with open(INIT_OUT_FILE, "r") as f:
        data = json.load(f)

    # load problems
    problems = [QAExample(**prob) for prob in data["problems"]]

    # setup OAI client
    client = LLMClient(
        provider=Provider.OPENAI,
        dotenv_path=(REPO_ROOT / ".env"),
        cache_dir=(REPO_ROOT / ".llm_cache"),
    )

    # pairing scheme
    # take len(problems) // 2
    # pair idx 0 with idx len // 2
    # pair idx 1 with idx len // 2 + 1
    # etc.

    # first generate code representations for back half
    midway = len(problems) // 2
    back_half = problems[midway:]
    code_reprs = generate_code_representations(
        client=client,
        gen_cfg=GenerationConfig(max_tokens=2048),
        examples=back_half,
        model=GPT5_MINI,
    )
    # save generated code representations
    out_code_repr_path = REPO_ROOT / "outputs/back_half_code_repr.json"
    with open(out_code_repr_path, "w") as f:
        artifact = {
            "problems": [asdict(prob) for prob in back_half],
            "code_representations": [asdict(cr) for cr in code_reprs],
        }
        json.dump(artifact, f, indent=2)
        print(f"Saved back half code representations to {out_code_repr_path}")

    # pair problems (and with correct code reprs)
    bad_repr_count = 0
    pairs_for_composition = []
    for i in range(midway):
        p1 = problems[i]
        p2 = problems[i + midway]
        cr = code_reprs[i]
        if cr.correct:
            pairs_for_composition.append((p1, p2, cr))
        else:
            bad_repr_count += 1
    print(f"Skipping {bad_repr_count} pairs due to incorrect code representations.")

    # NOTE: from early testing when compose was only in charge of the model inference
    # revised_code_q_pairs = compose_problems(
    #     q1_q2_code=pairs_for_composition,
    #     client=client,
    #     gen_cfg=GenerationConfig(max_tokens=2048),
    #     model=GPT5_MINI,
    # )
    # for rc, rq in revised_code_q_pairs:
    #     print("Revised Code Representation:")
    #     print(rc)
    #     print("Revised Question:")
    #     print(rq)
    #     print("-" * 40)
    # works!

    composed_problems = compose_problems(
        q1_q2_code=pairs_for_composition,
        client=client,
        gen_cfg=GenerationConfig(max_tokens=2048),
        model=GPT5_MINI,
    )
    # save composed problems
    out_composed_path = REPO_ROOT / "outputs/composed_gsm8k_problems.json"
    with open(out_composed_path, "w") as f:
        artifact = {
            "composed_examples": [
                asdict(ce) if ce is not None else None for ce in composed_problems
            ],
        }
        json.dump(artifact, f, indent=2)
        print(f"Saved composed problems to {out_composed_path}")

    # count and report how many were successfully composed
    successful_count = sum(1 for ce in composed_problems if ce is not None)
    print(f"Successfully composed {successful_count} problems.")


if __name__ == "__main__":
    main()
