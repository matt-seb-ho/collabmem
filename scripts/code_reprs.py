import json

from llmplus import GenerationConfig, LLMClient, Provider

from collabmem.compose_q import (
    QAExample,
    generate_code_representations,
    compose_problems,
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

    # next, convert to code repr
    # start with one
    # problem = problems[0]
    # code_reprs = generate_code_representations(
    #     client=client,
    #     gen_cfg=GenerationConfig(max_tokens=2048),
    #     examples=[problem],
    #     model="gpt-5-mini-2025-08-07",
    # )
    # print(code_reprs[0].code)

    # test subset with compose task
    subset = problems[2:4]
    code_reprs = generate_code_representations(
        client=client,
        gen_cfg=GenerationConfig(max_tokens=2048),
        examples=subset,
        model=GPT5_MINI,
    )
    pairs_for_composition = [
        (problems[0], problems[2], code_reprs[0]),
        (problems[1], problems[3], code_reprs[1]),
    ]

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
    print(composed_problems)


if __name__ == "__main__":
    main()
