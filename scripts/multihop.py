import json
import random

from dataclasses import asdict
from llmplus import GenerationConfig, LLMClient, Provider

from collabmem.compose_q import CodeRepresentation, QAExample
from collabmem.constants import REPO_ROOT
from collabmem.nhop import compose_multi_hop_problems

GPT5_MINI = "gpt-5-mini-2025-08-07"

INIT_OUT_FILE = REPO_ROOT / "outputs/gsm8k_baseline_predictions.json"

# setup OAI client
client = LLMClient(
    provider=Provider.OPENAI,
    dotenv_path=(REPO_ROOT / ".env"),
    cache_dir=(REPO_ROOT / ".llm_cache"),
)

# load selected problems file
with open(INIT_OUT_FILE, "r") as f:
    data = json.load(f)

with open(REPO_ROOT / "outputs/back_half_code_repr.json") as f:
    back_half_code_repr = json.load(f)

count = 0
rounding_free = []
for item in back_half_code_repr["code_representations"]:
    code = item["code"]
    # search for "//" double slash as integer division indicator in python
    if "//" in code:
        # print(code)
        count += 1
    else:
        rounding_free.append(item["example_id"])
print(f"Found {count} instances of '//' in code representations.")

rounding_free_q = [
    ex
    for ex in back_half_code_repr["code_representations"]
    if ex["example_id"] in rounding_free
]

cr_dict = {}
for ex in back_half_code_repr["code_representations"]:
    cr_dict[ex["example_id"]] = CodeRepresentation(**ex)

# load problems
problems = [QAExample(**prob) for prob in data["problems"]]
problem_dict = {prob.id: prob for prob in problems}
rf_dict = {idx: problem_dict[idx] for idx in rounding_free}
rf_probs = list(rf_dict.values())

random.seed(42)
# CHAIN_LEN = 10
CHAIN_LEN = 20
# random.sample(rf_probs, CHAIN_LEN)
chains = [
    (
        rf_probs[0],
        [(rf_probs[i], cr_dict[rf_probs[i].id]) for i in range(1, CHAIN_LEN)],
    ),
]

res = compose_multi_hop_problems(
    chains, client, GenerationConfig(max_tokens=4096), GPT5_MINI
)
# print(res)
tgt_out_path = REPO_ROOT / "outputs/multihop_compose_test_20.json"
with open(tgt_out_path, "w") as f:
    artifact = {"chains": [asdict(ri) if ri is not None else None for ri in res]}
    json.dump(artifact, f, indent=2)
    print(f"Saved multihop compose test to {tgt_out_path}")
