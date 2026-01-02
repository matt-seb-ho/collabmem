from pathlib import Path

# __file__ is root/src/collabmem/constants.py
REPO_ROOT = Path(__file__).parent.parent.parent

# lost in conversation data file
LIC_DATA_PATH = REPO_ROOT / "src/lic/data/sharded_instructions_600.json"
LIC_PROMPT_DIRECTORY = REPO_ROOT / "src/lic/prompts"
LIC_EVAL_SUBSET_PATH = REPO_ROOT / "src/lic/data/sharded_eval_subset.json"
LIC_MINI_EVAL_PATH = REPO_ROOT / "src/lic/data/lic_mini_eval.json"
