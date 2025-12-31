from pathlib import Path

# __file__ is root/src/collabmem/constants.py
REPO_ROOT = Path(__file__).parent.parent.parent

# lost in conversation data file
LIC_DATA_PATH = REPO_ROOT / "src/lic/data/sharded_instructions_600.json"
