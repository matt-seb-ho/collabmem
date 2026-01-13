from pathlib import Path

# models
GPT_4O_MINI = "gpt-4o-mini_2024-07-18"
GPT_5_MINI = "gpt-5-mini_2025-08-07"

# paths
# current file (__file__): collabmem/src/lic/constants.py
# PROJECT_ROOT = Path(__file__).parent
LIC_ROOT = Path(__file__).parent
DATA_DIR = LIC_ROOT / "data"
LOGS_DIR = LIC_ROOT / "logs"

PROMPT_FILE_DIRECTORY = LIC_ROOT / "prompts"
PROMPT_FILE_PATHS = {
    "dc_cu_vanilla": PROMPT_FILE_DIRECTORY / "dc_cu_curator_prompt.txt",
    "dc_cu_lic": PROMPT_FILE_DIRECTORY / "lic_dc_curator_prompt.txt",
    "editor_cheatsheet": PROMPT_FILE_DIRECTORY / "lic_editor_dc_curator_prompt.txt",
    "cheatsheet_v2": PROMPT_FILE_DIRECTORY / "cheatsheet_prompt_v2.txt",
}
