from pathlib import Path

# models
GPT_4O_MINI = "gpt-4o-mini_2024-07-18"
GPT_5_MINI = "gpt-5-mini_2025-08-07"

# paths
# current file (__file__): lost_in_conversation/constants.py
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
