from .abg_coqa import AbgCoQA
from .bfcl import BFCLSingleTurnDataset
from .bigcodebench import BigCodeBench
from .gsm8k import GSM8K
from .lic_code import LiCCode
from .math_hard import MATH
from .medium import Medium

# ADD NEW DATASET BELOW
datasets_info = {
    "math-hard": {"task_desc": "question answering", "class": MATH},
    "abg-coqa": {"task_desc": "question answering", "class": AbgCoQA},
    "medium": {"task_desc": "document editing", "class": Medium},
    "bigcodebench": {"task_desc": "code generation", "class": BigCodeBench},
    # lost in conversation benchmark
    "gsm8k": {"task_desc": "question answering", "class": GSM8K},
    "lic-code": {"task_desc": "code generation", "class": LiCCode},
    "bfcl": {"task_desc": "action generation", "class": BFCLSingleTurnDataset},
}
