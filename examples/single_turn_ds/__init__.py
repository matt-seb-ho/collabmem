from .abg_coqa import AbgCoQA
from .bigcodebench import BigCodeBench
from .math_hard import MATH
from .medium import Medium

# ADD NEW DATASET BELOW
datasets_info = {
    "math-hard": {"task_desc": "question answering", "class": MATH},
    "abg-coqa": {"task_desc": "question answering", "class": AbgCoQA},
    "medium": {"task_desc": "document editing", "class": Medium},
    "bigcodebench": {"task_desc": "code generation", "class": BigCodeBench},
}
