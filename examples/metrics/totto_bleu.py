# data2text_metric.py
from typing import Any, Dict, List, Optional

import sacrebleu

from collabllm.metric import BaseMetric, SingleTurnOrChatMetric


@SingleTurnOrChatMetric.register_metric("totto_bleu")
class Data2TextBLEUMetric(BaseMetric):
    """
    BLEU metric for Data2Text (LiC).

    Faithfully reproduces:
        sacrebleu.corpus_bleu(
            [prediction],
            [[ref] for ref in references]
        ) / 100.0
    """

    def score(
        self,
        prompt: str,
        groundtruth: str,
        completion: str,
        messages: Optional[List[Dict[str, str]]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        if completion is None:
            raise ValueError("Model completion must be provided.")

        references = metadata["references"]

        prediction = completion.strip()

        bleu = sacrebleu.corpus_bleu(
            [prediction],
            [[ref.strip()] for ref in references],
        )

        return bleu.score / 100.0
