import json
from dataclasses import asdict

from datasets import load_dataset
from llmplus import GenerationConfig, LLMClient, Provider

from collabmem.compose_q import (
    evaluate_baseline_accuracy,
    subsample_qa_examples,
)
from collabmem.constants import REPO_ROOT

GPT5_MINI = "gpt-5-mini-2025-08-07"


def main():
    # setup OAI client
    client = LLMClient(
        provider=Provider.OPENAI,
        dotenv_path=(REPO_ROOT / ".env"),
        cache_dir=(REPO_ROOT / ".llm_cache"),
    )

    gsm8k = load_dataset("gsm8k", "main", split="train")

    qa_examples = subsample_qa_examples(
        questions=list(gsm8k["question"]),
        answers=list(gsm8k["answer"]),
        n=200,
        seed=42,
    )

    preds, score = evaluate_baseline_accuracy(
        client=client,
        gen_cfg=GenerationConfig(max_tokens=2048),
        examples=qa_examples,
        model=GPT5_MINI,
    )
    print(f"score: {score:.4f}")

    # save preds to file
    output_path = REPO_ROOT / "outputs" / "gsm8k_baseline_predictions.json"
    with open(output_path, "w") as f:
        output_artifact = {
            "problems": [asdict(prob) for prob in qa_examples],
            "predictions": [asdict(pred) for pred in preds],
            "accuracy": score,
        }
        json.dump(output_artifact, f, indent=2)
        print(f"Saved baseline predictions to {output_path}")


if __name__ == "__main__":
    main()
