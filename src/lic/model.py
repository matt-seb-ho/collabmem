# lic/model.py
import json
import os
import re
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from lic.azure_oai_setup import initialize_azure_oai_client


GLOBAL_REASONING_EFFORT = "low"


def is_reasoning_model(
    model_name: str, override: dict[str, bool] | None = None
) -> bool:
    # override behaviour
    if override is not None:
        for prefix, val in override.items():
            if prefix in model_name:
                return val

    # default behaviour
    return (
        "o1" in model_name
        or "o3" in model_name
        or "deepseek-r1" in model_name
        or "gpt-5" in model_name
    )


# ----------------------------
# Shared prompt variable replace
# ----------------------------
def format_messages(messages, variables={}):
    last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]

    for k, v in variables.items():
        key_string = f"[[{k}]]"
        if key_string not in last_user_msg["content"]:
            print(f"[prompt] Key {k} not found in prompt; effectively ignored")
        assert type(v) == str, f"[prompt] Variable {k} is not a string"
        last_user_msg["content"] = last_user_msg["content"].replace(key_string, v)

    keys_still_in_prompt = re.findall(r"\[\[([^\]]+)\]\]", last_user_msg["content"])
    if len(keys_still_in_prompt) > 0:
        print(f"[prompt] The following keys were not replaced: {keys_still_in_prompt}")

    return messages


# ----------------------------
# Provider: OpenAI/Azure (your existing behavior)
# ----------------------------
class OpenAIProvider:
    def __init__(self):
        load_dotenv(".env")
        if (
            "AZURE_OPENAI_API_KEY" in os.environ
            and "AZURE_OPENAI_ENDPOINT" in os.environ
        ):
            self.client = AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version="2024-10-01-preview",
            )
        elif os.environ.get("USE_TRAPI", "0") == "1":
            print("[lic/model.py] Using TRAPI Azure OpenAI Client")
            self.client = initialize_azure_oai_client()
        else:
            assert "OPENAI_API_KEY" in os.environ
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def cost_calculator(
        self, model: str, usage: Dict[str, Any], is_batch_model: bool = False
    ) -> float:
        is_finetuned, base_model = False, model
        if model.startswith("ft:gpt"):
            is_finetuned = True
            base_model = model.split(":")[1]

        prompt_tokens = usage["prompt_tokens"]
        if "prompt_tokens_details" in usage:
            prompt_tokens_cached = usage["prompt_tokens_details"]["cached_tokens"]
        else:
            prompt_tokens_cached = 0
        prompt_tokens_non_cached = prompt_tokens - prompt_tokens_cached

        completion_tokens = usage["completion_tokens"]
        if base_model.startswith("gpt-4o-mini"):
            if is_finetuned:
                inp_token_cost, out_token_cost = 0.0003, 0.00015
            else:
                inp_token_cost, out_token_cost = 0.00015, 0.0006
        elif base_model.startswith("gpt-4o"):
            if is_finetuned:
                inp_token_cost, out_token_cost = 0.00375, 0.015
            else:
                inp_token_cost, out_token_cost = 0.0025, 0.01
        elif base_model.startswith("gpt-3.5-turbo"):
            inp_token_cost, out_token_cost = 0.0005, 0.0015
        elif base_model.startswith("o1-mini"):
            inp_token_cost, out_token_cost = 0.003, 0.012
        elif base_model.startswith("gpt-4.5-preview"):
            inp_token_cost, out_token_cost = 0.075, 0.150
        elif base_model.startswith("o1-preview") or base_model == "o1":
            inp_token_cost, out_token_cost = 0.015, 0.06
        elif base_model.startswith("gpt-5-mini"):
            # inp_per_million = 0.25
            # out_per_million = 2.0
            inp_token_cost, out_token_cost = 0.00025, 0.002
        elif base_model.startswith("gpt-5.2"):
            # input per million: 1.75
            # output per million: 14.0
            inp_token_cost, out_token_cost = 0.00175, 0.014
        else:
            raise Exception(f"Model {model} pricing unknown, please add")

        cache_discount = 0.5  # cached tokens are half the price
        batch_discount = 0.5  # batch API is half the price
        total_usd = (
            (prompt_tokens_non_cached + prompt_tokens_cached * cache_discount) / 1000
        ) * inp_token_cost + (completion_tokens / 1000) * out_token_cost
        if is_batch_model:
            total_usd *= batch_discount

        return total_usd

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        timeout=30,
        max_retries=3,
        temperature=1.0,
        is_json=False,
        return_metadata=False,
        max_tokens=None,
        variables={},
    ):
        kwargs = {}
        if is_json:
            kwargs["response_format"] = {"type": "json_object"}

        messages = format_messages(messages, variables)

        # Preserve your o1 system-message workaround
        if (
            model.startswith("o1")
            and len(messages) > 1
            and messages[0]["role"] == "system"
            and messages[1]["role"] == "user"
        ):
            system_message = messages[0]["content"]
            messages[1]["content"] = (
                f"System Message: {system_message}\n{messages[1]['content']}"
            )
            messages = messages[1:]

        N = 0
        while True:
            try:
                if model.startswith("gpt-5"):
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        timeout=timeout,
                        max_completion_tokens=max_tokens,
                        temperature=temperature,
                        reasoning_effort=GLOBAL_REASONING_EFFORT,
                        **kwargs,
                    )
                    break
                else:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        timeout=timeout,
                        max_completion_tokens=max_tokens,
                        temperature=temperature,
                        **kwargs,
                    )
                    break
            except Exception:
                N += 1
                if N >= max_retries:
                    raise Exception("Failed to get response from OpenAI/Azure")
                time.sleep(4)

        response = response.to_dict()
        usage = response.get("usage", {}) or {}
        response_text = response["choices"][0]["message"]["content"]

        total_usd = 0.0
        if usage:
            total_usd = self.cost_calculator(model, usage)

        prompt_tokens_cached = 0
        if "prompt_tokens_details" in usage:
            prompt_tokens_cached = usage["prompt_tokens_details"].get(
                "cached_tokens", 0
            )

        if not return_metadata:
            return response_text

        return {
            "message": response_text,
            "total_tokens": usage.get("total_tokens", None),
            "prompt_tokens": usage.get("prompt_tokens", None),
            "prompt_tokens_cached": prompt_tokens_cached,
            "completion_tokens": usage.get("completion_tokens", None),
            "total_usd": total_usd,
            "provider": "openai",
        }


# ----------------------------
# Provider: vLLM via OpenAI-compatible server
# ----------------------------
class VLLMProvider:
    """
    Talks to a locally-hosted vLLM OpenAI-compatible server using the OpenAI python client.
    vLLM supports OpenAI Chat Completions style endpoints. :contentReference[oaicite:1]{index=1}
    """

    def __init__(self):
        load_dotenv(".env")
        base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = os.environ.get(
            "VLLM_API_KEY", "token-abc123"
        )  # can be anything if you set --api-key
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str,
        timeout=60,
        max_retries=3,
        temperature=1.0,
        is_json=False,
        return_metadata=False,
        max_tokens=None,
        variables={},
    ):
        # Strip routing prefix: local:my-llama -> my-llama
        served_model_name = model.split("local:", 1)[1]

        messages = format_messages(messages, variables)

        # vLLM supports chat completions; JSON mode is not universally supported.
        # We'll attempt it only if requested, but don’t rely on it.
        kwargs = {}
        if is_json:
            # Many open-source models won't obey strict JSON, but this is harmless as a hint.
            # vLLM may ignore response_format if unsupported.
            kwargs["response_format"] = {"type": "json_object"}

        N = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=served_model_name,
                    messages=messages,
                    timeout=timeout,
                    temperature=temperature,
                    max_tokens=max_tokens,  # OpenAI uses max_tokens; your OpenAI path uses max_completion_tokens
                    **kwargs,
                )
                break
            except Exception:
                N += 1
                if N >= max_retries:
                    raise Exception("Failed to get response from local vLLM server")
                time.sleep(2)

        response = response.to_dict()
        response_text = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {}) or {}

        if not return_metadata:
            return response_text

        # Cost is “local”, so set to 0; keep token accounting if returned.
        return {
            "message": response_text,
            "total_tokens": usage.get("total_tokens", None),
            "prompt_tokens": usage.get("prompt_tokens", None),
            "prompt_tokens_cached": 0,
            "completion_tokens": usage.get("completion_tokens", None),
            "total_usd": 0.0,
            "provider": "vllm",
        }


# ----------------------------
# Router
# ----------------------------
_openai_provider = OpenAIProvider()
_vllm_provider = VLLMProvider()


def generate(*args, model="gpt-4o-mini", **kwargs):
    if model.startswith("local:"):
        return _vllm_provider.generate(*args, model=model, **kwargs)
    return _openai_provider.generate(*args, model=model, **kwargs)


def generate_json(messages, model="gpt-4o-mini", **kwargs):
    out = generate(messages, model=model, is_json=True, **kwargs)
    out["message"] = json.loads(out["message"])
    return out
