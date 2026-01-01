from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI

# --------------------
# default values
# --------------------
SCOPE = "api://trapi/.default"
# Ensure this is a valid API version see: https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release
API_VERSION = "2024-12-01-preview"
# See https://aka.ms/trapi/models for the instance name
INSTANCE = "gcr/shared"
ENDPOINT = f"https://trapi.research.microsoft.com/{INSTANCE}"


def initialize_azure_oai_client(
    scope: str = SCOPE,
    endpoint: str = ENDPOINT,
    api_version: str = API_VERSION,
) -> AzureOpenAI:
    """Initialize and return an AzureOpenAI client."""
    # Authenticate by trying az login first, then a managed identity, if one exists on the system)
    credential = get_bearer_token_provider(
        ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        ),
        scope,
    )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=credential,
        api_version=api_version,
    )


def example_usage():
    client = initialize_azure_oai_client()

    # Ensure this is a valid deployment name see https://aka.ms/trapi/models for the deployment name
    # deployment_name = 'gpt-4o_2024-11-20'
    deployment_name = "gpt-5-mini_2025-08-07"

    # Do a chat completion and capture the response
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "user",
                "content": "Give a one word answer, what is the capital of France?",
            },
        ],
    )

    # Parse out the message and print
    response_content = response.choices[0].message.content
    print(response_content)


if __name__ == "__main__":
    example_usage()
