import json
from pathlib import Path

from azure_oai_setup import initialize_azure_oai_client
from tqdm import tqdm

CHECK_USER_SIM_ERROR_RATE = """\
# Context:
For a human + AI conversational agent research project, we simulated conversations between an AI assistant and a human user.

This work is about seeing how distributed a task description across a conversation affects the AI assistant's ability to complete the given task.

So the full process involves first taking a complete problem statement, and breaking it up into pieces (shards).
To simulate conversation, we use another AI model to play the role of the human user (this is the "user simulator").
The user simulator sees the full conversation history but it also sees the full set of shards.
At each turn, the user simulator responds to the AI assistant's last message by providing another shard. The simulator is free to choose any currently unprovided shard at each turn.
Moreover the user simulator is instructed to paraphrase the shard to make it a realistic conversational response to the assistant.

We observed that the user simulator sometimes makes mistakes and rephrases a shard in a way that changes its meaning and the correct answer of the problem, affecting downstream evaluation of the AI assistant.

Your task is to review a full conversation between the AI assistant and the user simulator and identify if the user simulator made any mistakes that would change the meaning/correct answer to the original problem statement.

# Example:
Question: How much did John pay for therapy in total?
All Shards:
- shard 1: John goes to therapy 2 times a week.
- shard 2: John pays $100 per hour for therapy.
- shard 3: John went to therapy for 6 weeks.
- shard 4: Each session lasted 2 hours.

The user simulator mistakenly paraphrased shard 2 as "John pays $100 per session for therapy."
- This is a mistake because in actuality John pays $200 per session (2 hours at $100/hour).

# Instructions:
Review the conversation and the original problem statement. Identify if the user simulator made any mistakes that would change the meaning/correct answer to the original problem.

## Output Format:
put yes/no inside <found_mistake></found_mistake> tags.
inside <mistake_details></mistake_details> tags, write a json array of objects, each object containing the correct shard, the user simulator's paraphrase, and an explanation of why it is a mistake.

# Your Input
## Problem Statement:
{problem_statement}
## Conversation:
{conversation}"""

# TGT_JSONL = Path("/home/v-homatthew/lost_in_conversation/logs/math/sharded/sharded_math_gpt-4o-mini_2024-07-18.jsonl")
TGT_JSONL = Path(
    "/home/v-homatthew/lost_in_conversation/temp/sharded_math_gpt-4o-mini.jsonl"
)
DATA_JSON = Path(
    "/home/v-homatthew/lost_in_conversation/data/sharded_instructions_600.json"
)

# read files
with TGT_JSONL.open("r") as f:
    logs = [json.loads(line) for line in f]
with DATA_JSON.open("r") as f:
    data = json.load(f)

data_by_task_id = {e["task_id"]: e for e in data}


# print(logs[0]["trace"])
# print(data[0]["shards"])

deployment_name = "gpt-5-mini_2025-08-07"
client = initialize_azure_oai_client()

results = {}
mistakes = []

for e in tqdm(logs):
    task_id = e["task_id"]
    datum = data_by_task_id[task_id]
    all_shards = json.dumps(datum["shards"], indent=2)
    convo = json.dumps(e["trace"], indent=2)
    prompt = CHECK_USER_SIM_ERROR_RATE.format(
        problem_statement=all_shards,
        conversation=convo,
    )
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    response_content = response.choices[0].message.content
    if "<found_mistake>yes</found_mistake>" in response_content:
        mistakes.append((task_id, response_content))
    results[task_id] = response_content

# save results
with Path("user_sim_error_rate_results.json").open("w") as f:
    json.dump(results, f, indent=2)

print(f"Found {len(mistakes)} mistakes out of {len(logs)} conversations.")
