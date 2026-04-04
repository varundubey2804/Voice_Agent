import os
import time
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3-8b-8192")

# OpenAI Client using Groq endpoint
openai_client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY", "dummy"),
    base_url="https://api.groq.com/openai/v1"
)

def reset_env(task_name: str):
    import requests
    res = requests.post(f"{API_BASE_URL}/reset", json={"task_name": task_name})
    res.raise_for_status()
    return res.json()

def step_env(action: dict):
    import requests
    res = requests.post(f"{API_BASE_URL}/step", json=action)
    res.raise_for_status()
    return res.json()

def select_action(obs: dict) -> dict:
    prompt = f"""
You are an AI assistant managing emails and tasks.
Current State:
{json.dumps(obs, indent=2)}

Think step-by-step about what to do.
Consider the RL recommendations provided in the observation to maximize reward.
Your goal is to clear out unneeded emails (archive), reply to actionable/important emails, and create tasks from requests.

Available actions:
- "read_email": target_id
- "reply_email": target_id, content
- "archive_email": target_id
- "create_task": content
- "prioritize_tasks"
- "do_nothing"

Please write out your reasoning starting with 'Thought:'.
Once you have decided on the action, output a JSON object containing exactly the keys: "action_type", "target_id" (can be null), "content" (can be null).
Example JSON format:
```json
{{"action_type": "archive_email", "target_id": "e2", "content": null}}
```

Ensure your response always ends with the JSON block.
"""
    messages = [{"role": "system", "content": prompt}]

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0
        )
        text = response.choices[0].message.content

        # Print the thought process for ReAct tracing
        print("\n--- Agent Thought Process ---")
        print(text)
        print("-----------------------------\n")

        # Extract JSON block
        json_str = "{}"
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            # sometimes it might not include the 'json' keyword
            json_blocks = text.split("```")
            if len(json_blocks) > 2:
                json_str = json_blocks[1].strip()
        else:
            # Try finding first { and last }
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                json_str = text[start:end+1]

        action_dict = json.loads(json_str)

        # Basic validation
        if action_dict.get("action_type") not in ["read_email", "reply_email", "archive_email", "create_task", "prioritize_tasks", "do_nothing"]:
            return {"action_type": "do_nothing"}

        return action_dict
    except Exception as e:
        print(f"Failed to get or parse action: {e}")
        return {"action_type": "do_nothing"}


def run_inference(task_name: str):
    print(f"[START] task={task_name} env=AutoPilotEnv++ model={MODEL_NAME}")

    # Reset Environment
    try:
        obs = reset_env(task_name)
    except Exception as e:
        print(f"Failed to reset environment: {e}")
        return

    done = False
    step_count = 0
    total_rewards = []
    final_score = 0.0

    while not done and step_count < 15:
        step_count += 1

        # Agent decides action based on observation
        action = select_action(obs)

        # Execute action in environment
        try:
            step_data = step_env(action)

            obs = step_data["observation"]
            reward_data = step_data["reward"]
            done = step_data["done"]

            step_reward = reward_data["breakdown"].get("step_reward", 0.0)
            final_score = reward_data["score"]
            total_rewards.append(step_reward)

            print(f"[STEP] step={step_count} action={action.get('action_type')} target={action.get('target_id')} reward={step_reward:.2f} done={str(done).lower()} error=null")

        except Exception as e:
            print(f"[STEP] step={step_count} action={action.get('action_type', 'unknown')} reward=0.00 done=false error=\"{e}\"")
            break

    success = done and final_score >= 1.0
    print(f"[END] success={str(success).lower()} steps={step_count} score={final_score:.2f} rewards={total_rewards}")

if __name__ == "__main__":
    # Ensure server is running before executing this
    import sys
    tasks_to_run = ["triage_emails", "extract_tasks", "full_workflow"]

    if len(sys.argv) > 1 and sys.argv[1] in tasks_to_run:
        run_inference(sys.argv[1])
    else:
        for task in tasks_to_run:
            run_inference(task)
            print("-" * 40)
            time.sleep(1) # Brief pause between tasks
