import requests
from typing import Optional

API_BASE_URL = "http://localhost:8000"

def execute_action(action_type: str, target_id: Optional[str] = None, content: Optional[str] = None) -> str:
    payload = {
        "action_type": action_type,
        "target_id": target_id,
        "content": content
    }
    try:
        response = requests.post(f"{API_BASE_URL}/step", json=payload)
        response.raise_for_status()
        data = response.json()
        return f"Action executed. Last action result: {data['observation']['last_action_result']}. Reward: {data['reward']['score']}. Done: {data['done']}"
    except Exception as e:
        return f"Error executing action: {e}"

def read_email(target_id: str) -> str:
    """Reads an email with the given target_id."""
    return execute_action("read_email", target_id=target_id)

def reply_email(target_id: str, content: str) -> str:
    """Replies to an email with the given target_id and content."""
    return execute_action("reply_email", target_id=target_id, content=content)

def archive_email(target_id: str) -> str:
    """Archives an email with the given target_id."""
    return execute_action("archive_email", target_id=target_id)

def create_task(content: str) -> str:
    """Creates a new task with the given description content."""
    return execute_action("create_task", content=content)

def prioritize_tasks() -> str:
    """Prioritizes all pending tasks."""
    return execute_action("prioritize_tasks")

def do_nothing() -> str:
    """Takes no action for this step."""
    return execute_action("do_nothing")
