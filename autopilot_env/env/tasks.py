from typing import List, Dict
from env.models import Email, Task

def get_task_config(task_name: str) -> Dict:
    if task_name == "triage_emails":
        return {
            "emails": [
                Email(id="e1", sender="boss@company.com", subject="Urgent: Q3 Report", body="Please send me the Q3 report by EOD.", thread_id="t1", importance="high"),
                Email(id="e2", sender="newsletter@tech.com", subject="Weekly Tech News", body="Here is your weekly digest...", thread_id="t2", importance="low"),
                Email(id="e3", sender="colleague@company.com", subject="Lunch?", body="Want to grab lunch at 12?", thread_id="t3", importance="medium"),
            ],
            "tasks": [],
            "target_state": {
                "archived_emails": ["e2"],
                "replied_emails": ["e1", "e3"]
            }
        }
    elif task_name == "extract_tasks":
        return {
            "emails": [
                Email(id="e4", sender="client@external.com", subject="Project Update", body="Hi, can you fix the login bug and update the documentation by tomorrow?", thread_id="t4", importance="high")
            ],
            "tasks": [],
            "target_state": {
                "created_tasks": [
                    {"keywords": ["login", "bug"]},
                    {"keywords": ["documentation", "update"]}
                ]
            }
        }
    elif task_name == "full_workflow":
        return {
            "emails": [
                Email(id="e5", sender="security@company.com", subject="Action Required: Password Reset", body="Your password expires in 2 hours. Please reset it.", thread_id="t5", importance="high"),
                Email(id="e6", sender="spam@phishing.com", subject="You won a lottery!", body="Click here to claim your $1M prize.", thread_id="t6", importance="low")
            ],
            "tasks": [
                Task(id="tsk1", title="Weekly Sync", description="Prepare slides for weekly sync", status="pending", deadline="15:00")
            ],
            "target_state": {
                "archived_emails": ["e6"],
                "created_tasks": [{"keywords": ["password", "reset"]}],
                "prioritized_tasks": True # Need to see tasks prioritized correctly
            }
        }
    else:
        raise ValueError(f"Unknown task: {task_name}")
