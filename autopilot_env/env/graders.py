from typing import Dict, Any
import logging

def evaluate_task(task_name: str, state: Dict[str, Any], target_state: Dict[str, Any]) -> float:
    score = 0.0
    total_checks = 0

    if task_name == "triage_emails":
        # Check archived
        for email in state["emails"]:
            if email.id in target_state.get("archived_emails", []):
                total_checks += 1
                if email.is_archived:
                    score += 1.0

            # Check read/replied (assuming replied implies read for this simple grader)
            if email.id in target_state.get("replied_emails", []):
                total_checks += 1
                if email.is_read: # Agent should at least read it
                    score += 1.0

    elif task_name == "extract_tasks":
        target_tasks = target_state.get("created_tasks", [])
        total_checks = len(target_tasks)

        for target in target_tasks:
            keywords = target["keywords"]
            for task in state["tasks"]:
                text = (task.title + " " + task.description).lower()
                if all(kw.lower() in text for kw in keywords):
                    score += 1.0
                    break

    elif task_name == "full_workflow":
        # Check archived spam
        for email in state["emails"]:
            if email.id in target_state.get("archived_emails", []):
                total_checks += 1
                if email.is_archived:
                    score += 1.0

        # Check task created for password
        target_tasks = target_state.get("created_tasks", [])
        total_checks += len(target_tasks)
        for target in target_tasks:
            keywords = target["keywords"]
            for task in state["tasks"]:
                text = (task.title + " " + task.description).lower()
                if all(kw.lower() in text for kw in keywords):
                    score += 1.0
                    break

        # Check prioritization
        total_checks += 1
        tasks_with_priority = [t for t in state["tasks"] if t.priority is not None]
        if len(tasks_with_priority) > 0:
            score += 1.0

    if total_checks == 0:
        return 0.0

    return score / total_checks
