from typing import Dict, Any, Tuple
from datetime import datetime
import uuid

from env.models import Observation, Action, Email, Task, Reward
from env.tasks import get_task_config
from env.graders import evaluate_task
from env.rl import RLSystem

class AutoPilotEnv:
    def __init__(self):
        self.rl_system = RLSystem()
        self.reset("triage_emails") # Default task

    def reset(self, task_name: str) -> Observation:
        self.task_name = task_name
        config = get_task_config(task_name)
        self.emails = {e.id: e for e in config["emails"]}
        self.tasks = {t.id: t for t in config["tasks"]}
        self.target_state = config["target_state"]
        self.current_time = "09:00"
        self.last_action_result = "Environment reset."
        self.step_count = 0
        self.total_reward = 0.0
        return self.get_observation()

    def get_observation(self) -> Observation:
        # Hide importance from the observation to simulate partial observability
        obs_emails = []
        for e in self.emails.values():
            e_dict = e.dict()
            e_dict.pop('importance', None)
            obs_emails.append(e_dict)

        # Get RL feedback to bias agent
        state_key = self.rl_system._get_state_key(self.task_name)
        rl_feedback = self.rl_system.q_table.get(state_key, {})

        return Observation(
            emails=obs_emails,
            tasks=[t.dict() for t in self.tasks.values()],
            current_time=self.current_time,
            last_action_result=self.last_action_result,
            rl_feedback=rl_feedback
        )

    def _get_state_dict(self) -> Dict[str, Any]:
        return {
            "emails": list(self.emails.values()),
            "tasks": list(self.tasks.values())
        }

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self.step_count += 1
        success = False
        is_unnecessary = False
        msg = ""

        if action.action_type == "read_email":
            if action.target_id in self.emails:
                if not self.emails[action.target_id].is_read:
                    self.emails[action.target_id].is_read = True
                    success = True
                    msg = f"Email {action.target_id} marked as read."
                else:
                    is_unnecessary = True
                    msg = f"Email {action.target_id} is already read."
            else:
                msg = f"Email {action.target_id} not found."

        elif action.action_type == "reply_email":
            if action.target_id in self.emails:
                self.emails[action.target_id].is_read = True # Replying implies reading
                success = True
                msg = f"Replied to email {action.target_id} with content: {action.content}"
            else:
                msg = f"Email {action.target_id} not found."

        elif action.action_type == "archive_email":
            if action.target_id in self.emails:
                if not self.emails[action.target_id].is_archived:
                    self.emails[action.target_id].is_archived = True
                    success = True
                    msg = f"Archived email {action.target_id}."
                else:
                    is_unnecessary = True
                    msg = f"Email {action.target_id} is already archived."
            else:
                msg = f"Email {action.target_id} not found."

        elif action.action_type == "create_task":
            new_id = f"tsk_{uuid.uuid4().hex[:6]}"
            new_task = Task(id=new_id, title="Agent Created Task", description=action.content or "No description")
            self.tasks[new_id] = new_task
            success = True
            msg = f"Created task {new_id}."

        elif action.action_type == "prioritize_tasks":
            # For simplicity, assign random priorities or based on content if we were sophisticated.
            # Here we just mark them as prioritized.
            if self.tasks:
                for idx, t in enumerate(self.tasks.values()):
                    t.priority = idx % 3 + 1
                success = True
                msg = "Prioritized all tasks."
            else:
                is_unnecessary = True
                msg = "No tasks to prioritize."

        elif action.action_type == "do_nothing":
            success = True
            msg = "Did nothing."

        else:
            msg = f"Unknown action: {action.action_type}"

        self.last_action_result = msg

        # Calculate step reward and update RL
        step_reward = self.rl_system.calculate_step_reward(action.action_type, success, is_unnecessary)
        self.rl_system.update(self.task_name, action.action_type, step_reward)
        self.total_reward += step_reward

        # Check completion
        score = evaluate_task(self.task_name, self._get_state_dict(), self.target_state)
        done = score >= 1.0 or self.step_count >= 15 # Max steps

        # Normalize final reward if done
        final_score = score if done else 0.0

        reward_obj = Reward(
            score=final_score,
            breakdown={"step_reward": step_reward, "task_score": score}
        )

        return self.get_observation(), reward_obj, done, {"msg": msg, "success": success}
