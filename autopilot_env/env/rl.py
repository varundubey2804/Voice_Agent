import json
import os
from typing import Dict

# Save state memory in memory so it's not a build artifact locally unless we want to persist between docker runs.
# Since the requirements say "custom lightweight reward-learning system" and "Store in JSON or simple DB",
# we will use an in-memory dictionary here and write to a file during execution,
# but we shouldn't commit it.
RL_STORE_FILE = "rl_store.json"

class RLSystem:
    def __init__(self):
        self.q_table: Dict[str, Dict[str, float]] = self._load_store()

    def _load_store(self) -> Dict:
        if os.path.exists(RL_STORE_FILE):
            try:
                with open(RL_STORE_FILE, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_store(self):
        with open(RL_STORE_FILE, "w") as f:
            json.dump(self.q_table, f)

    def _get_state_key(self, task_name: str) -> str:
        return task_name

    def get_action_value(self, task_name: str, action_type: str) -> float:
        state_key = self._get_state_key(task_name)
        if state_key not in self.q_table:
            return 0.0
        return self.q_table[state_key].get(action_type, 0.0)

    def update(self, task_name: str, action_type: str, reward: float):
        state_key = self._get_state_key(task_name)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}

        current_val = self.q_table[state_key].get(action_type, 0.0)
        # Simple exponential moving average update
        alpha = 0.1
        self.q_table[state_key][action_type] = current_val + alpha * (reward - current_val)
        self._save_store()

    def calculate_step_reward(self, action_type: str, success: bool, is_unnecessary: bool) -> float:
        reward = 0.0
        if success:
            reward += 0.3 # +0.2 to +0.4 for correct action
            if action_type == "create_task":
                reward += 0.2 # extra for task completion implicitly
        else:
            reward -= 0.2

        if is_unnecessary:
            reward -= 0.05

        return max(-1.0, min(1.0, reward)) # clamp to [-1, 1] for individual steps, total will be normalized later if needed
