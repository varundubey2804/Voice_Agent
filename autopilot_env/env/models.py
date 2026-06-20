from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    is_read: bool = False
    is_archived: bool = False
    thread_id: str
    importance: Optional[str] = None # hidden from agent initially

class Task(BaseModel):
    id: str
    title: str
    description: str
    status: str = "pending" # pending, completed
    deadline: Optional[str] = None
    priority: Optional[int] = None # 1 (high) to 3 (low)

class Observation(BaseModel):
    emails: List[Dict[str, Any]]
    tasks: List[Dict[str, Any]]
    current_time: str
    last_action_result: str
    rl_feedback: Optional[Dict[str, float]] = Field(default_factory=dict) # To bias future decisions

class Action(BaseModel):
    action_type: str
    target_id: Optional[str] = None
    content: Optional[str] = None

class Reward(BaseModel):
    score: float
    breakdown: Dict[str, float]
