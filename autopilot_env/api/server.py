from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from env.environment import AutoPilotEnv
from env.models import Action, Observation, Reward

app = FastAPI(title="AutoPilot-Env++ API")
env = AutoPilotEnv()

class ResetRequest(BaseModel):
    task_name: str = "triage_emails"

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]

@app.post("/reset", response_model=Observation)
def reset_env(req: ResetRequest):
    try:
        obs = env.reset(req.task_name)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=StepResponse)
def step_env(action: Action):
    try:
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state", response_model=Observation)
def get_state():
    return env.get_observation()
