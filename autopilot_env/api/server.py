from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any
import os

from env.environment import AutoPilotEnv
from env.models import Action, Observation, Reward, Email
from env.email_client import fetch_recent_emails

app = FastAPI(title="AutoPilot-Env++ API")

# Serve static files for frontend
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_dashboard():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found"}
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

class IngestEmailRequest(BaseModel):
    username: str
    password: str
    server: str = "imap.gmail.com"
    count: int = 5

@app.post("/ingest-real-emails")
def ingest_real_emails(req: IngestEmailRequest):
    try:
        real_emails = fetch_recent_emails(req.username, req.password, req.server, req.count)

        # Inject into environment
        for e_dict in real_emails:
            new_email = Email(**e_dict)
            env.emails[new_email.id] = new_email

        env.last_action_result = f"Ingested {len(real_emails)} real emails from {req.username}."

        return {"status": "success", "ingested_count": len(real_emails)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
