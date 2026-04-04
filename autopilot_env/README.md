# AutoPilot-Env++

## Problem Statement
Modern digital assistants struggle with realistic, messy, and dynamic environments. They often fail at reasoning about multi-step tasks like managing threaded emails, distinguishing phishing from important alerts, extracting tasks, and prioritizing workflows. AutoPilot-Env++ bridges this gap by providing an OpenEnv-compliant simulation that evaluates AI agents on real-world assistant workflows using a custom RL system to encourage self-improvement.

## Real-world Relevance
The system mimics an employee's daily inbox and task management duties. By incorporating partial observability (hidden priorities) and time-based pressures, it evaluates how effectively an LLM-based agent can triage information, take appropriate actions, and align with desired organizational goals—crucial capabilities for autonomous enterprise assistants.

## Action / Observation Space
**Action Space**:
- `read_email` (target_id)
- `reply_email` (target_id, content)
- `archive_email` (target_id)
- `create_task` (content)
- `prioritize_tasks`
- `do_nothing`

**Observation Space**:
- `emails`: List of emails with metadata (ID, sender, subject, body, read status, archived status, thread ID).
- `tasks`: List of current tasks (ID, title, description, status).
- `current_time`: The simulated time of day.
- `last_action_result`: Feedback from the environment on the last executed action.

## Tasks Description
1. **triage_emails (Easy)**: The agent must archive unimportant newsletters and reply to urgent requests from a manager.
2. **extract_tasks (Medium)**: The agent must read a client email and extract two specific tasks (fixing a bug, updating documentation) into the task list.
3. **full_workflow (Hard)**: The agent deals with a mix of spam and critical alerts (password reset). It must archive the spam, create a task for the password reset, and prioritize all pending tasks.

## Reward Logic
The custom RL layer assigns rewards to encourage efficient and correct actions:
- Correct/helpful action (e.g., archiving spam): `+0.2` to `+0.4`
- Task completion (creating needed task): `+0.2`
- Incorrect action / Wrong target: `-0.2`
- Unnecessary action (archiving an already archived email): `-0.05`

The final score evaluates the target state against the current state, outputting `0.0` to `1.0`.

## Setup Steps
1. Clone the repository.
2. Create a `.env` file in the root directory with your `GROQ_API_KEY`.
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Ensure Docker is installed.

## How to Run
### Using Docker
1. Build the Docker image:
   ```bash
   docker build -t autopilot-env .
   ```
2. Run the FastAPI server:
   ```bash
   docker run -p 8000:8000 -e GROQ_API_KEY="your_key_here" autopilot-env
   ```

### Running Inference
In a separate terminal, install the Python requirements locally (or enter a bash session inside the docker container) and run:
```bash
pip install -r requirements.txt
python inference.py
```

## Baseline Results
- `triage_emails`: Agents generally succeed in 3-4 steps.
- `extract_tasks`: Agents succeed in 1-2 steps.
- `full_workflow`: Agents succeed in 3-5 steps, effectively ignoring spam while prioritizing tasks.
