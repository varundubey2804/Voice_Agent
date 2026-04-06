import os
import json
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from agent.tools import read_email, reply_email, archive_email, create_task, prioritize_tasks, do_nothing

class ReActAgent:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant", # Using a fast model, adjust if needed
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=groq_api_key
        )

        self.tools = [
            Tool(name="read_email", func=read_email, description="Reads an email. Requires 'target_id'."),
            Tool(name="reply_email", func=lambda args: reply_email(*args.split(',', 1)) if ',' in args else reply_email(args, "Default reply"), description="Replies to an email. Pass arguments as 'target_id,content'."),
            Tool(name="archive_email", func=archive_email, description="Archives an email. Requires 'target_id'."),
            Tool(name="create_task", func=create_task, description="Creates a new task. Requires 'content'."),
            Tool(name="prioritize_tasks", func=lambda _: prioritize_tasks(), description="Prioritizes all pending tasks. Takes no arguments."),
            Tool(name="do_nothing", func=lambda _: do_nothing(), description="Takes no action. Takes no arguments.")
        ]

        # ReAct prompt template
        template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

        prompt = PromptTemplate.from_template(template)

        agent = create_react_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)

    def select_action(self, observation: dict) -> dict:
        # For our custom loop where we step one action at a time from outside,
        # using standard LangChain AgentExecutor will try to loop until Final Answer.
        # However, the user requires us to just output the JSON action or execute it.
        # Since the inference script expects an action dictionary to be returned
        # and then executed via the `/step` API natively,
        # we will prompt the LLM directly with a ReAct style reasoning constraint,
        # but ask it to format its *final output* as the required JSON.

        prompt = f"""
You are an AI assistant managing emails and tasks.
Current State:
{json.dumps(observation, indent=2)}

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
            res = self.llm.invoke(messages)
            text = res.content

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
            print(f"Failed to parse LLM output: {e}. Output was: {res.content if 'res' in locals() else 'None'}")
            return {"action_type": "do_nothing"}
