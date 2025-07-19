"""
agentic_rag.py ────────────────────────────────────────────────────────────
Self‑contained LangChain Agent + RAG pipeline
• Loads a pre-built FAISS index for document retrieval.
• Exposes a Tool for the agent to call.
• Keeps conversational memory and uses the 'Veena' persona.
"""

import os
from pathlib import Path
from typing import List

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

# --- Imports for a robust, modern agent ---
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor


# ──────────────────────────────
# Configuration
# ──────────────────────────────
EMBED_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME   = "llama3"
FAISS_PATH       = "faiss_rag.index"

# ──────────────────────────────
# Public factory
# ──────────────────────────────
def build_agent():
    """Return a LangChain Agent with memory + RAG tool."""
    # 1) Initialize models
    embedding = OllamaEmbeddings(model=EMBED_MODEL_NAME)
    llm       = ChatOllama(model=LLM_MODEL_NAME)

    # 2) Load the pre-built vector DB
    if not Path(FAISS_PATH).exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_PATH}'. "
            "Please run the 'index_documents.py' script first to create it."
        )
    print(f"📂  Loading existing FAISS index: {FAISS_PATH}")
    vector_db = FAISS.load_local(
        FAISS_PATH,
        embeddings=embedding,
        allow_dangerous_deserialization=True,
    )

    # 3) Create the retriever tool
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    rag_tool  = Tool(
        name="rag_search_transcripts",
        func=lambda q: "\n".join([d.page_content for d in retriever.invoke(q)]),
        description="Search internal knowledge base (FAISS) for facts about customer history and policies.",
    )
    tools = [rag_tool]

    # 4) Define the agent's persona and instructions
    persona = """You are "Veena," a female insurance agent for "ValuEnable life insurance".
Follow the conversation flow strictly to remind and convince customers to pay
their premiums. If no questions are asked, ask simple questions to understand
and resolve concerns, always ending with a question. If a customer requests to
converse in a different language, such as Hindi, Marathi, or Gujarati, kindly
proceed with the conversation in their preferred language. Use max 35 easy
english words to respond."""

    # 5) Create the prompt template for ReAct agent (using PromptTemplate, not ChatPromptTemplate)
    # This is the correct format for create_react_agent
    template = persona + """

TOOLS:
------
You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: your final response to the human here
```

Previous conversation history:
{chat_history}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    # Create the prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names"]
    )

    # 6) Create the agent
    agent = create_react_agent(llm, tools, prompt)

    # 7) Create memory - using simple ConversationBufferMemory
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=False,  # Return as string, not messages
        input_key="input",
        output_key="output"
    )

    # 8) Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="force"
    )
    
    print("✅ Agentic RAG with 'Veena' persona is ready!")
    return agent_executor


# ──────────────────────────────
# CLI quick test
# ──────────────────────────────
if __name__ == "__main__":
    ag = build_agent()
    while True:
        q = input("🗣  You: ")
        if q.lower() in {"exit", "quit"}:
            break
        # Correctly invoke the agent and access the output
        try:
            response = ag.invoke({"input": q})
            print("🤖 Veena:", response['output'])
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🤖 Veena: I apologize, I'm having trouble responding right now.")