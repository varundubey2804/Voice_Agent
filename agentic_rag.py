"""
agentic_rag.py ────────────────────────────────────────────────────────────
Self‑contained LangChain Agent + RAG pipeline
• Embeds docs with Ollama (nomic‑embed‑text  → 768‑d vectors)
• Stores/retrieves with FAISS (persisted on disk)
• Exposes a Tool for the agent to call
• Keeps conversational memory
"""

import os
import faiss
import dill
from pathlib import Path
from typing import List

from langchain_ollama import OllamaEmbeddings, ChatOllama  # ✅ updated here
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.tools import Tool


# ──────────────────────────────
# Configuration
# ──────────────────────────────
EMBED_MODEL_NAME = "nomic-embed-text"        # pull once: ollama pull nomic-embed-text
LLM_MODEL_NAME   = "llama3"                  # or mistral, phi3, etc.
FAISS_PATH       = "faiss_rag.index"         # persisted index file
DOCS_FOLDER      = "rag_docs"                # put your .txt docs here
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50


# ──────────────────────────────
# Helpers
# ──────────────────────────────
def _load_docs(folder: str) -> List[str]:
    """Return list of .txt file contents from folder (may be empty)."""
    docs = []
    for f in Path(folder).glob("*.txt"):
        docs.append(f.read_text(encoding="utf-8", errors="ignore"))
    if not docs:
        print(f"⚠️  No .txt files found in {folder}. RAG will still work, but only the LLM will answer.")
    return docs


def _build_or_load_vectorstore(texts: List[str], embedding: OllamaEmbeddings) -> FAISS:
    """First run: build FAISS; later runs: load with deserialization flag."""
    if Path(FAISS_PATH).exists():
        print(f"📂  Loading existing FAISS index: {FAISS_PATH}")
        return FAISS.load_local(
            FAISS_PATH,
            embeddings=embedding,
            allow_dangerous_deserialization=True,   # ✅ trust your own file
        )
    else:
        print("🦄  Building FAISS index … one‑time cost.")
        vectorstore = FAISS.from_texts(texts, embedding)
        vectorstore.save_local(FAISS_PATH)
        print(f"💾  Saved FAISS index → {FAISS_PATH}")
        return vectorstore


# ──────────────────────────────
# Public factory
# ──────────────────────────────
def build_agent():
    """Return a LangChain Agent with memory + RAG tool."""
    # 1) Embedding model + LLM
    embedding = OllamaEmbeddings(model=EMBED_MODEL_NAME)
    llm       = ChatOllama(model=LLM_MODEL_NAME)

    # 2) Load & split docs  → vector DB
    raw_docs = _load_docs(DOCS_FOLDER)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts    = splitter.split_text("\n".join(raw_docs)) if raw_docs else []
    vector_db = _build_or_load_vectorstore(texts, embedding)

    # 3) Make retriever tool
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    rag_tool  = Tool(
        name="rag_search_transcripts",
        func=lambda q: "\n".join([d.page_content for d in retriever.get_relevant_documents(q)]),
        description="Search internal knowledge base (FAISS) for facts.",
    )

    # 4) Memory + Agent
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent  = initialize_agent(
        tools=[rag_tool],
        llm=llm,
        agent="chat-conversational-react-description",
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
    )
    print("✅ Agentic RAG ready!")
    return agent


# ──────────────────────────────
# CLI quick test
# ──────────────────────────────
if __name__ == "__main__":
    ag = build_agent()
    while True:
        q = input("🗣  You: ")
        if q.lower() in {"exit", "quit"}:
            break
        print("🤖", ag.run(q))
