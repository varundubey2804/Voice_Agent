oice-Enabled RAG Insurance Agent: Veena
Veena is a sophisticated, voice-enabled conversational AI that acts as a friendly insurance agent for ValuEnable Life Insurance.
Her primary mission: engage customers about their policy renewals — all in real-time, with Agentic RAG and a local Ollama LLM.

🚀 Overview
Veena is a local-first backend service that enables real-time voice conversations with:

Speech-to-text transcription

Reasoning & memory with LangChain ReAct Agent

Retrieval-Augmented Generation (RAG)

Persona-driven replies

Natural speech output

WebSocket frontend integration

✨ Core Features
🎧 Real-Time Voice Conversation – Captures microphone input, transcribes, and responds instantly.

🧠 Agentic AI Core – LangChain’s ReAct agent for reasoning, tool use, and memory.

📚 RAG Search – Access internal policy/customer data via FAISS vector store.

🔒 Fully Local – Runs on local Ollama instance (e.g., LLaMA 3).

🗣 Natural Speech Output – gTTS + Pygame for smooth voice playback.

🔌 WebSocket Backend – Connect from React, Vue, or any WebSocket client.

🛠 Architecture
mermaid
Copy
Edit
flowchart LR
    A[🎤 Microphone Input] --> B[ASR: Faster-Whisper]
    B --> C[LangChain Agent: Veena Persona]
    C -->|Needs Info| D[🔍 FAISS RAG Search]
    C -->|Generates Reply| E[Local Ollama LLM]
    E --> F[gTTS Text-to-Speech]
    F --> G[🔊 Audio Playback]
    C --> H[🔌 WebSocket Updates to Frontend]
📂 Project Structure
graphql
Copy
Edit
.
├── app.py                # Main backend app (audio loop + WebSocket server)
├── agentic_rag.py        # LangChain agent, RAG tool, memory, and persona
├── index_documents.py    # Build FAISS vector index from knowledge base
├── voice_service.py      # Text-to-speech + audio playback
├── Requestollama.py      # Test Ollama API connection
├── requirements.txt      # Python dependencies
├── rag_docs/             # Knowledge base text files
├── faiss_rag.index       # Generated FAISS vector store
└── .gitignore
⚙️ Setup & Installation
1️⃣ Prerequisites
Python 3.8+

Ollama installed & running locally

Working microphone

2️⃣ Installation Steps
bash
Copy
Edit
# Clone repo
git clone <repo_url>
cd <repo_folder>

# Create virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
3️⃣ Pull Ollama Models
bash
Copy
Edit
ollama pull llama3
ollama pull nomic-embed-text
4️⃣ Prepare Knowledge Base
bash
Copy
Edit
# Create rag_docs folder and add your .txt files
mkdir rag_docs
5️⃣ Build FAISS Vector Index
bash
Copy
Edit
python index_documents.py
6️⃣ Run Backend
bash
Copy
Edit
python app.py
✅ You should see:

arduino
Copy
Edit
Audio recording started...
WebSocket server running on ws://localhost:8765
🛠 Configuration
Component	File	Setting
LLM & Embeddings	agentic_rag.py	LLM_MODEL_NAME, EMBED_MODEL_NAME
RAG Chunking	index_documents.py	Chunk size & overlap
Persona Behavior	agentic_rag.py	Persona prompt
Audio Parameters	app.py	DEFAULT_CHUNK_LENGTH

🧪 Utility Script
Test Ollama Connection

bash
Copy
Edit
python Requestollama.py
📜 License
MIT License – free to use, modify, and distribute.

💡 Tip
To integrate with a custom frontend, connect to:

arduino
Copy
Edit
ws://localhost:8765
and listen for:

user_message

agent_response

speaking_started

speaking_ended

