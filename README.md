Voice-Enabled RAG Insurance Agent: Veena
Veena is a sophisticated, voice-enabled conversational AI that acts as a friendly insurance agent for ValuEnable Life Insurance.
Her primary goal is to engage customers about policy renewals, offering real-time, human-like conversations using Agentic RAG and a local Ollama LLM.

1. Executive Overview
Veena is built as a local-first backend service for real-time voice interactions, combining:

Speech-to-text (ASR) with Whisper/Faster-Whisper

Agentic reasoning with LangChain ReAct Agent

RAG with FAISS vector store

Persona-driven responses

Text-to-speech (TTS) output

WebSocket connectivity for frontend integration

2. Core Features
🎙 Real-Time Voice Conversation
Captures microphone input, transcribes it, and responds instantly.

🧠 Agentic AI Core
Uses LangChain’s ReAct agent pattern for reasoning, tool usage, and memory retention.

📚 Retrieval-Augmented Generation (RAG)
Searches internal knowledge base (customer history, policy details) via FAISS for factual, context-aware responses.

🔒 100% Local & Private
Runs entirely on a local Ollama instance (e.g., LLaMA 3), no external API calls for core AI logic.

🗣 Natural Voice Output
Generates smooth speech with Google gTTS and plays it using Pygame.

🔌 WebSocket Backend
Allows any frontend (React, Vue, etc.) to connect to ws://localhost:8765 for live updates and audio exchange.

3. How It Works (Architecture)
Audio Capture
app.py listens continuously for microphone input using PyAudio.

Speech-to-Text (ASR)
Detected speech is transcribed with Faster-Whisper.

Agent Invocation
Transcribed text is passed to the LangChain AgentExecutor (agentic_rag.py).

Thought & Action (RAG Search)
If more information is needed, the agent calls the rag_search_transcripts tool to query FAISS vector store.

Response Generation
The local Ollama LLM (LLaMA 3) generates a persona-consistent reply as “Veena”.

Text-to-Speech Conversion
voice_service.py uses gTTS to convert the reply into an MP3 file.

Audio Playback
Pygame plays the generated speech.

WebSocket Communication
Status updates (user_message, agent_response, speaking_started, etc.) are broadcast to all connected frontends.

4. Project Structure
graphql
Copy
Edit
.
├── app.py                # Main backend app (audio loop + WebSocket server)
├── agentic_rag.py        # LangChain agent, RAG tool, memory, and "Veena" persona
├── index_documents.py    # Script to build FAISS vector index from knowledge base
├── voice_service.py      # Text-to-speech + audio playback
├── Requestollama.py      # Test Ollama API connectivity
├── requirements.txt      # Python dependencies
├── rag_docs/             # Knowledge base text files
├── faiss_rag.index       # Generated FAISS vector store
└── .gitignore            # Git ignore file
5. Setup & Installation
Prerequisites
Python 3.8+

Ollama installed & running locally

Working microphone

Steps
Clone Repository

bash
Copy
Edit
git clone <repo_url>
cd <repo_folder>
Create Virtual Environment

bash
Copy
Edit
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Pull Ollama Models

bash
Copy
Edit
ollama pull llama3
ollama pull nomic-embed-text
Prepare Knowledge Base

Create a rag_docs/ folder.

Place .txt documents with relevant info inside.

Build Vector Index

bash
Copy
Edit
python index_documents.py
Run Backend

bash
Copy
Edit
python app.py
You should see:

arduino
Copy
Edit
Audio recording started...
WebSocket server running on ws://localhost:8765
6. Configuration
LLM & Embedding Models
Change LLM_MODEL_NAME and EMBED_MODEL_NAME in agentic_rag.py.

RAG Chunk Size & Overlap
Adjust in index_documents.py.

Agent Persona
Modify Veena’s instructions & style in agentic_rag.py.

Audio Recording Parameters
Edit DEFAULT_CHUNK_LENGTH in app.py.

7. Utility Scripts
Requestollama.py
Simple test to confirm Ollama is running and accessible.
Run:

bash
Copy
Edit
python Requestollama.py
