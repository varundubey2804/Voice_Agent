Voice-Enabled RAG Insurance Agent: "Veena"
This project implements Veena, a sophisticated, voice-enabled conversational AI. Veena assumes the persona of an insurance agent for "ValuEnable Life Insurance," with the primary goal of engaging customers about their policy renewals.

This application functions as a powerful backend service that handles real-time audio processing, agentic thinking with RAG, and voice synthesis. It operates a WebSocket server, allowing any compatible frontend (e.g., a web application) to connect and interact with Veena.

 Core Features
Real-time Voice Conversation: Captures microphone input using PyAudio and performs fast, accurate speech-to-text transcription with faster-whisper.

Agentic AI Core: Utilizes the LangChain framework to create a ReAct agent. This allows Veena to reason, use tools, and maintain conversational memory.

Retrieval-Augmented Generation (RAG): Veena can access a knowledge base of internal documents (e.g., customer history, policy details) using a FAISS vector store. This ensures responses are factual and context-aware.

Local & Private: Powered entirely by a local Ollama instance (using models like Llama 3), ensuring complete data privacy and no reliance on external APIs for core AI logic.

Text-to-Speech (TTS): The agent's responses are converted back to natural-sounding speech using Google's Text-to-Speech (gTTS) service.

WebSocket Backend: Runs a websockets server, making it easy to integrate with modern web frontends or other client applications for a seamless user experience.

 How It Works (Architecture)
The application follows a clear, sequential process for handling user interaction:

Audio Capture: The audio_recording_loop in app.py continuously listens for audio from the microphone using PyAudio.

Speech-to-Text: Once a non-silent chunk of audio is detected, it's sent to the WhisperModel for transcription into text.

Agent Invocation: The transcribed text is passed as input to the LangChain AgentExecutor (agentic_rag.py).

Thought & Action (RAG): The agent decides if it needs more information. If so, it uses its rag_search_transcripts tool to query the FAISS vector store for relevant documents.

Response Generation: With the necessary context, the agent prompts the local Ollama LLM (llama3) to generate a final, human-like response based on its "Veena" persona.

Text-to-Speech: The generated text response is passed to the voice_service.py, which uses gTTS to create an MP3 audio file.

Audio Playback: Pygame is used to play the generated audio file back to the user.

WebSocket Communication: Throughout the process, status updates (e.g., "user_message", "agent_response", "speaking_started") are broadcast to all connected frontend clients via the WebSocket server.

 Project Structure
.
├── app.py                  # Main application: handles audio, websockets, and glues all services together.
├── agentic_rag.py          # Defines the LangChain agent, RAG tool, memory, and "Veena" persona.
├── index_documents.py      # One-time script to create the FAISS vector index from your documents.
├── voice_service.py        # Handles text-to-speech conversion and playback using gTTS and Pygame.
├── requirments.txt         # All Python dependencies for the project.
├── rag_docs/               # Folder to store your knowledge base documents (.txt files).
├── faiss_rag.index         # The generated FAISS vector store file (binary).
├── Requestollama.py        # A simple utility script to test connection to the Ollama API directly.
└── .gitignore              # Specifies files and folders (like /venv) to be ignored by Git.
 Setup and Installation
Prerequisites
Python 3.8+

Ollama installed and running.

A working microphone.

Installation Steps
Clone the repository:

Bash

git clone <repository-url>
cd <repository-folder>
Create and activate a virtual environment:

Bash

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
Install the required Python packages:

Bash

pip install -r requirments.txt
Set up Ollama Models:

Ensure the Ollama application is running in your terminal or as a desktop application.

Pull the necessary models for the agent and embeddings:

Bash

ollama pull llama3
ollama pull nomic-embed-text
▶ How to Run
Prepare Your Knowledge Base:

Create a folder named rag_docs in the project's root directory.

Place any text documents (.txt files) you want the agent to have access to inside this folder.

Build the Vector Index:

Run the index_documents.py script once. This will read the files in rag_docs, generate embeddings, and create the faiss_rag.index file.

Bash

python index_documents.py
Start the Backend Server:

Run the main application script. This will start the audio recording loop and the WebSocket server.

Bash

python app.py
You should see messages indicating that the audio recording has started and the WebSocket server is running on ws://localhost:8765.

Connect a Frontend:

This project is a backend service. To interact with it, you need a WebSocket client.

You can use a simple web-based WebSocket client for testing or build a full-fledged chat interface (e.g., using React, Vue, or Svelte) that connects to ws://localhost:8765.

 Configuration
AI Models: The LLM and embedding models can be changed by modifying the LLM_MODEL_NAME and EMBED_MODEL_NAME constants in agentic_rag.py.

RAG Parameters: The chunk size and overlap for document splitting can be adjusted in index_documents.py.

Agent Persona: Veena's core instructions, rules, and personality can be modified in the persona string within the agentic_rag.py file.

Audio: The recording chunk length can be adjusted via DEFAULT_CHUNK_LENGTH in app.py.

 Utility Scripts
Requestollama.py: This is a simple script to send a direct request to your Ollama server. You can run python Requestollama.py to quickly verify that Ollama is running and the specified model (mistral in the script) is accessible before launching the main application.