# Voice-Enabled RAG Insurance Agent

This project implements a voice-enabled conversational AI agent named "Veena". Veena acts as an insurance agent for "ValuEnable Life Insurance". The application is built using Streamlit for the user interface, `faster-whisper` for speech-to-text, `gTTS` for text-to-speech, and a LangChain-based agentic Retrieval-Augmented Generation (RAG) pipeline powered by a local Ollama LLM.

## Features

* **Voice Interaction**: Users can interact with the agent using their voice.
* **Speech-to-Text**: Utilizes `faster-whisper` for accurate and fast transcription of user's speech.
* **Text-to-Speech**: The agent's responses are converted to speech using Google's Text-to-Speech service.
* **Retrieval-Augmented Generation (RAG)**: The agent can answer questions based on a knowledge base of documents. It uses a FAISS vector store for efficient retrieval.
* **Agentic Framework**: A LangChain agent is used to provide a conversational experience, with a defined persona and access to tools.
* **Local LLM**: Powered by a local Ollama instance (e.g., Llama 3), ensuring data privacy and control.
* **Web Interface**: A user-friendly chat interface built with Streamlit, featuring a modern design and animations.

## Project Structure
Of course. Here is the project structure as described in the README file:

```
.
├── app.py                  # Main Streamlit application
├── agentic_rag.py          # Defines the LangChain agent and RAG pipeline
├── index_documents.py      # Script to create the FAISS vector store from documents
├── voice_service.py        # Handles text-to-speech functionality
├── requirments.txt         # Python dependencies
├── rag_docs/               # Folder to store your knowledge base documents (.txt files)
├── faiss_rag.index         # The generated FAISS vector store
└── .gitignore              # Specifies files to be ignored by Git
```
## Setup and Installation

### Prerequisites

* Python 3.8+
* [Ollama](https://ollama.ai/) installed and running with the required models.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirments.txt
    ```

4.  **Set up Ollama:**
    * Make sure the Ollama application is running.
    * Pull the necessary models:
        ```bash
        ollama pull llama3
        ollama pull nomic-embed-text
        ```

## How to Run

1.  **Prepare your knowledge base:**
    * Create a folder named `rag_docs` in the project root.
    * Place your text documents (`.txt` files) inside this folder.

2.  **Index the documents:**
    * Run the `index_documents.py` script once to create the FAISS vector index. This will create a `faiss_rag.index` file in your project directory.
    ```bash
    python index_documents.py
    ```

3.  **Launch the application:**
    * Run the Streamlit app.
    ```bash
    streamlit run app.py
    ```
    * Open your web browser and navigate to the local URL provided by Streamlit.

4.  **Interact with the agent:**
    * Click the "Record" button to start speaking.
    * The agent will listen, process your query, and respond with voice.

## Configuration

* **Models**: The LLM and embedding models can be changed in `agentic_rag.py` and `index_documents.py`.
* **RAG Parameters**: Chunk size and overlap for document splitting can be adjusted in `index_documents.py`.
* **Agent Persona**: The agent's persona and instructions can be modified in the `persona` string within the `agentic_rag.py` file.
