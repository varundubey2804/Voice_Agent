# Veena AI - Intelligent Voice Insurance Agent 🤖

<div align="center">

![Veena AI](https://img.shields.io/badge/Veena-AI%20Agent-10b981?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![WebSocket](https://img.shields.io/badge/WebSocket-Real--time-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**An AI-powered voice insurance agent with bilingual support (English/Hindi), real-time speech recognition, and RAG-based knowledge retrieval.**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Configuration](#configuration)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Frontend Pages](#frontend-pages)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**Veena AI** is a sophisticated voice-enabled insurance agent system built with Python and modern web technologies. It combines speech recognition, natural language processing, text-to-speech, and RAG (Retrieval-Augmented Generation) to create an intelligent conversational agent that helps insurance customers with policy inquiries, payment reminders, and general support.

The system features:
- **Bilingual Support**: Seamlessly handles English and Hindi conversations
- **Real-time Voice Interaction**: Uses Whisper for speech recognition and Edge TTS for natural-sounding responses
- **3D Avatar**: Interactive VRM model that responds to user input with realistic animations
- **Knowledge Base**: RAG pipeline powered by FAISS and Ollama embeddings
- **Agent Dashboard**: Web interface for managing policies and knowledge base
- **Multi-platform**: Desktop and web-based interfaces

---

## ✨ Key Features

### 🎙️ Voice Capabilities
- **Real-time Speech Recognition**: Powered by faster-whisper with CUDA acceleration
- **Automatic Language Detection**: Detects Hindi (Devanagari) vs English text
- **Natural TTS**: Edge TTS with Indian English and Hindi voices
- **Silence Detection**: Intelligent audio processing to filter background noise

### 🧠 AI Intelligence
- **Agentic RAG**: LangChain-based agent with conversational memory
- **Vector Database**: FAISS for efficient similarity search
- **LLM Integration**: Uses Groq API with Llama 3.3 70B model
- **Contextual Responses**: Maintains conversation history and persona

### 🌐 Web Interface
- **Customer Portal**: Modern glass-morphism UI with 3D avatar
- **Agent Dashboard**: Policy management with Supabase integration
- **Real-time Updates**: WebSocket communication for instant responses
- **Responsive Design**: Mobile-friendly Tailwind CSS styling

### 🔧 Technical Stack
- **Backend**: Python, WebSockets, asyncio
- **AI/ML**: LangChain, Ollama, Groq, faster-whisper
- **Frontend**: React, Three.js, VRM models
- **Database**: FAISS (vector), Supabase (optional)
- **Audio**: PyAudio, pygame, edge-tts

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (HTML/JS)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Customer UI  │  │ Agent Portal │  │ Login Pages  │     │
│  │ (index.html) │  │ (dashboard)  │  │              │     │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘     │
│         │                  │                                │
│         └──────────┬───────┘                                │
│                    │ WebSocket (ws://localhost:8765)        │
└────────────────────┼────────────────────────────────────────┘
                     │
┌────────────────────┼────────────────────────────────────────┐
│                    ▼                                         │
│              ┌──────────┐                                    │
│              │  app.py  │  (WebSocket Server + Audio Loop)  │
│              └─────┬────┘                                    │
│                    │                                         │
│         ┌──────────┼──────────┐                             │
│         │          │           │                             │
│    ┌────▼────┐ ┌──▼──────┐ ┌─▼─────────┐                   │
│    │ Whisper │ │Agentic  │ │Voice      │                   │
│    │ (STT)   │ │RAG      │ │Service    │                   │
│    │         │ │(Agent)  │ │(TTS)      │                   │
│    └─────────┘ └────┬────┘ └───────────┘                   │
│                     │                                        │
│                ┌────▼────┐                                   │
│                │  FAISS  │  (Vector Database)               │
│                │  Index  │                                   │
│                └─────────┘                                   │
│                     ▲                                        │
│                     │                                        │
│              ┌──────┴──────────┐                            │
│              │ index_documents │                            │
│              │      .py        │                            │
│              └─────────────────┘                            │
└──────────────────────────────────────────────────────────────┘
```

---

## 📦 Prerequisites

### System Requirements
- **OS**: Windows/Linux/macOS
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended for CUDA)
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster performance)

### Required Software
1. **Ollama**: For embeddings
   ```bash
   # Install from https://ollama.ai
   ollama pull nomic-embed-text
   ```

2. **Audio Drivers**: PyAudio dependencies
   - Windows: Included with PyAudio
   - Linux: `sudo apt-get install portaudio19-dev`
   - macOS: `brew install portaudio`

3. **Groq API Key**: Sign up at https://groq.com

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/veena-ai.git
cd veena-ai
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt content:**
```
# Core
asyncio
websockets

# AI/ML
langchain
langchain-community
langchain-groq
langchain-ollama
faster-whisper
faiss-cpu  # or faiss-gpu for CUDA support

# Audio
edge-tts
pygame
pyaudio
soundfile
scipy
numpy

# Utilities
python-dotenv
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
KMP_DUPLICATE_LIB_OK=TRUE
```

### 5. Download VRM Models
Place your VRM avatar models in the `models/` directory:
- `models/veena.vrm` - Customer interface avatar
- `models/agent.vrm` - Agent dashboard avatar

**Note**: You can download free VRM models from [VRoid Hub](https://hub.vroid.com/)

### 6. Prepare Knowledge Base
Create a `rag_docs/` folder and add your policy documents as `.txt` files:
```
rag_docs/
├── Calling Script.txt
├── Knowledge Base.txt
└── ... (other policy documents)
```

### 7. Index Documents
```bash
python index_documents.py
```
This creates the `faiss_rag.index/` directory with vector embeddings.

---

## ⚙️ Configuration

### Voice Configuration (`voice_service.py`)
```python
VOICE_EN = "en-IN-NeerjaNeural"  # Indian English Female
VOICE_HI = "hi-IN-SwaraNeural"   # Hindi Female
```

### Agent Configuration (`agentic_rag.py`)
```python
EMBED_MODEL_NAME = "nomic-embed-text"  # Ollama embedding model
LLM_MODEL_NAME = "llama-3.3-70b-versatile"  # Groq LLM
FAISS_PATH = "faiss_rag.index"
```

### WebSocket Configuration (`app.py`)
```python
# Server runs on ws://localhost:8765
# Adjust in both app.py and frontend HTML files if needed
```

### Supabase Configuration (Optional)
If using the agent dashboard with database:
1. Create a Supabase project at https://supabase.com
2. Update credentials in `agent-dashboard.html`:
```javascript
const SUPABASE_URL = 'your-project-url';
const SUPABASE_ANON_KEY = 'your-anon-key';
```

---

## 🎮 Usage

### Starting the System

1. **Start Ollama** (if not running):
```bash
ollama serve
```

2. **Run the Main Application**:
```bash
python app.py
```

You should see:
```
🔍 Loading Whisper on CUDA ...
✅ Agentic RAG with 'Veena' persona is ready!
🎙 Audio recording started...
🌐 Starting WebSocket server on ws://localhost:8765
```

3. **Open the Web Interface**:
Open `index.html` in a modern web browser (Chrome/Edge recommended).

### Using the Customer Interface

1. **Wait for Model Load**: The VRM avatar will load (progress shown)
2. **Choose Language**: Toggle between English and Hindi
3. **Voice Input**: Click the microphone button to speak
4. **Text Input**: Type messages in the chat box
5. **Receive Responses**: Veena will respond with voice and text

### Using the Agent Dashboard

1. Open `login.html` → Click "Bank Agent"
2. Login with agent credentials (or use bypass mode)
3. **Add Policies**: Fill out the form to add new knowledge
4. **Search Policies**: Use the search bar to filter
5. **Delete Policies**: Hover over policies to show delete button

### Testing the RAG Agent (CLI)

```bash
python agentic_rag.py
```

Interact with Veena in the terminal:
```
🗣  You: What is the premium for term insurance?
🤖 Veena: Based on our policies, the term insurance premium starts at ₹500/month...
```

---

## 📁 Project Structure

```
VOICE_AGENT/
│
├── 📄 app.py                      # Main WebSocket server + audio loop
├── 📄 agentic_rag.py              # LangChain agent with RAG
├── 📄 voice_service.py            # TTS with language detection
├── 📄 index_documents.py          # FAISS indexing script
├── 📄 Requestollama.py            # Ollama interaction helper
│
├── 📂 frontend/                   # Web interface
│   ├── index.html                 # Customer chat interface
│   ├── login.html                 # Main login page
│   ├── customer-login.html        # Customer authentication
│   ├── agent-login.html           # Agent authentication
│   ├── agent-dashboard.html       # Policy management dashboard
│   ├── signup.html                # User registration
│   │
│   ├── 📂 js/
│   │   └── supabase-client.js     # (Referenced but not shown)
│   │
│   └── 📂 libs/                   # Three.js libraries
│       ├── GLTFLoader.js
│       └── VRMLoaderPlugin.js
│
├── 📂 models/                     # VRM avatar files
│   ├── veena.vrm                  # Customer interface avatar
│   └── agent.vrm                  # Agent dashboard avatar
│
├── 📂 rag_docs/                   # Knowledge base documents
│   ├── Calling Script.txt
│   ├── Knowledge Base.txt
│   └── ...
│
├── 📂 faiss_rag.index/            # Vector database (generated)
│   ├── index.faiss
│   └── index.pkl
│
├── 📄 .env                        # Environment variables
├── 📄 .gitignore                  # Git ignore rules
├── 📄 requirements.txt            # Python dependencies
└── 📄 README.md                   # This file
```

---

## 🔌 API Documentation

### WebSocket Messages

#### Client → Server

**Text Input**
```json
{
  "type": "text_input",
  "text": "Tell me about term insurance",
  "language": "en"
}
```

**Start/Stop Listening**
```json
{
  "type": "start_listening"
}
{
  "type": "stop_listening"
}
```

#### Server → Client

**User Message**
```json
{
  "type": "user_message",
  "text": "User's transcribed speech",
  "timestamp": "2026-01-21T10:30:00"
}
```

**Agent Response**
```json
{
  "type": "agent_response",
  "text": "Veena's response",
  "timestamp": "2026-01-21T10:30:05"
}
```

**Status Updates**
```json
{
  "type": "speaking_started"
}
{
  "type": "speaking_finished"
}
{
  "type": "listening_started"
}
{
  "type": "listening_stopped"
}
```

---

## 🌐 Frontend Pages

### 1. **index.html** - Customer Interface
- 3D VRM avatar with animations
- Real-time voice/text chat
- Language toggle (English/Hindi)
- WebSocket connection status
- Orbiting keyword animations

### 2. **login.html** - Entry Point
- Role selection (Customer/Agent)
- Animated background effects
- VRM avatar preview

### 3. **agent-dashboard.html** - Policy Management
- Add/Edit/Delete policies
- Search and filter
- Statistics dashboard
- Real-time Supabase sync

### 4. **customer-login.html / agent-login.html**
- Separate authentication flows
- Email/password validation
- Account type verification

### 5. **signup.html** - Registration
- User type selection
- Form validation
- Supabase integration

---

## 🐛 Troubleshooting

### Common Issues

**1. WebSocket Connection Failed**
```
Error: WebSocket connection to 'ws://localhost:8765' failed
```
- Ensure `app.py` is running
- Check firewall settings
- Verify port 8765 is not in use

**2. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
- Use CPU mode: Change in `app.py`:
  ```python
  whisper_model = WhisperModel(size, device="cpu", compute_type="int8")
  ```

**3. PyAudio Installation Errors (Windows)**
```
error: Microsoft Visual C++ 14.0 is required
```
- Download PyAudio wheel: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
- Install: `pip install PyAudio‑0.2.11‑cp38‑cp38‑win_amd64.whl`

**4. FAISS Index Not Found**
```
FileNotFoundError: FAISS index not found at 'faiss_rag.index'
```
- Run: `python index_documents.py`
- Ensure `rag_docs/` contains `.txt` files

**5. Groq API Rate Limit**
```
Error: Rate limit exceeded
```
- Wait and retry
- Consider using a different LLM provider
- Check your Groq API tier

**6. VRM Model Not Loading**
- Verify model files exist in `models/` directory
- Check browser console for CORS errors
- Ensure models are valid VRM 1.0 format

---

## 🔒 Security Notes

- **API Keys**: Never commit `.env` file to Git
- **Supabase**: Use Row Level Security (RLS) policies
- **CORS**: Configure properly for production
- **WebSocket**: Consider TLS (wss://) for production

---

## 🚀 Deployment

### Local Network Access
1. Update WebSocket URLs in HTML files to use your local IP
2. Run: `python app.py`
3. Access from other devices: `http://YOUR_IP:PORT`

### Production Deployment
1. Use a production WSGI server (e.g., Gunicorn)
2. Set up reverse proxy (Nginx)
3. Enable HTTPS and WSS
4. Configure firewall rules
5. Use environment variables for all secrets

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use meaningful commit messages
- Add docstrings to functions
- Test thoroughly before submitting

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👏 Acknowledgments

- **LangChain**: For the RAG framework
- **Groq**: For fast LLM inference
- **Ollama**: For local embeddings
- **Edge TTS**: For natural-sounding voices
- **Faster Whisper**: For accurate speech recognition
- **VRoid**: For VRM avatar standards
- **Supabase**: For backend infrastructure

---

## 📧 Contact

**Created by**: BlackDragons  
**Project Link**: (https://github.com/varundubey2804/Voice_Agent)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ for the future of conversational AI

</div>
