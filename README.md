<div align="center">

```
                                  ██╗   ██╗███████╗███████╗███╗   ██╗ █████╗      █████╗ ██╗
                                  ██║   ██║██╔════╝██╔════╝████╗  ██║██╔══██╗    ██╔══██╗██║
                                  ██║   ██║█████╗  █████╗  ██╔██╗ ██║███████║    ███████║██║
                                  ╚██╗ ██╔╝██╔══╝  ██╔══╝  ██║╚██╗██║██╔══██║    ██╔══██║██║
                                   ╚████╔╝ ███████╗███████╗██║ ╚████║██║  ██║    ██║  ██║██║
                                    ╚═══╝  ╚══════╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝
```

### *Your insurance agent. In your language. At your pace.*

<img src="https://raw.githubusercontent.com/varundubey2804/Voice_Agent/main/demo/Screenshot%202026-04-12%20122529.png" alt="Veena AI" width="800"/>
<br/>

[![Python](https://img.shields.io/badge/Python_3.8+-FFD43B?style=flat-square&logo=python&logoColor=black)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq_Llama_3.3_70B-F55036?style=flat-square&logo=meta&logoColor=white)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS_Vector_DB-0467DF?style=flat-square&logo=meta&logoColor=white)](https://faiss.ai)
[![WebSocket](https://img.shields.io/badge/WebSocket_Real--Time-10b981?style=flat-square)](https://websockets.readthedocs.io)
[![Whisper](https://img.shields.io/badge/faster--whisper_STT-412991?style=flat-square)](https://github.com/SYSTRAN/faster-whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-white?style=flat-square)](LICENSE)

<br/>

> **Veena AI** is a production-ready, bilingual voice agent built for the Indian insurance market —  
> combining real-time speech recognition, RAG-powered knowledge retrieval, a 3D avatar, and  
> a full agent dashboard into one cohesive system.

<br/>

[**Quick Start**](#-quick-start) · [**Architecture**](#-architecture) · [**Features**](#-features) · [**API**](#-websocket-api) · [**Troubleshooting**](#-troubleshooting)


---

</div>

## What is Veena AI?

Veena is not just a chatbot. She is a fully voiced, persona-driven AI insurance agent that speaks both **English and Hindi**, understands your customers' policy questions, and responds with the warmth of a real agent — complete with a **3D animated VRM avatar**.

Built for hackathon speed, designed for production scale.

---


## 📸 Screenshots

<div align="center">

### 🏠 Landing Page
<img src="https://raw.githubusercontent.com/varundubey2804/Voice_Agent/main/demo/Screenshot%202026-04-12%20122529.png" alt="Landing Page" width="800"/>

<br/><br/>

### ✨ Features
<img src="https://raw.githubusercontent.com/varundubey2804/Voice_Agent/main/demo/Screenshot%202026-04-12%20122503.png" alt="Features" width="800"/>

<br/><br/>

### 🎙️ Agent in Action
<img src="https://raw.githubusercontent.com/varundubey2804/Voice_Agent/main/demo/Screenshot%202026-04-12%20101022.png" alt="Agent Working" width="800"/>

</div>

## ✨ Features

### 🎙️ Voice — Hear and Be Heard

| Capability | Details |
|---|---|
| **Speech-to-Text** | `faster-whisper` with optional CUDA acceleration |
| **Text-to-Speech** | `edge-tts` — Indian English (`en-IN-NeerjaNeural`) & Hindi (`hi-IN-SwaraNeural`) |
| **Language Detection** | Automatic Devanagari ↔ Latin script detection |
| **Silence Filtering** | Smart audio processing — no false triggers |

### 🧠 Intelligence — RAG Done Right

| Capability | Details |
|---|---|
| **LLM** | Groq API · Llama 3.3 70B Versatile |
| **Embeddings** | Ollama · `nomic-embed-text` |
| **Vector Store** | FAISS — fast cosine similarity search |
| **Agent Framework** | LangChain with conversational memory |
| **Persona** | Consistent "Veena" identity across all turns |

### 🌐 Interface — Built for Real Use

| Interface | Description |
|---|---|
| **Customer Portal** | Glassmorphism UI · 3D VRM avatar · voice + text chat |
| **Agent Dashboard** | Policy management · Supabase sync · search & filter |
| **Auth System** | Separate customer / agent login flows |
| **Real-time** | Full-duplex WebSocket on `ws://localhost:8765` |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BROWSER (HTML / JS)                         │
│                                                                     │
│   ┌─────────────────┐   ┌──────────────────┐   ┌───────────────┐   │
│   │   index.html    │   │ agent-dashboard  │   │  login pages  │   │
│   │ Customer Portal │   │  Policy Manager  │   │  Auth Flows   │   │
│   └────────┬────────┘   └────────┬─────────┘   └───────────────┘   │
│            └─────────────────────┘                                  │
│                         WebSocket (ws://localhost:8765)             │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                           app.py                                    │
│               WebSocket Server  +  Audio Event Loop                 │
│                                                                     │
│        ┌──────────────┬───────────────────┬───────────────┐        │
│        │              │                   │               │        │
│   ┌────▼──────┐  ┌────▼──────────┐  ┌────▼──────────┐   │        │
│   │  Whisper  │  │  agentic_rag  │  │ voice_service │   │        │
│   │   (STT)   │  │  LangChain    │  │  Edge TTS     │   │        │
│   └───────────┘  │  Agent + Mem  │  └───────────────┘   │        │
│                  └──────┬────────┘                       │        │
│                         │                                │        │
│                  ┌──────▼────────┐                       │        │
│                  │  FAISS Index  │◄──── index_documents  │        │
│                  │ faiss_rag.idx │          .py          │        │
│                  └───────────────┘                       │        │
└─────────────────────────────────────────────────────────────────────┘
```

**Data flow in 4 steps:**

1. User speaks → `faster-whisper` transcribes audio to text
2. Text hits LangChain agent → FAISS retrieves relevant policy chunks
3. Groq/Llama generates a contextual, persona-consistent reply
4. `edge-tts` converts response to audio → streamed back via WebSocket

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai) installed and running
- A [Groq API key](https://groq.com) (free tier works)
- `portaudio` for your OS:
  ```bash
  # Linux
  sudo apt-get install portaudio19-dev

  # macOS
  brew install portaudio

  # Windows — handled automatically by PyAudio
  ```

### 1 — Clone & Install

```bash
git clone https://github.com/varundubey2804/Voice_Agent.git
cd Voice_Agent

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2 — Configure

```bash
# Create .env in project root
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
echo "KMP_DUPLICATE_LIB_OK=TRUE" >> .env
```

Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

### 3 — Add Your Knowledge Base

Drop your insurance policy `.txt` files into `rag_docs/`:

```
rag_docs/
├── Calling Script.txt
├── Knowledge Base.txt
└── any_other_policy.txt
```

Then index them:

```bash
python index_documents.py
# Creates: faiss_rag.index/
```

### 4 — Add VRM Avatars

Download free VRM models from [VRoid Hub](https://hub.vroid.com) and place them at:

```
models/
├── veena.vrm    # Customer interface avatar
└── agent.vrm   # Agent dashboard avatar
```

### 5 — Run

```bash
# Terminal 1: Ollama (if not already running)
ollama serve

# Terminal 2: Veena backend
python app.py
```

You should see:

```
🔍 Loading Whisper on CUDA ...
✅ Agentic RAG with 'Veena' persona is ready!
🎙  Audio recording started...
🌐 Starting WebSocket server on ws://localhost:8765
```

Open `frontend/index.html` in Chrome or Edge. Veena is live.

---

## ⚙️ Configuration Reference

### `voice_service.py`
```python
VOICE_EN = "en-IN-NeerjaNeural"   # Indian English female
VOICE_HI = "hi-IN-SwaraNeural"    # Hindi female
```

### `agentic_rag.py`
```python
EMBED_MODEL_NAME = "nomic-embed-text"        # Ollama model
LLM_MODEL_NAME   = "llama-3.3-70b-versatile" # Groq model
FAISS_PATH       = "faiss_rag.index"
```

### `app.py` — CPU Fallback
If you don't have a GPU:
```python
# Change this line in app.py:
whisper_model = WhisperModel(size, device="cpu", compute_type="int8")
```

### Supabase (Optional — Agent Dashboard)
```javascript
// In frontend/agent-dashboard.html
const SUPABASE_URL      = 'https://your-project.supabase.co';
const SUPABASE_ANON_KEY = 'your-anon-key';
```

---

## 🔌 WebSocket API

**Endpoint:** `ws://localhost:8765`

### Client → Server

```jsonc
// Send a text message
{ "type": "text_input", "text": "What is the premium for term insurance?", "language": "en" }

// Control microphone
{ "type": "start_listening" }
{ "type": "stop_listening" }
```

### Server → Client

```jsonc
// Transcribed speech (what Veena heard)
{ "type": "user_message",   "text": "...", "timestamp": "2026-04-12T10:30:00" }

// Veena's reply
{ "type": "agent_response", "text": "...", "timestamp": "2026-04-12T10:30:05" }

// State events
{ "type": "speaking_started"  }
{ "type": "speaking_finished" }
{ "type": "listening_started" }
{ "type": "listening_stopped" }
```

---

## 📁 Project Structure

```
Voice_Agent/
│
├── app.py                 ← WebSocket server + audio event loop
├── agentic_rag.py         ← LangChain agent (RAG + memory + persona)
├── voice_service.py       ← TTS engine with language detection
├── index_documents.py     ← Indexes rag_docs/ into FAISS
├── lic_policies.py        ← LIC policy data helpers
├── finance_tools.py       ← Financial calculation tools
├── Requestollama.py       ← Ollama API helper
│
├── frontend/
│   ├── index.html         ← Customer chat UI (3D avatar + voice)
│   ├── login.html         ← Role selection entry point
│   ├── customer-login.html
│   ├── agent-login.html
│   ├── agent-dashboard.html  ← Policy management dashboard
│   ├── signup.html
│   ├── js/
│   │   └── supabase-client.js
│   └── libs/
│       ├── GLTFLoader.js
│       └── VRMLoaderPlugin.js
│
├── models/
│   ├── veena.vrm          ← Customer avatar (download separately)
│   └── agent.vrm          ← Agent avatar  (download separately)
│
├── rag_docs/              ← Your knowledge base (.txt files)
├── faiss_rag.index/       ← Auto-generated vector index
│   ├── index.faiss
│   └── index.pkl
│
├── .env                   ← API keys (never commit this)
├── requirements.txt
└── README.md
```

---

## 🐛 Troubleshooting

<details>
<summary><strong>WebSocket connection failed</strong></summary>

```
Error: WebSocket connection to 'ws://localhost:8765' failed
```

- Is `python app.py` running?
- Is port `8765` blocked by a firewall?
- Check: `lsof -i :8765` (Linux/Mac) or `netstat -ano | findstr 8765` (Windows)

</details>

<details>
<summary><strong>CUDA out of memory</strong></summary>

```
RuntimeError: CUDA out of memory
```

Switch to CPU in `app.py`:
```python
whisper_model = WhisperModel(size, device="cpu", compute_type="int8")
```

</details>

<details>
<summary><strong>PyAudio install fails on Windows</strong></summary>

```
error: Microsoft Visual C++ 14.0 is required
```

Download a prebuilt wheel from [Gohlke's Pythonlibs](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio):
```bash
pip install PyAudio-0.2.11-cp38-cp38-win_amd64.whl
```

</details>

<details>
<summary><strong>FAISS index not found</strong></summary>

```
FileNotFoundError: FAISS index not found at 'faiss_rag.index'
```

Run the indexer first, and ensure your `rag_docs/` folder is not empty:
```bash
python index_documents.py
```

</details>

<details>
<summary><strong>Groq rate limit exceeded</strong></summary>

Wait a moment and retry. If it persists, check your Groq API tier at [console.groq.com](https://console.groq.com).

</details>

<details>
<summary><strong>VRM model won't load</strong></summary>

- Confirm files exist at `models/veena.vrm` and `models/agent.vrm`
- Open browser devtools → check for CORS or 404 errors
- Ensure the model is **VRM 1.0** format (some older VRoid exports are VRM 0.x)

</details>

---

## 🔒 Security Checklist

Before going to production:

- [ ] Add `.env` to `.gitignore` — **never** commit API keys
- [ ] Enable [Row Level Security](https://supabase.com/docs/guides/auth/row-level-security) in Supabase
- [ ] Switch WebSocket from `ws://` to `wss://` (TLS)
- [ ] Set strict CORS headers on your backend
- [ ] Rotate all API keys used during development

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss what you'd like to change.

```bash
git checkout -b feature/your-feature-name
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
# → Open a Pull Request
```

Please follow **PEP 8** for Python and add docstrings to new functions.

---

## 🙏 Built With

| Library | Role |
|---|---|
| [LangChain](https://langchain.com) | Agent framework + RAG pipeline |
| [Groq](https://groq.com) | Ultra-fast LLM inference (Llama 3.3 70B) |
| [Ollama](https://ollama.ai) | Local embedding generation |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Real-time speech recognition |
| [edge-tts](https://github.com/rany2/edge-tts) | Natural Indian English / Hindi TTS |
| [FAISS](https://faiss.ai) | Vector similarity search |
| [Supabase](https://supabase.com) | Agent dashboard backend |
| [Three.js + VRM](https://hub.vroid.com) | 3D avatar rendering |

---

## 📄 License

MIT © [BlackDragons](https://github.com/varundubey2804)

---

<div align="center">

**If Veena helped you, give the repo a ⭐ — it means a lot.**

Made in India, for India 🇮🇳

</div>
