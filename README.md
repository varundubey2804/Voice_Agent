# Handheld AI Assistant System

An open-source, voice-first, multilingual, privacy-preserving handheld AI assistant designed to work primarily offline.

## Core Features
- **Offline-First:** Runs locally utilizing quantized models and lightweight architectures.
- **Voice-First:** Uses Wake Word detection (OpenWakeWord) and VAD (Silero) to start transcription seamlessly.
- **Multilingual Support:** Supports multiple languages through Whisper.cpp, Piper TTS, and FastText language detection.
- **Privacy-Preserving:** 100% local processing. No data is sent to the cloud. Conversations and memory are secured via AES-256 local encryption.
- **Agentic RAG:** Features localized document retrieval via FAISS and bge-small, and multi-step reasoning via LangGraph and local LLMs (via Ollama/llama.cpp).

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Models
Before starting the services, ensure you download the necessary models into `data/models/`:
- `lid.176.ftz` (FastText Language Detection)
- Whisper.cpp quantized models (e.g., `ggml-tiny.en.bin`)
- Piper ONNX voice models (e.g., `en_US-lessac-medium.onnx` and its JSON config)
- Optionally pull the Ollama models by running `docker-compose up ollama` followed by `curl -X POST http://localhost:11434/api/pull -d '{"name": "qwen:0.5b"}'`

### 3. Start Backend
Run the backend system via Docker:
```bash
docker-compose up --build
```
Or run locally:
```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start Frontend
Run the lightweight Kivy UI locally on the edge device:
```bash
python ui/main.py
```

## Architecture Details
Check out `ARCHITECTURE.md` for diagrams, full software stack pipeline details, and optimization techniques.

## Security & Privacy
Data is written locally to `data/db/` and `data/docs/`.
Data is kept entirely offline.
Encryption utilities in `backend/security/encryption.py` are provided to encrypt localized memory or data on disk.
