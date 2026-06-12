# Software Architecture & Implementation Plan

## System Overview
The handheld AI Assistant is an offline-first, voice-driven, multilingual system built for edge devices, similar to BHASHINI x Current AI's vision.

### Complete Software Pipeline
```mermaid
graph TD
    A[Wake Word Detection (OpenWakeWord)] --> B[Voice Activity Detection (Silero VAD)]
    B --> C[Speech-to-Text (Whisper.cpp)]
    C --> D[Language Detection (FastText)]
    D --> E[Conversation Manager]
    E --> F[RAG Retrieval (FAISS + BGE-Small)]
    F --> G[LLM Inference (Ollama - Qwen/Gemma/Phi)]
    G --> H[LangGraph Agent]
    H --> I[Response Generation]
    I --> J[Text-to-Speech (Piper TTS)]
    J --> K[Audio Playback]
```

## Folder Structure
```text
.
├── ARCHITECTURE.md
├── README.md
├── docker-compose.yml
├── requirements.txt
├── backend/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas/
│   ├── agents/
│   │   └── graph.py
│   ├── audio/
│   │   ├── vad.py
│   │   └── wakeword.py
│   ├── core/
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── manager.py
│   ├── llm/
│   │   └── service.py
│   ├── rag/
│   │   ├── ingestion.py
│   │   └── retrieval.py
│   ├── security/
│   │   └── encryption.py
│   ├── stt/
│   │   ├── lang_detect.py
│   │   └── whisper_service.py
│   └── tts/
│   │   └── piper_service.py
├── configs/
│   └── config.yaml
├── data/
│   ├── db/
│   ├── docs/
│   └── models/
├── docker/
│   ├── Dockerfile.backend
│   └── Dockerfile.ui
├── tests/
│   ├── test_api.py
│   └── test_core.py
└── ui/
    └── main.py
```

## Deployment & Execution
1. **Download Models:** Place your GGUF/bin models inside `data/models/`.
2. **Run Backend Services:**
   ```bash
   docker-compose up --build
   ```
3. **Run Kivy UI:** (Typically on the host device to access microphone/speakers directly)
   ```bash
   python ui/main.py
   ```

## Optimization Recommendations
- Use `ggml-tiny.en.bin` or `ggml-base.bin` for Whisper.
- Use 4-bit or 8-bit quantized GGUF models for the LLM.
- Keep Piper models set to the `lessac-medium` or `low` tier for faster voice generation on edge hardware.
