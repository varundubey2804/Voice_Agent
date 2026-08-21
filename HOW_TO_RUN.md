# 🚀 How to Run Veena AI - Complete Setup Guide

## ✅ Prerequisites

### **1. System Requirements**
- **Python**: 3.8 or higher
- **OS**: Windows, macOS, or Linux
- **RAM**: 8GB minimum (for Whisper model + Ollama)
- **Storage**: 2GB free (for models)
- **Microphone**: For voice input
- **Internet**: For API calls (Groq, Edge TTS, yfinance)

### **2. API Keys & Accounts**
You'll need these credentials (already in `.env`):
- ✅ **GROQ_API_KEY** - For LLM (Llama 3.3 70B)
- ✅ **SUPABASE_URL** - For portfolio database (optional)
- ✅ **SUPABASE_ANON_KEY** - For Supabase auth (optional)

---

## 📦 Step-by-Step Setup

### **Step 1: Navigate to Project Directory**
```powershell
cd c:\Users\Admin\Desktop\voice_agent
```

### **Step 2: Create & Activate Virtual Environment**

#### **Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate it
& .\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

#### **Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### **macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### **Step 3: Install Dependencies**

```powershell
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install langchain-groq langchain-ollama langchain-community faster-whisper pyaudio edge-tts pygame yfinance websockets numpy scikit-learn soundfile python-dotenv
```

Or if you have a `requirements.txt`:
```powershell
pip install -r requirements.txt
```

**Key packages:**
- `langchain-groq` - LLM integration
- `langchain-ollama` - Embeddings
- `faster-whisper` - Speech recognition
- `edge-tts` - Text-to-speech
- `pyaudio` - Microphone access
- `websockets` - Real-time communication
- `yfinance` - Stock market data

### **Step 4: Install & Run Ollama (For Embeddings)**

Ollama provides the embedding model locally.

#### **Download Ollama:**
```
Visit: https://ollama.ai
Download for your OS
```

#### **Pull Embedding Model:**
```powershell
# After installing Ollama, run:
ollama pull nomic-embed-text

# Verify it's running:
ollama serve
```

**Keep Ollama running in the background** - the system needs it for embeddings.

### **Step 5: Build FAISS Index (Knowledge Base)**

Before running the main app, index your documents:

```powershell
# Make sure you're in the project folder with venv activated
python index_documents.py
```

**This will:**
- Load `.txt` files from `rag_docs/` folder
- Split them into chunks
- Embed them using Ollama
- Create `faiss_rag.index/` directory with indexed data

**Expected output:**
```
============================================================
🚀  Veena AI — Document Indexer
    Source   : rag_docs
    Index    : faiss_rag.index
    Model    : nomic-embed-text
    Chunk    : size=500, overlap=50
============================================================

📁  Loading documents from 'rag_docs' …
✅  2 file(s) loaded.

✂️   Split into 35 chunk(s).

🧠  Loading Ollama embeddings (nomic-embed-text) …
    (Make sure you have run: ollama pull nomic-embed-text)

🧱  Building FAISS index …

✅  Index saved  → faiss_rag.index
    Chunks indexed: 35
============================================================
```

---

## 🎯 Running the Application

### **Step 1: Start Ollama (if not already running)**

Open a new PowerShell/Terminal and run:
```powershell
ollama serve
```

You should see:
```
pulling manifest
pulling 3c2df53fcf96
pulling f752399f4d5b
pulling 5f7dd5ab3c55
...
```

**Leave this terminal open!**

### **Step 2: Start Veena AI Server**

In your main PowerShell terminal (with venv activated):
```powershell
python app.py
```

**Expected output:**
```
🔍 Loading Whisper 'small' on CPU …
📂  Loading FAISS index: faiss_rag.index
🤖 Veena Agent built successfully
🎙  Audio producer started …
🎧 Serving HTTP on port 8080 …
🔌 WebSocket server listening on ws://0.0.0.0:8765
✅ Veena AI ready! Open http://localhost:8080 in your browser
```

### **Step 3: Open Browser & Access UI**

1. **Open**: http://localhost:8080
2. **You should see**:
   - 3D Avatar (Veena AI) in center
   - Chat interface below
   - Microphone button on left
   - Text input box in middle
   - Connected status indicator

---

## 🎤 Using the Application

### **Voice Input Mode:**
```
1. Click 🎤 Mic button
   └─ Microphone starts listening (blue indicator)
2. Speak your question
   └─ "What is term insurance?"
   └─ "Show me my portfolio"
   └─ "Compare LIC vs Stocks"
3. Stop speaking or wait 4 seconds
   └─ Audio auto-stops and transcribes
4. AI responds with voice + text
   └─ Avatar animates while speaking
   └─ Message appears in chat
```

### **Text Input Mode:**
```
1. Click in text box
2. Type your question
3. Press Enter or click Send
4. Same response flow
```

### **Example Questions:**
- 📋 "What are the benefits of Jeevan Anand?"
- 📊 "Show my portfolio"
- 💹 "What's the price of TCS?"
- 🏆 "Top gainers today?"
- 💰 "Calculate my taxes"
- 🤝 "Recommend an insurance policy"
- 📈 "SIP calculator for 5000 monthly"
- 🔀 "Compare LIC vs Stocks vs Mutual Funds"

---

## 🛑 Stopping the Application

Press `Ctrl+C` in the PowerShell terminal running `app.py`.

You should see:
```
^C
🛑 Server shutting down...
✅ Goodbye!
```

---

## 🐛 Troubleshooting

### **Issue: "GROQ_API_KEY not found"**
```
Solution:
1. Check .env file has GROQ_API_KEY set
2. Restart terminal after editing .env
3. Run: echo $env:GROQ_API_KEY (should show key)
```

### **Issue: "Cannot open microphone"**
```
Solutions:
- Grant microphone permission to Python
- Try: pip install --upgrade pyaudio
- On Windows: Control Panel → Sound → App volume and device preferences
- Ensure no other app is using mic
```

### **Issue: "FAISS index not found"**
```
Solution:
1. Run: python index_documents.py
2. Wait for indexing to complete
3. Verify faiss_rag.index/ folder created
4. Then run app.py
```

### **Issue: "Ollama connection refused"**
```
Solution:
1. Open new PowerShell
2. Run: ollama serve
3. Keep it running
4. Then start app.py in another terminal
```

### **Issue: "ModuleNotFoundError"**
```
Solution:
1. Activate venv: & .\venv\Scripts\Activate.ps1
2. Check: pip list
3. Install missing: pip install [package-name]
4. Restart terminal
```

### **Issue: "WebSocket connection failed"**
```
Solution:
- Check port 8765 is not used: netstat -ano | findstr :8765
- Kill process if needed: taskkill /PID [PID] /F
- Check firewall allows port 8765
```

### **Issue: "Edge TTS error / No audio"**
```
Solution:
1. Check internet connection
2. Try different language: "hello" (English) vs "नमस्ते" (Hindi)
3. Check volume is not muted
4. Restart browser tab
```

---

## 📊 Checking Everything Works

### **Test Checklist:**

```
✓ Step 1: Venv activated
  └─ Check: (venv) appears in terminal prompt

✓ Step 2: Ollama running
  └─ Run: curl http://localhost:11434
  └─ Should respond (no error)

✓ Step 3: .env configured
  └─ Check: GROQ_API_KEY is set
  └─ Run: python -c "import os; os.getenv('GROQ_API_KEY')"

✓ Step 4: FAISS index exists
  └─ Check: ls faiss_rag.index/
  └─ Should show: index.faiss and index.pkl

✓ Step 5: App starts
  └─ Run: python app.py
  └─ Should show: "✅ Veena AI ready!"

✓ Step 6: Browser connects
  └─ Open: http://localhost:8080
  └─ Should see: Avatar + chat interface + "Connected" badge

✓ Step 7: Voice works
  └─ Click mic, speak "hello"
  └─ Should hear: Transcription in chat
  └─ Should hear: AI response voice + avatar animation
```

---

## 🎯 Quick Start (TL;DR)

```powershell
# 1. Navigate
cd c:\Users\Admin\Desktop\voice_agent

# 2. Activate venv
& .\venv\Scripts\Activate.ps1

# 3. Start Ollama (in separate terminal)
ollama serve

# 4. Index documents (first time only)
python index_documents.py

# 5. Run app
python app.py

# 6. Open browser
# http://localhost:8080

# 7. Click mic, speak!
```

---

## 📝 Project Structure

```
voice_agent/
├── app.py                    ← Main server (WebSocket + audio)
├── agentic_rag.py           ← Agent logic & tools
├── voice_service.py         ← TTS streaming
├── finance_tools.py         ← Market data, portfolio, tax calc
├── lic_policies.py          ← Insurance policy database
├── index_documents.py       ← Build FAISS index
├── .env                     ← API keys (KEEP SECURE!)
├── requirements.txt         ← Python dependencies
├── faiss_rag.index/         ← Vector database (auto-created)
├── rag_docs/                ← Knowledge base documents
│   ├── Calling Script.txt
│   └── Knowledge Base.txt
├── frontend/                ← Web UI
│   ├── index.html          ← Main app
│   ├── agent-dashboard.html
│   ├── login.html
│   ├── js/
│   │   └── supabase-client.js
│   ├── libs/
│   │   ├── GLTFLoader.js
│   │   └── VRMLoaderPlugin.js
│   └── models/
│       ├── agent.vrm
│       └── veena.vrm
└── venv/                    ← Virtual environment
```

---

## 🔄 Development Workflow

### **Making Changes:**

```
1. Edit Python files (app.py, agentic_rag.py, etc.)
2. Save file
3. Restart app.py (Ctrl+C then python app.py)
4. Browser auto-reconnects via WebSocket

Changes take effect immediately!
```

### **Adding New Tools:**

```
1. Add function to finance_tools.py
2. Add Tool object in agentic_rag.py build_agent()
3. Restart app.py
4. Agent can now call your new tool
```

### **Updating Knowledge Base:**

```
1. Add/edit .txt files in rag_docs/
2. Run: python index_documents.py
3. Restart app.py (FAISS reloaded)
4. Agent uses updated knowledge
```

---

## 🚀 Performance Tips

```
• Whisper model size affects speed/quality:
  - "tiny" = fastest, lowest quality
  - "small" = 4x faster than medium, good quality ✅ (current)
  - "medium" = slower, high quality
  - "large" = slowest, best quality

• Change in app.py:
  DEFAULT_MODEL_SIZE = "small"

• Chunk length affects responsiveness:
  - Smaller chunks = faster response
  - Current: 4 seconds ✅

• LLM temperature for consistency:
  - 0 = deterministic, fast ✅ (current)
  - 0.7 = creative, slower
```

---

## ⚙️ Configuration Files

### **.env (API Keys)**
```
GROQ_API_KEY="gsk_..."              ← Required for LLM
SUPABASE_URL="https://..."          ← Optional for database
SUPABASE_ANON_KEY="sb_..."          ← Optional for auth
```

**NEVER commit `.env` to git!** Add to `.gitignore`

### **Optional: Custom Voices**

Edit in `voice_service.py`:
```python
VOICES = {
    "en": "en-IN-NeerjaNeural",  ← Indian English female
    "hi": "hi-IN-SwaraNeural",   ← Hindi female
}
# Other options:
# en-IN-PrabhatNeural           (Indian male)
# hi-IN-ManishNeural            (Hindi male)
```

---

## 📞 Getting Help

### **Check Logs:**
```powershell
# App.py logs to console in real-time
# Look for:
# 🎤 Audio producer started
# 🔍 Loading Whisper
# 📂 Loading FAISS
# ✅ Success messages or ❌ errors
```

### **Test Individual Components:**
```powershell
# Test Ollama
curl http://localhost:11434

# Test FAISS index
python -c "from langchain_community.vectorstores import FAISS; db = FAISS.load_local('faiss_rag.index', allow_dangerous_deserialization=True); print('✅ FAISS OK')"

# Test Groq API
python -c "from langchain_groq import ChatGroq; llm = ChatGroq(api_key='your-key'); print(llm.invoke('hello'))"

# Test Whisper
python -c "from faster_whisper import WhisperModel; model = WhisperModel('small'); print('✅ Whisper OK')"
```

---

## 🎉 You're Ready!

Once everything is running:
- 🎤 Speak in English or Hindi
- 📊 Ask about insurance, markets, taxes
- 💼 Manage your portfolio
- 🤖 Chat with avatar Veena AI
- 📈 Get investment advice
- 🎯 Get personalized recommendations

**Enjoy! 🚀**

