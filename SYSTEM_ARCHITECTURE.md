# Veena AI - Complete System Architecture & Logic Flow

## 🎯 System Overview

**Veena AI** is an intelligent voice-enabled insurance agent system that combines speech recognition, NLP, and RAG (Retrieval-Augmented Generation) to provide conversational insurance advisory with real-time voice interaction.

---

## 📊 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND LAYER (Web UI)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  index.html  │  │ agent-dash   │  │ login/signup │  │ customer UI  │   │
│  │  (React UI)  │  │  (Portal)    │  │ (Supabase)   │  │ (3D Avatar)  │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                  │                  │                 │           │
│         └──────────────────┼──────────────────┼─────────────────┘           │
│                            │ WebSocket (ws://localhost:8765)               │
└────────────────────────────┼─────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────────────┐
│                      BACKEND SERVER LAYER                                    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  app.py - WebSocket Server (Port 8765)                              │   │
│  │  ├─ handle_websocket()      : Connection management                 │   │
│  │  ├─ handle_client_message() : Route incoming requests               │   │
│  │  ├─ process_user_input()    : Text → Agent → Response               │   │
│  │  └─ play_tts_and_notify()   : Stream TTS output                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│         ┌──────────────────────────┼──────────────────────────┐             │
│         │                          │                          │             │
│         ▼                          ▼                          ▼             │
│  ┌──────────────┐  ┌──────────────┐────┐  ┌──────────────┐  │             │
│  │ agentic_rag  │  │ voice_service│    │  │ finance_tools│  │             │
│  ├─ Agent Exec │  ├─ TTS (Edge)  │    │  ├─ Portfolio  │  │             │
│  ├─ Tools Mgmt │  ├─ Streaming   │    │  ├─ Market API │  │             │
│  ├─ Memory     │  └─ Playback    │    │  ├─ Tax Calc   │  │             │
│  └──────┬───────┘                 │    │  └─ Insurance  │  │             │
│         │                         │    │                   │             │
│         └─────────────────────────┴────┴───────────────────┘             │
│                                    │                                      │
└────────────────────────────────────┼──────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
    ┌─────────────┐         ┌──────────────┐         ┌──────────────┐
    │  FAISS Index│         │  Ollama LLM  │         │  Groq API    │
    │ (Vector DB) │         │ (Embeddings) │         │ (Llama 3.3)  │
    │ RAG Docs    │         │              │         │              │
    └─────────────┘         └──────────────┘         └──────────────┘
         │                           │                           │
         ▼                           ▼                           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    EXTERNAL SERVICES & DATA                     │
    │  • yfinance (Stock prices, Market data)                         │
    │  • Supabase (Auth, Portfolio DB)                                │
    │  • Edge TTS (Azure Text-to-Speech)                              │
    │  • faster-whisper (Speech Recognition)                          │
    │  • LIC Policy Database (in-memory)                              │
    └─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Complete User Journey - Start to End

### **Phase 1: User Connection & Initialization**

```
1. USER LOADS WEBSITE
   └─ Browser opens http://localhost:8080
      ├─ Serves: index.html (React UI)
      ├─ Loads: Frontend assets
      │  ├─ Tailwind CSS
      │  ├─ Three.js (3D avatar rendering)
      │  └─ VRM Loader (Avatar animation)
      └─ Initializes WebSocket connection to ws://localhost:8765

2. WebSocket HANDSHAKE
   └─ handle_websocket() in app.py
      ├─ Adds client to connected_clients set
      ├─ Sends: "connection_status" message
      │         {"type": "connection_status", "status": "connected"}
      └─ Awaits incoming messages

3. FRONTEND READY
   └─ User sees:
      ├─ 3D Avatar (Veena AI) animated on canvas
      ├─ Microphone icon (for voice input)
      ├─ Text input box
      ├─ Chat history display
      └─ Dashboard button (market data)
```

---

### **Phase 2: User Provides Input (Voice or Text)**

#### **Option A: Voice Input**

```
1. USER CLICKS "START LISTENING"
   └─ Frontend sends: {"type": "start_listening"}
   └─ app.py broadcasts: {"type": "listening_started"}

2. AUDIO RECORDING PIPELINE
   └─ audio_producer() thread (runs continuously)
      ├─ Opens microphone stream via PyAudio
      │  └─ Sample rate: 16kHz, channels: 1, format: int16
      ├─ Records chunks: 4 seconds each
      │  └─ Frame buffer: 1024 samples
      ├─ Silence Detection (RMS-based)
      │  ├─ If >85% of frames are below SILENCE_THRESHOLD (800)
      │  └─ Chunk discarded
      └─ Non-silent chunks pushed to audio_queue
         └─ Queue size: 8 buffers (producer-consumer decoupling)

3. AUDIO CONSUMER THREAD
   └─ audio_consumer() thread
      ├─ Waits for chunks from queue
      ├─ Calls transcribe() on each chunk
      │  └─ Uses faster-whisper model (small, CPU optimized)
      │     ├─ Beam size: 3 (vs 5, faster)
      │     ├─ Compute type: int8 (quantized)
      │     └─ Language auto-detection (Hindi/English)
      ├─ Detects emotion from audio
      │  └─ Analyzes: words-per-second, amplitude
      │     ├─ stressed: fast speech + high amplitude
      │     ├─ calm: slow speech + low amplitude
      │     ├─ confused: detected keywords + question marks
      │     └─ neutral: default
      ├─ Checks for echo (user echoing AI response)
      │  └─ If >50% word overlap, skip
      └─ Broadcasts to frontend:
         └─ {"type": "user_message", "text": "...", "emotion_detected": "calm"}

4. LANGUAGE DETECTION
   └─ Check if text contains Devanagari characters (U+0900-U+097F)
      ├─ If Yes: language = "hi"
      └─ If No: language = "en"
```

#### **Option B: Text Input**

```
1. USER TYPES IN TEXT BOX
   └─ Frontend sends:
      {
        "type": "text_input",
        "text": "What is term insurance?",
        "language": "en",
        "emotion": "neutral"
      }

2. FRONTEND BROADCASTED
   └─ Converted to user_message event
   └─ Language auto-detected if not provided
```

---

### **Phase 3: Agent Processing & Response Generation**

```
1. INVOKE AGENT (process_user_input in app.py)
   └─ Set: is_thinking = True
   └─ Send broadcast: {"type": "user_message", ...}
   └─ Call: agent.invoke() in executor thread

2. AGENT WORKFLOW (agentic_rag.py)
   └─ build_agent() creates:
      ├─ LLM: ChatGroq (Llama 3.3 70B)
      ├─ Embeddings: OllamaEmbeddings (nomic-embed-text)
      ├─ Memory: ConversationBufferMemory (keeps conversation history)
      ├─ Tools: 13 specialized tools (see below)
      └─ Executor: ReactAgent (LLM decides which tools to use)

3. AGENT DECISION LOOP (ReAct pattern)
   ┌─ LLM receives user message + conversation history + tools
   ├─ LLM decides: "Do I need tools, or can I answer directly?"
   ├─ If needs tools, LLM chooses from:
   │
   │  A. rag_search_transcripts
   │     └─ Queries FAISS index for relevant documents
   │        ├─ Returns top 4 similar chunks
   │        └─ Used for policy/product questions
   │
   │  B. MARKET DATA TOOLS
   │     ├─ StockPrice
   │     │  └─ Input: ticker symbol (e.g., "RELIANCE", "TCS")
   │     │  └─ Calls: yfinance API
   │     │  └─ Returns: price, change %, day-high/low, volume
   │     │
   │     ├─ MarketSummary
   │     │  └─ Returns: NIFTY 50, SENSEX, top gainers/losers
   │     │  └─ Cached 60 seconds (avoid redundant API calls)
   │     │
   │     └─ IPOTracker
   │        └─ Returns: Upcoming IPOs, price bands, GMP
   │
   │  C. PORTFOLIO MANAGEMENT
   │     └─ PortfolioManager
   │        ├─ "view" → Shows all holdings with P&L
   │        │  ├─ Parallel price fetch (ThreadPoolExecutor)
   │        │  ├─ Calculates: avg-buy, current-value, P&L %
   │        │  └─ Shows: total invested, current value
   │        │
   │        └─ "add, SYMBOL, qty, price" → Adds to portfolio
   │           ├─ Tries Supabase first (if configured)
   │           └─ Falls back to local JSON DB
   │
   │  D. TAX & INVESTMENT TOOLS
   │     ├─ TaxCalculator
   │     │  └─ Compares: Old vs New tax regime
   │     │  └─ Calculates: Tax with 80C/80D deductions
   │     │
   │     ├─ SIPCalculator
   │     │  └─ Input: monthly_amount, annual_return%, years
   │     │  └─ Returns: future value, wealth gained, milestones
   │     │
   │     ├─ InsuranceGapAnalysis
   │     │  └─ Input: investments, insurance_cover, annual_income
   │     │  └─ Detects: under-insurance scenarios
   │     │
   │     └─ InvestmentComparator
   │        └─ Compares: LIC vs Stocks vs Mutual Funds
   │        └─ Scenario-aware (normal, war, recession, inflation)
   │
   │  E. LIC INSURANCE TOOLS
   │     ├─ LICPolicyInfo
   │     │  └─ Returns: detailed policy info (coverage, benefits, riders)
   │     │
   │     ├─ LICPolicyCatalogue
   │     │  └─ Lists: all 30+ available LIC policies
   │     │
   │     ├─ LICPolicyCompare
   │     │  └─ Side-by-side: 2-3 policies compared
   │     │
   │     └─ LICPolicyRecommend
   │        └─ Personalised recommendation based on user profile
   │
   └─ LLM assembles final response from tool outputs

4. RESPONSE GENERATION
   └─ LLM generates natural language response
      ├─ Incorporates tool outputs
      ├─ Maintains persona: "Veena" (friendly insurance advisor)
      ├─ Supports: Hindi or English based on language flag
      └─ Max iterations: prevents infinite loops

5. FALLBACK HANDLING
   └─ If agent timeout or error:
      ├─ Return predefined fallback:
      │  ├─ Hindi: "मुझे खेद है, मैं अभी आपकी बात ठीक से समझ नहीं पाई..."
      │  └─ English: "I'm sorry, I couldn't process that properly..."
```

---

### **Phase 4: Text-to-Speech & Audio Playback**

```
1. TTS INITIALIZATION (voice_service.py)
   └─ play_text_to_speech_stream(text, language)
   
2. VOICE SELECTION
   └─ Auto-detect if text contains Hindi (Devanagari)
      ├─ Hindi text → Voice: hi-IN-SwaraNeural (Edge TTS)
      └─ English text → Voice: en-IN-NeerjaNeural (Edge TTS)

3. SENTENCE SPLITTING
   └─ Split text on sentence boundaries: [.!?।]
      ├─ Merge chunks <3 chars with next chunk
      └─ Example: "Hello. How are you?" → ["Hello.", "How are you?"]

4. STREAMING PIPELINE
   └─ Async producer: Generate sentences to Edge TTS
      ├─ Each sentence sent to edge_tts API
      └─ Streamed to queue as MP3 bytes
   
   └─ Audio consumer: Play sentences back-to-back
      ├─ Uses pygame mixer for playback
      ├─ Waits for producer if queue empty
      ├─ Plays while next sentence is being generated
      └─ Result: Near-zero inter-sentence gap

5. AUDIO OUTPUT
   └─ Broadcast: {"type": "speaking_started"}
   └─ Play audio via speaker
   └─ Broadcast: {"type": "speaking_finished"}
   └─ Set: is_speaking = False

6. GRACEFUL CANCELLATION (Barge-in)
   └─ User can interrupt mid-speech
      ├─ Clicking microphone sets: _stop_event.set()
      └─ TTS playback stops at sentence boundary
```

---

### **Phase 5: Frontend Display & Avatar Animation**

```
1. RECEIVE AGENT RESPONSE
   └─ Frontend gets WebSocket message:
      {
        "type": "agent_response",
        "text": "Term insurance provides..."
      }

2. UPDATE CHAT UI
   └─ Add message to chat history
   └─ Display in glass-morphism chat bubble
   └─ Auto-scroll to latest message

3. ANIMATE AVATAR
   └─ Trigger avatar animation:
      ├─ Speaking animation (lips sync to audio)
      ├─ 3D model: Veena.vrm (VRM format)
      ├─ Uses: Three.js + VRMLoaderPlugin
      └─ Maintains animation loop during speech

4. BROADCAST EVENTS TIMELINE
   ├─ user_message → Display user text
   ├─ speaking_started → Start avatar animation
   ├─ agent_response → Display response text
   └─ speaking_finished → Stop animation
```

---

## 🧠 AI & Backend Components Deep Dive

### **1. FAISS Vector Database (Knowledge Base)**

```
FILE: faiss_rag.index/
├─ index.faiss         : Vector embeddings (binary)
└─ index.pkl           : Metadata mapping

INDEXING PROCESS (index_documents.py):
1. Load all .txt files from rag_docs/
   ├─ Calling Script.txt   : Insurance sales scripts
   ├─ Knowledge Base.txt   : Policy details, FAQs
2. Split into chunks
   ├─ Chunk size: 500 tokens
   ├─ Overlap: 50 tokens (context preservation)
3. Embed using Ollama (nomic-embed-text)
   └─ Produces 768-dim vectors for each chunk
4. Store in FAISS index
   └─ O(log n) similarity search

RETRIEVAL (in agentic_rag.py):
└─ When agent needs: rag_search_transcripts
   ├─ User query embedded → 768-dim vector
   ├─ FAISS searches: top 4 most similar chunks
   ├─ Returns: chunk text + source filename
   └─ LLM incorporates into response
```

### **2. Agent Architecture (LangChain React Agent)**

```
Agent Loop:
┌─────────────────────────────────────┐
│ THOUGHT: LLM reads tools & decides  │
├─────────────────────────────────────┤
│ ACTION: LLM chooses tool + inputs   │
├─────────────────────────────────────┤
│ OBSERVATION: Tool runs, returns     │
│ result                              │
├─────────────────────────────────────┤
│ REPEAT if more info needed          │
│ ELSE: FINAL ANSWER synthesized      │
└─────────────────────────────────────┘

LLM: ChatGroq
├─ Model: Llama 3.3 70B Versatile
├─ Temperature: 0 (deterministic)
├─ Max tokens: (default ~2K)
└─ System prompt: "Veena" persona

Memory:
├─ Type: ConversationBufferMemory
├─ Keeps: All conversation history
├─ No limit (may need ConversationSummaryMemory for long chats)
└─ Cleared on server restart
```

### **3. Finance Tools Module**

```
finance_tools.py functions:

1. get_stock_price(symbol)
   ├─ Add .NS suffix if needed (NSE)
   ├─ Fetch: yfinance API (cached)
   └─ Return: Price, change%, high/low, volume

2. portfolio_manager(action, ...)
   ├─ Backend: Supabase OR local JSON
   ├─ "view": Lists holdings with P&L
   │  ├─ Parallel fetch prices (ThreadPoolExecutor)
   │  ├─ Calc: avg-buy, current-value, P&L
   │  └─ Aggregate by symbol
   └─ "add": Stores new holding

3. get_market_summary()
   ├─ Fetch: NIFTY 50, SENSEX via yfinance
   ├─ Cache: 60 seconds
   ├─ Return: Market trend, top 5 gainers/losers
   └─ Broadcast to dashboard

4. calculate_tax(income, 80c, 80d)
   ├─ Old regime: Graduated slabs
   ├─ New regime: Flat 30% above threshold
   └─ Return: Tax comparison + recommendation

5. analyze_insurance_gap(investments, cover, income)
   ├─ Rule: Insurance ≥ 10x annual income
   ├─ Check: investments vs cover
   └─ Return: Gap analysis + recommendation

6. compare_investments(options, amount, years, scenario)
   ├─ Options: LIC, STOCKS, MUTUAL_FUNDS
   ├─ Scenario: normal/war/recession/inflation
   └─ Return: Side-by-side comparison with projections

7. sip_calculator(monthly, rate%, years)
   ├─ Formula: FV = PMT × [((1+r)^n - 1) / r] × (1+r)
   ├─ Milestones: Snapshots every 5 years
   └─ Return: FV, wealth gained, % return
```

### **4. LIC Policy Database**

```
lic_policies.py structure:

Database: 30+ LIC policies organized by category

CATEGORIES:
├─ Life/Endowment (8)
│  └─ Jeevan Anand, Jeevan Labh, Jeevan Umang, etc.
├─ Health/Medical (3)
│  └─ Arogya Rakshak, Cancer Cover, Jeevan Arogya
├─ Pension/ULIP (3)
│  └─ Jeevan Shanti, New Jeevan Nidhi, SIIP
├─ Child Plans (2)
│  └─ New Children's Money Back, Jeevan Tarun
└─ Women's Plans (2)
   └─ Aadhaar Stambh, Shila

Per-policy structure:
{
  "full_name": "LIC Jeevan Anand (Plan 915)",
  "category": "Endowment + Whole Life",
  "entry_age": "18 – 50 years",
  "maturity_age": "Up to 75 years",
  "policy_term": "15 – 35 years",
  "sum_assured": "Min ₹1,00,000 (no upper limit)",
  "key_benefits": [...],
  "riders_available": ["Accidental Death", "Term Assurance"],
  "tax_benefit": "80C (premium), 10(10D) (proceeds)",
  "loan_facility": true
}

API Functions:
├─ get_policy_info(query)
│  └─ Fuzzy search on policy name/category
├─ list_all_policies()
│  └─ Returns summary table of all policies
├─ compare_policies(names)
│  └─ Side-by-side comparison
└─ recommend_policy(profile)
   ├─ Input: age, income, goals, dependents
   └─ Returns: Top 3 recommendations with rationale
```

---

## 🌐 Frontend Components

### **HTML Pages**

```
1. index.html (Main Application)
   ├─ React UI in embedded <script>
   ├─ Tailwind CSS styling
   ├─ Three.js 3D canvas
   ├─ Chat interface with glass morphism
   ├─ WebSocket connection handler
   └─ Microphone & text input controls

2. agent-dashboard.html (Agent Portal)
   ├─ Policy management interface
   ├─ Market data display (NIFTY, SENSEX)
   ├─ Portfolio tracking
   └─ Supabase integration for auth

3. login.html / signup.html
   ├─ Supabase auth forms
   ├─ Email/password registration
   └─ User profile setup

4. customer-login.html
   └─ Customer authentication portal
```

### **JavaScript Libraries**

```
1. Three.js
   ├─ 3D scene rendering
   ├─ Camera & lighting
   └─ Real-time rendering loop

2. VRMLoaderPlugin
   ├─ Loads .vrm model files
   ├─ Supports: bone animations, morphs
   └─ Models: agent.vrm, veena.vrm

3. Supabase Client
   ├─ Authentication
   ├─ Real-time database
   └─ Row-level security
```

---

## 💾 Data Flow Diagrams

### **User Voice Input → Response Output**

```
┌─────────────┐
│ User speaks │
└──────┬──────┘
       │ (microphone captures audio)
       ▼
┌──────────────────┐
│ audio_producer   │ (records 4s chunks)
└──────┬───────────┘
       │ (silence filtered)
       ▼
┌──────────────────┐
│ audio_queue      │ (queue of wav buffers)
└──────┬───────────┘
       │
       ▼
┌──────────────────────────┐
│ audio_consumer           │ (transcribe thread)
└──────┬───────────────────┘
       │
       ├─ faster-whisper (Llama transcription)
       ├─ Language detection (Hindi/English)
       ├─ Emotion analysis (from audio)
       └─ Echo detection
       │
       ▼
┌──────────────────────────┐
│ WebSocket broadcast      │ (send transcription to frontend)
└──────┬───────────────────┘
       │
       ├─ Frontend updates chat
       └─ Frontend displays user message
       │
       ▼
┌──────────────────────────┐
│ process_user_input()     │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ agent.invoke()           │ (LangChain ReAct loop)
└──────┬───────────────────┘
       │
       ├─ Determine if tools needed
       ├─ Call appropriate tools
       │  ├─ RAG search
       │  ├─ Market data
       │  ├─ Portfolio manager
       │  ├─ Tax calculator
       │  ├─ Insurance analyzer
       │  └─ LIC policy lookup
       ├─ Synthesize response
       └─ Return final answer
       │
       ▼
┌──────────────────────────┐
│ WebSocket broadcast      │ (send response)
└──────┬───────────────────┘
       │
       ├─ Frontend updates chat
       ├─ Frontend plays TTS
       └─ Avatar animates
       │
       ▼
┌──────────────────────────┐
│ play_tts_and_notify()    │ (voice_service.py)
└──────┬───────────────────┘
       │
       ├─ Split text to sentences
       ├─ Async sentence → Edge TTS
       ├─ Stream audio playback
       └─ Broadcast completion event
       │
       ▼
┌──────────────────────────┐
│ Speaker output           │ (user hears response)
└──────────────────────────┘
```

### **Text Input → Response Flow (Simplified)**

```
User types in text box
       │
       ▼
Frontend sends: {"type": "text_input", "text": "...", "language": "..."}
       │
       ▼
handle_client_message() routes to process_user_input()
       │
       └─ [Same as above from agent.invoke() onwards]
```

---

## 🔐 Security & Authentication

```
1. WebSocket Connection
   ├─ No authentication (assumes localhost)
   └─ Could add: JWT tokens in connection headers

2. Supabase Integration (Optional)
   ├─ Email/password auth
   ├─ Row-level security on portfolio data
   ├─ API key in .env (SUPABASE_URL, SUPABASE_ANON_KEY)
   └─ Fallback: Local JSON storage if not configured

3. API Keys (Environment Variables)
   ├─ GROQ_API_KEY (LLM access)
   └─ Stored in .env file (not committed to git)

4. Local Storage
   ├─ Browser: Session/localStorage for frontend state
   └─ Backend: .json files for portfolio (no credentials)
```

---

## 🚀 Deployment Stack

```
RUNTIME:
├─ Python 3.8+
├─ asyncio (async WebSocket server)
├─ ThreadPoolExecutor (parallel tasks)
└─ threading (audio producer/consumer)

SERVICES:
├─ Ollama (local embeddings model)
├─ Groq API (remote LLM)
├─ Edge TTS API (Azure text-to-speech)
├─ yfinance (stock data)
├─ Supabase (optional auth + DB)
└─ faster-whisper (speech recognition)

HTTP SERVER:
├─ Python http.server (serves frontend)
├─ Port 8080 (frontend)
└─ WebSocket server: Port 8765

DATABASE:
├─ FAISS (vector DB, local)
├─ JSON files (portfolio DB, local)
└─ Supabase (optional, cloud)
```

---

## 📋 Configuration Files

```
.env
├─ GROQ_API_KEY=<your-key>
├─ SUPABASE_URL=<optional>
└─ SUPABASE_ANON_KEY=<optional>

requirements.txt
├─ langchain-groq
├─ langchain-ollama
├─ langchain-community
├─ faster-whisper
├─ pyaudio
├─ edge-tts
├─ pygame
├─ yfinance
├─ websockets
├─ numpy
└─ scikit-learn (for audio processing)

rag_docs/
├─ Calling Script.txt
└─ Knowledge Base.txt
```

---

## 🔄 Server Startup Sequence

```
1. python app.py
   ├─ Load environment (.env)
   ├─ Build agent (agentic_rag.build_agent())
   │  ├─ Initialize ChatGroq LLM
   │  ├─ Load FAISS index
   │  ├─ Wire up all tools
   │  └─ Create ConversationMemory
   ├─ Load Whisper model (small, CPU)
   ├─ Start HTTP server (port 8080)
   │  └─ Serves index.html
   ├─ Start WebSocket server (port 8765)
   │  └─ listen on all interfaces
   ├─ Start audio_producer thread
   │  └─ Opens microphone, records chunks
   ├─ Start audio_consumer thread
   │  └─ Transcribes chunks in parallel
   └─ Print: "Server running on http://localhost:8080"

2. Browser navigates to http://localhost:8080
   ├─ Loads index.html
   ├─ Connects to WebSocket
   └─ Receives: {"type": "connection_status"}

3. System Ready
   └─ User can start voice/text input
```

---

## 📊 Performance Optimizations

```
1. Audio Processing
   ├─ Chunk size: 4s (was 8s, 2x faster)
   ├─ Whisper model: small (3-4x faster than medium)
   ├─ Compute type: int8 quantization
   ├─ Beam size: 3 (faster, ~same quality)
   └─ Threading: audio_producer + audio_consumer parallel

2. LLM Inference
   ├─ Temperature: 0 (no randomness, faster)
   ├─ Model: Llama 3.3 70B (via Groq API, fast)
   └─ ThreadPoolExecutor: non-blocking agent calls

3. TTS Streaming
   ├─ Sentence-level streaming (vs. full paragraph)
   ├─ Async producer: generate while playing
   └─ First-chunk latency: ~0.3s (was 2-4s)

4. Data Retrieval
   ├─ FAISS: O(log n) similarity search
   ├─ Market summary: cached 60s
   ├─ Portfolio: parallel price fetch via ThreadPoolExecutor
   └─ Result: sub-second tool responses

5. WebSocket
   ├─ Ping/pong keepalive
   ├─ Async broadcast to all clients
   └─ Graceful connection cleanup
```

---

## ✅ Complete User Experience Timeline

```
T=0s   | User arrives at http://localhost:8080
T=0.5s | Page loads, 3D avatar appears
T=1s   | WebSocket connected, microphone ready
T=2s   | User clicks "Start Listening"
T=3s   | Audio captured & transcribed by Whisper
T=3.5s | Transcription sent to frontend
T=3.5s | Agent receives user input
T=4s   | Agent decides which tools to call
T=4.5s | Tools run (RAG search, market lookup, etc.)
T=5s   | Agent synthesizes response
T=5.5s | Response broadcast to frontend
T=5.5s | Chat bubble appears, TTS starts
T=6s   | First sentence playing on speaker
T=6.5s | Avatar animates, speaking visual feedback
T=8s   | Response complete, listening resumes
```

---

## 🎯 Key Takeaways

✅ **Architecture**: Async WebSocket server + agent executor + parallel threads
✅ **Intelligence**: LangChain ReAct agent with 13 specialized tools + RAG
✅ **Audio**: Fast transcription (Whisper) + streaming TTS (Edge TTS)
✅ **Data**: FAISS vector DB + real-time market data + LIC policy catalog
✅ **Frontend**: React UI + Three.js avatar + real-time WebSocket updates
✅ **Performance**: Optimized chunk size, model size, parallel processing, caching
✅ **Multimodal**: Voice, text, emotion detection, avatar animation
✅ **Extensible**: Easy to add new tools, policies, integrations

