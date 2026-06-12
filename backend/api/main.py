from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import aiofiles

from backend.core.config import settings
from backend.core.logger import logger
from backend.core.manager import manager
from backend.api.schemas.schemas import ChatRequest, ChatResponse, IngestionResponse, SearchRequest, SearchResponse

app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI Server starting up...")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Process text-based chat.
    """
    agent_text, audio_bytes = await manager.process_text_input(request.text)
    # Normally we might save the audio and return a URL, but for simplicity here we just return text.
    return ChatResponse(user_text=request.text, agent_text=agent_text)

@app.post("/transcribe-and-chat")
async def audio_chat_endpoint(audio: UploadFile = File(...)):
    """
    Process audio file, transcribe, get AI response, and synthesize TTS audio.
    """
    audio_data = await audio.read()
    user_text, agent_text, audio_response = await manager.process_audio_input(audio_data)

    # Returning raw WAV audio for simplicity; in practice, use multipart or a separate endpoint
    return Response(content=audio_response, media_type="audio/wav")

@app.post("/upload-documents", response_model=IngestionResponse)
async def upload_documents(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    """
    Upload documents for RAG.
    """
    saved_files = []
    for file in files:
        file_path = os.path.join(settings.DOCS_DIR, file.filename)
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        saved_files.append(file_path)

    def background_ingest(files_to_ingest):
        for f in files_to_ingest:
            try:
                manager.ingestion.ingest_file(f)
            except Exception as e:
                logger.error(f"Failed to ingest {f} in background: {e}")

    background_tasks.add_task(background_ingest, saved_files)

    return IngestionResponse(status="success", message=f"Received {len(files)} files. Ingestion started in background.")

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """
    Directly query the RAG vector store.
    """
    results = manager.rag.search(request.query, request.top_k)
    return SearchResponse(results=results)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
