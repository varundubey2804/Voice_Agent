import asyncio
from typing import Optional
from backend.core.logger import logger
from backend.audio.wakeword import WakeWordDetector
from backend.audio.vad import VADService
from backend.stt.whisper_service import WhisperSTT
from backend.stt.lang_detect import LanguageDetector
from backend.rag.retrieval import RAGService
from backend.rag.ingestion import DocumentIngestion
from backend.llm.service import LLMService
from backend.agents.graph import AssistantAgent
from backend.tts.piper_service import PiperTTSService

class ConversationManager:
    """
    Central orchestration layer for the AI assistant pipeline.
    """
    def __init__(self):
        logger.info("Initializing ConversationManager components...")
        self.wakeword = WakeWordDetector()
        self.vad = VADService()
        self.stt = WhisperSTT()
        self.lang_detect = LanguageDetector()
        self.rag = RAGService()
        self.ingestion = DocumentIngestion(self.rag)
        self.llm = LLMService()
        self.agent = AssistantAgent(self.llm, self.rag)
        self.tts = PiperTTSService()
        logger.info("ConversationManager initialized.")

    async def process_audio_input(self, audio_data: bytes) -> tuple[str, str, bytes]:
        """
        Process user audio to text, run agent, and return TTS audio.
        Returns: (user_text, agent_text, tts_audio_bytes)
        """
        try:
            # 1. STT
            user_text = await self.stt.transcribe(audio_data)
            if not user_text:
                return ("", "Could not hear anything.", b"")

            logger.info(f"User: {user_text}")

            # 2. Language Detection
            lang_code, confidence = self.lang_detect.detect_language(user_text)
            logger.info(f"Detected Language: {lang_code} (Confidence: {confidence})")

            # 3. Agent & LLM (RAG included inside Agent Graph)
            agent_response = await self.agent.invoke(user_text)
            logger.info(f"Agent: {agent_response}")

            # 4. TTS
            audio_response = await self.tts.synthesize(agent_response)

            return (user_text, agent_response, audio_response)

        except Exception as e:
            logger.error(f"Error in process_audio_input pipeline: {e}")
            return ("", "An internal error occurred.", b"")

    async def process_text_input(self, text: str) -> tuple[str, bytes]:
        """
        Process user text, run agent, and return text and TTS audio.
        Returns: (agent_text, tts_audio_bytes)
        """
        try:
            # Language Detection
            lang_code, confidence = self.lang_detect.detect_language(text)
            logger.info(f"Detected Language: {lang_code} (Confidence: {confidence})")

            # Agent logic
            agent_response = await self.agent.invoke(text)

            # TTS
            audio_response = await self.tts.synthesize(agent_response)
            return (agent_response, audio_response)
        except Exception as e:
            logger.error(f"Error in process_text_input: {e}")
            return ("An internal error occurred.", b"")

manager = ConversationManager()
