import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_community.llms import Ollama
from backend.core.config import settings
from backend.core.logger import logger

class LLMService:
    def __init__(self, model_name: str = None):
        """
        Initializes the LLM connection. Uses Ollama as it manages local models easily.
        """
        self.model_name = model_name or settings.LLM_MODEL
        try:
            self.llm = Ollama(model=self.model_name)
            logger.info(f"LLM Service initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Service: {e}")
            self.llm = None

        self.executor = ThreadPoolExecutor(max_workers=2)

    def generate_sync(self, prompt: str, system_prompt: str = "") -> str:
        """
        Synchronously generates a response.
        """
        if not self.llm:
            return "LLM not initialized."

        try:
            full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:" if system_prompt else prompt
            response = self.llm.invoke(full_prompt)
            return response
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "Sorry, I encountered an error generating a response."

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Asynchronously generates a response.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.generate_sync, prompt, system_prompt)
