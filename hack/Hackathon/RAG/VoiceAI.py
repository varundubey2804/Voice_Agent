from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore 
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.settings import Settings

import faiss  # ⚠️ Required for FAISS backend


class AIVoiceAssistant:
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        llm_model: str = "mistral",
        embed_model: str = "nomic-embed-text",
        faiss_index_path: str = "faiss_index.idx",
        kb_path: str = "hack/Hackathon/Knowledge Base.txt",
    ):
        self._llm = Ollama(base_url=ollama_url, model=llm_model, request_timeout=120.0)
        self._embed = OllamaEmbedding(base_url=ollama_url, model_name=embed_model)

        # ✅ Set global Settings
        Settings.llm = self._llm
        Settings.embed_model = self._embed

        self._index = self._build_vector_index(kb_path, faiss_index_path)

        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=ChatMemoryBuffer.from_defaults(token_limit=1500),
            system_prompt=self._system_prompt,
        )

        print("✅ Voice Assistant Ready with FAISS!")

    def chat(self, user_text: str) -> str:
        return self._chat_engine.chat(user_text).response

    def _build_vector_index(self, kb_path: str, faiss_index_path: str):
        reader = SimpleDirectoryReader(input_files=[kb_path])
        docs = reader.load_data()

        # 🔁 Load or create FAISS index
        try:
            faiss_index = faiss.read_index(faiss_index_path)
            vstore = FaissVectorStore(faiss_index=faiss_index)
            print("📂 Loaded existing FAISS index.")
        except:
            faiss_index = faiss.IndexFlatL2(768)
            vstore = FaissVectorStore(faiss_index=faiss_index)
            print("🆕 Created new FAISS index.")

        storage_ctx = StorageContext.from_defaults(vector_store=vstore)

        index = VectorStoreIndex.from_documents(docs, storage_context=storage_ctx)

        # 💾 Save FAISS index to disk
        faiss.write_index(faiss_index, faiss_index_path)

        return index

    @property
    def _system_prompt(self) -> str:
        return (
            '''You are "Veena," a female insurance agent for "ValuEnable life insurance".
Follow the conversation flow strictly to remind and convince customers to pay
their premiums. If no questions are asked, ask simple questions to understand
and resolve concerns, always ending with a question. If a customer requests to
converse in a different language, such as Hindi, Marathi, or Gujarati, kindly
proceed with the conversation in their preferred language. Use max 35 easy
english words to respond.'''
        )

