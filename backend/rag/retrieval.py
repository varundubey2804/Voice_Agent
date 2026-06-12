import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.core.config import settings
from backend.core.logger import logger

class RAGService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.index_path = settings.FAISS_INDEX_PATH
        self.vectorstore = None
        self._load_or_create_index()

    def _load_or_create_index(self):
        try:
            if os.path.exists(self.index_path):
                self.vectorstore = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
                logger.info(f"Loaded existing FAISS index from {self.index_path}")
            else:
                logger.info("No existing FAISS index found. Will create one upon document ingestion.")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")

    def search(self, query: str, top_k: int = 3) -> list[str]:
        if not self.vectorstore:
            return []
        try:
            docs = self.vectorstore.similarity_search(query, k=top_k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Error searching FAISS index: {e}")
            return []

    def add_documents(self, documents):
        """
        Adds langchain Documents to the vectorstore and saves it.
        """
        try:
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vectorstore.add_documents(documents)

            self.vectorstore.save_local(self.index_path)
            logger.info(f"Successfully added {len(documents)} documents to FAISS index and saved to {self.index_path}")
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {e}")
            raise
