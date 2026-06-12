import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.core.config import settings
from backend.core.logger import logger
from backend.rag.retrieval import RAGService

class DocumentIngestion:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.docs_dir = settings.DOCS_DIR
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

    def _get_loader(self, file_path: str):
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            return PyPDFLoader(file_path)
        elif ext == '.txt':
            return TextLoader(file_path)
        elif ext == '.md':
            return UnstructuredMarkdownLoader(file_path)
        elif ext == '.docx':
            return Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def ingest_directory(self):
        """
        Ingests all supported documents from the DOCS_DIR.
        """
        all_splits = []
        for file in os.listdir(self.docs_dir):
            file_path = os.path.join(self.docs_dir, file)
            if os.path.isfile(file_path):
                try:
                    logger.info(f"Processing file: {file_path}")
                    loader = self._get_loader(file_path)
                    docs = loader.load()
                    splits = self.text_splitter.split_documents(docs)
                    all_splits.extend(splits)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

        if all_splits:
            self.rag_service.add_documents(all_splits)
            logger.info(f"Ingested {len(all_splits)} total document chunks.")
        else:
            logger.info("No new documents ingested.")

    def ingest_file(self, file_path: str):
        """Ingests a single file."""
        try:
            logger.info(f"Ingesting file: {file_path}")
            loader = self._get_loader(file_path)
            docs = loader.load()
            splits = self.text_splitter.split_documents(docs)
            if splits:
                self.rag_service.add_documents(splits)
        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {e}")
            raise
