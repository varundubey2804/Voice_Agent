"""
index_documents.py
------------------
One-time script to read documents from the 'rag_docs' folder,
generate embeddings using Ollama, and save them to a FAISS vector store.

Run this script once before starting the main application.
"""

import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
EMBED_MODEL_NAME = "nomic-embed-text"
DOCS_FOLDER      = "rag_docs"
FAISS_PATH       = "faiss_rag.index"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50

def main():
    print("🚀 Starting document indexing...")

    # 1. Ensure the folder exists
    doc_path = Path(DOCS_FOLDER)
    if not doc_path.is_dir():
        print(f"❌ Folder '{DOCS_FOLDER}' not found. Please create it and add .txt files.")
        return

    # 2. Load text from all .txt files
    all_text = []
    for file in doc_path.glob("*.txt"):
        content = file.read_text(encoding="utf-8", errors="ignore")
        all_text.append(content)

    if not all_text:
        print(f"⚠️ No .txt files found in '{DOCS_FOLDER}'. Creating an empty index.")
    
    print(f"📚 Loaded {len(all_text)} document(s).")

    # 3. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text("\n".join(all_text)) if all_text else []
    print(f"✂️ Split into {len(chunks)} chunks.")

    # 4. Create embeddings
    print(f"🧠 Loading embedding model '{EMBED_MODEL_NAME}'...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME)

    # 5. Build FAISS index
    print("🧱 Building FAISS index...")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # 6. Save index
    vectorstore.save_local(FAISS_PATH)
    print(f"✅ Index saved to: {FAISS_PATH}")

if __name__ == "__main__":
    main()
