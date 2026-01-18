"""
index_documents.py
"""
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
# ✅ CHANGED: Use Ollama here too
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

DOCS_FOLDER      = "rag_docs"
FAISS_PATH       = "faiss_rag.index"
EMBED_MODEL_NAME = "nomic-embed-text"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50

def main():
    print(f"🚀 Starting document indexing (Ollama: {EMBED_MODEL_NAME})...")

    # 1. Load Documents
    doc_path = Path(DOCS_FOLDER)
    if not doc_path.is_dir():
        print(f"❌ Create '{DOCS_FOLDER}' and add .txt files first.")
        return

    all_text = []
    for file in doc_path.glob("*.txt"):
        content = file.read_text(encoding="utf-8", errors="ignore")
        all_text.append(content)

    print(f"📚 Loaded {len(all_text)} files.")

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_text("\n".join(all_text)) if all_text else []

    # 3. Embed (Using Ollama)
    print(f"🧠 Loading Ollama Embeddings ({EMBED_MODEL_NAME})...")
    # Ensure you have run: ollama pull nomic-embed-text
    embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME)

    # 4. Save
    print("🧱 Building FAISS index...")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local(FAISS_PATH)
    print(f"✅ Index saved to: {FAISS_PATH}")

if __name__ == "__main__":
    main()