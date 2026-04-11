"""
index_documents.py
Improvements:
  - Accepts CLI overrides for docs folder, index path, model, chunk params
  - Exits cleanly with a clear error if no .txt files are found
  - Progress counter per file instead of a single total
  - Split text per-document so metadata (source file) is preserved
  - Informative summary at the end (chunk count, index path)
"""

import argparse
import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ──────────────────────────────
# Defaults (overridable via CLI)
# ──────────────────────────────
DOCS_FOLDER      = "rag_docs"
FAISS_PATH       = "faiss_rag.index"
EMBED_MODEL_NAME = "nomic-embed-text"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50


def load_documents(docs_folder: Path) -> list[Document]:
    """Load all .txt files from *docs_folder*, returning LangChain Documents."""
    txt_files = sorted(docs_folder.glob("*.txt"))
    if not txt_files:
        print(f"❌  No .txt files found in '{docs_folder}'. Aborting.")
        sys.exit(1)

    docs: list[Document] = []
    for i, file in enumerate(txt_files, 1):
        text = file.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": file.name}))
            print(f"   [{i}/{len(txt_files)}] Loaded: {file.name} ({len(text):,} chars)")
        else:
            print(f"   [{i}/{len(txt_files)}] Skipped (empty): {file.name}")
    return docs


def main(
    docs_folder: str = DOCS_FOLDER,
    faiss_path: str = FAISS_PATH,
    embed_model: str = EMBED_MODEL_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> None:
    print("=" * 60)
    print("🚀  Veena AI — Document Indexer")
    print(f"    Source   : {docs_folder}")
    print(f"    Index    : {faiss_path}")
    print(f"    Model    : {embed_model}")
    print(f"    Chunk    : size={chunk_size}, overlap={chunk_overlap}")
    print("=" * 60)

    # 1. Load
    doc_path = Path(docs_folder)
    if not doc_path.is_dir():
        print(f"❌  Folder '{docs_folder}' does not exist. Create it and add .txt files.")
        sys.exit(1)

    print(f"\n📁  Loading documents from '{docs_folder}' …")
    documents = load_documents(doc_path)
    print(f"✅  {len(documents)} file(s) loaded.\n")

    # 2. Split — preserve source metadata per chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️   Split into {len(chunks)} chunk(s).\n")

    # 3. Embed
    print(f"🧠  Loading Ollama embeddings ({embed_model}) …")
    print("    (Make sure you have run: ollama pull nomic-embed-text)")
    embeddings = OllamaEmbeddings(model=embed_model)

    # 4. Build & save FAISS index
    print("🧱  Building FAISS index …")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(faiss_path)

    print(f"\n✅  Index saved  → {faiss_path}")
    print(f"    Chunks indexed: {len(chunks)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index .txt documents into a FAISS vector store.")
    parser.add_argument("--docs",    default=DOCS_FOLDER,      help="Folder containing .txt files")
    parser.add_argument("--index",   default=FAISS_PATH,        help="Output path for FAISS index")
    parser.add_argument("--model",   default=EMBED_MODEL_NAME,  help="Ollama embedding model name")
    parser.add_argument("--chunk",   default=CHUNK_SIZE,   type=int, help="Chunk size (tokens)")
    parser.add_argument("--overlap", default=CHUNK_OVERLAP, type=int, help="Chunk overlap (tokens)")
    args = parser.parse_args()

    main(
        docs_folder=args.docs,
        faiss_path=args.index,
        embed_model=args.model,
        chunk_size=args.chunk,
        chunk_overlap=args.overlap,
    )