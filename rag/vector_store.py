import os
from pathlib import Path
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from loguru import logger

from .embeddings import load_pdfs, chunk_documents

DEFAULT_PDF_FOLDER = "pdfs"
DEFAULT_INDEX_PATH = "rag/faiss_index"

def get_embeddings() -> OpenAIEmbeddings:
    """Get OpenAI embeddings model."""
    return OpenAIEmbeddings(model="text-embedding-3-small")

def create_vector_store(
    pdf_folder: str = DEFAULT_PDF_FOLDER,
    index_path: str = DEFAULT_INDEX_PATH,
    chunk_size: int = 500
) -> Optional[FAISS]:
    """Create FAISS vector store from PDFs and save to disk."""
    logger.info(f"Creating vector store from: {pdf_folder}")
    
    documents = load_pdfs(pdf_folder)
    if not documents:
        logger.error("No documents loaded. Vector store creation aborted.")
        return None
    
    text_file_path = Path("rag/extracted_text.txt")
    combined_text = "\n\n--- PAGE BREAK ---\n\n".join([doc.page_content for doc in documents])
    try:
        text_file_path.write_text(combined_text, encoding="utf-8")
        logger.info(f"Combined PDF text saved to: {text_file_path}")
    except Exception as e:
        logger.error(f"Error saving combined text: {e}")
        
    chunks = chunk_documents(documents, chunk_size=chunk_size)
    
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(index_path)
    logger.info(f"Vector store saved to: {index_path}")
    
    return vector_store


def load_vector_store(index_path: str = DEFAULT_INDEX_PATH) -> Optional[FAISS]:
    """Load existing FAISS vector store from disk."""
    if not Path(index_path).exists():
        logger.warning(f"No vector store found at: {index_path}")
        return None
    
    embeddings = get_embeddings()
    try:
        vector_store = FAISS.load_local(
            index_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Vector store loaded from: {index_path}")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return None


if __name__ == "__main__":
    """Run this script to create the vector store from PDFs."""
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    
    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_FOLDER
    create_vector_store(pdf_folder=folder)