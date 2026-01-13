import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger

def load_pdfs(pdf_folder: str) -> List[Document]:
    """
    Load all PDF files from a folder.
    Uses PyMuPDF which is fast and handles most text and images well.
    For OCR, LangChain's PyMuPDFLoader can be used with extract_images=True
    if desired, but we'll stick to robust text extraction first.
    """
    pdf_path = Path(pdf_folder)
    if not pdf_path.exists():
        logger.error(f"PDF folder not found: {pdf_folder}")
        return []
    
    documents = []
    pdf_files = list(pdf_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in: {pdf_folder}")
        return []
    
    for pdf_file in pdf_files:
        try:
            # PyMuPDFLoader is generally better for complex layouts
            loader = PyMuPDFLoader(str(pdf_file))
            docs = loader.load()
            documents.extend(docs)
            logger.info(f"Loaded: {pdf_file.name} ({len(docs)} pages)")
        except Exception as e:
            logger.error(f"Error loading {pdf_file.name}: {e}")
    
    logger.info(f"Total pages loaded: {len(documents)}")
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """Split documents into smaller chunks for better retrieval."""
    if not documents:
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks

if __name__ == "__main__":
    # Test loading
    from dotenv import load_dotenv
    load_dotenv()
    docs = load_pdfs("pdfs")
    chunks = chunk_documents(docs)
    print(f"Test: {len(chunks)} chunks created.")
