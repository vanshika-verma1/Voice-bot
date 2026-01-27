from typing import List, Optional

try:
    from .vector_store import load_vector_store, DEFAULT_INDEX_PATH
except (ImportError, ValueError):
    from vector_store import load_vector_store, DEFAULT_INDEX_PATH

class RAGRetriever:
    """Singleton retriever that loads vector store once."""
    
    _instance: Optional["RAGRetriever"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._vector_store = load_vector_store()
        self._initialized = True
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """Search for relevant documents."""
        if self._vector_store is None:
            return []
        
        docs = self._vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def get_context(self, query: str, k: int = 3) -> str:
        """Get formatted context string for LLM."""
        results = self.search(query, k=k)
        if not results:
            return ""
        
        context = "\n\n---\n\n".join(results)
        return f"Relevant information:\n{context}"
    
    @property
    def is_available(self) -> bool:
        """Check if vector store is loaded."""
        return self._vector_store is not None

_retriever: Optional[RAGRetriever] = None


def get_relevant_context(query: str, k: int = 3) -> str:
    """Convenience function to get context for a query."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever.get_context(query, k=k)