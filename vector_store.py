"""Vector Store module for ChromaDB operations"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from config import VECTOR_STORE_CONFIG


class VectorStore:
    """Manages vector database operations using ChromaDB"""
    
    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or VECTOR_STORE_CONFIG["collection_name"]
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.embedding_model = SentenceTransformer(VECTOR_STORE_CONFIG["embedding_model"])
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add documents to the vector store with optional metadata"""
        if not documents:
            return
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Generate IDs if not provided
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Add to collection
        self.collection.add(
            documents=documents,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(documents)
        )
    
    def query(self, query: str, n_results: int = 3, metadata_filter: Dict = None) -> Dict[str, Any]:
        """Query the vector store and return similar documents"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Query with optional metadata filter
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=metadata_filter
        )
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
        }
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.embedding_model.encode(text).tolist()
    
    def update_collection(self):
        """Reinitialize collection (useful for updates)"""
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
    
    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(name=self.collection_name)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "count": count,
        }

