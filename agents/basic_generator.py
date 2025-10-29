"""Basic Generator Agent for simple RAG queries"""

from agents.base_agent import BaseAgent
from vector_store import VectorStore
from utils.prompt_templates import BASIC_GENERATOR_PROMPT
from config import BASIC_GENERATOR_CONFIG, AGENT_CONFIG
from typing import Dict, Any, Optional


class BasicGeneratorAgent(BaseAgent):
    """Basic generator agent that performs simple retrieval and generation"""
    
    def __init__(self, vector_store: VectorStore):
        super().__init__(config=AGENT_CONFIG)
        self.vector_store = vector_store
        self.n_results = BASIC_GENERATOR_CONFIG["n_results"]
    
    def generate_answer(
        self, 
        query: str, 
        n_results: Optional[int] = None,
        debug: bool = False
    ) -> Dict[str, Any]:
        """Generate answer for a query using basic RAG"""
        n_results = n_results or self.n_results
        
        if debug:
            print(f"[Basic] Retrieving {n_results} chunks for query...")
        
        # Retrieve relevant documents
        results = self.vector_store.query(query, n_results=n_results)
        retrieved_docs = results["documents"]
        
        if debug:
            print(f"[Basic] Retrieved {len(retrieved_docs)} chunks")
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "context": "",
                "retrieved_chunks": [],
                "metadata": {
                    "agent": "basic",
                    "n_chunks": 0,
                    "technique": "simple_retrieval"
                }
            }
        
        # Combine context
        context = "\n\n".join(retrieved_docs)
        
        if debug:
            print(f"[Basic] Generating answer...")
        
        # Generate answer
        prompt = BASIC_GENERATOR_PROMPT.format(
            context=context,
            query=query
        )
        
        answer = self.generate(prompt)
        
        if debug:
            print(f"[Basic] Answer generated")
        
        return {
            "answer": answer,
            "context": context,
            "retrieved_chunks": retrieved_docs,
            "metadata": {
                "agent": "basic",
                "n_chunks": len(retrieved_docs),
                "technique": "simple_retrieval"
            }
        }

