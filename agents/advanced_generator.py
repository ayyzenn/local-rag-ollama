"""Advanced Generator Agent with Query Decomposition, HyDE, and Multi-Query techniques"""

from agents.base_agent import BaseAgent
from vector_store import VectorStore
from utils.prompt_templates import (
    DECOMPOSITION_PROMPT,
    DECOMPOSITION_SYNTHESIS_PROMPT,
    HYDE_PROMPT,
    HYDE_GENERATION_PROMPT,
    MULTI_QUERY_PROMPT,
    ADVANCED_GENERATION_PROMPT,
    BASIC_GENERATOR_PROMPT
)
from config import ADVANCED_GENERATOR_CONFIG, AGENT_CONFIG
from typing import Dict, Any, List, Optional


class AdvancedGeneratorAgent(BaseAgent):
    """Advanced generator agent using multiple RAG techniques"""
    
    def __init__(self, vector_store: VectorStore):
        super().__init__(config=AGENT_CONFIG)
        self.vector_store = vector_store
        self.config = ADVANCED_GENERATOR_CONFIG
    
    def generate_answer(
        self, 
        query: str,
        techniques: Optional[List[str]] = None,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Generate answer using advanced techniques
        
        Args:
            query: User query
            techniques: List of techniques to use ['decomposition', 'hyde', 'multi_query']
                       If None, uses all techniques
            debug: Enable debug output
        """
        techniques = techniques or ["decomposition", "hyde", "multi_query"]
        
        all_context = []
        technique_metadata = {}
        
        # Technique 1: Query Decomposition
        if "decomposition" in techniques:
            if debug:
                print("[Advanced] Using Query Decomposition technique...")
            
            decomp_result = self._query_decomposition(query, debug=debug)
            if decomp_result:
                all_context.extend(decomp_result["context_chunks"])
                technique_metadata["decomposition"] = decomp_result["metadata"]
        
        # Technique 2: HyDE
        if "hyde" in techniques:
            if debug:
                print("[Advanced] Using HyDE technique...")
            
            hyde_result = self._hyde_retrieval(query, debug=debug)
            if hyde_result:
                all_context.extend(hyde_result["context_chunks"])
                technique_metadata["hyde"] = hyde_result["metadata"]
        
        # Technique 3: Multi-Query
        if "multi_query" in techniques:
            if debug:
                print("[Advanced] Using Multi-Query technique...")
            
            multi_result = self._multi_query_retrieval(query, debug=debug)
            if multi_result:
                all_context.extend(multi_result["context_chunks"])
                technique_metadata["multi_query"] = multi_result["metadata"]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_context = []
        for chunk in all_context:
            chunk_hash = hash(chunk)
            if chunk_hash not in seen:
                seen.add(chunk_hash)
                unique_context.append(chunk)
        
        if not unique_context:
            return {
                "answer": "I couldn't find relevant information to answer your question using advanced retrieval techniques.",
                "context": "",
                "retrieved_chunks": [],
                "metadata": {
                    "agent": "advanced",
                    "techniques_used": techniques,
                    "technique_details": technique_metadata
                }
            }
        
        if debug:
            print(f"[Advanced] Combined {len(unique_context)} unique chunks from all techniques")
            print("[Advanced] Generating final answer...")
        
        # Generate final answer from combined context
        combined_context = "\n\n".join(unique_context)
        prompt = ADVANCED_GENERATION_PROMPT.format(
            query=query,
            context=combined_context
        )
        
        answer = self.generate(prompt)
        
        return {
            "answer": answer,
            "context": combined_context,
            "retrieved_chunks": unique_context,
            "metadata": {
                "agent": "advanced",
                "n_chunks": len(unique_context),
                "techniques_used": techniques,
                "technique_details": technique_metadata
            }
        }
    
    def _query_decomposition(self, query: str, debug: bool = False) -> Optional[Dict[str, Any]]:
        """Query Decomposition: Break complex query into sub-queries"""
        try:
            # Step 1: Decompose query
            decomp_prompt = DECOMPOSITION_PROMPT.format(query=query)
            decomp_response = self.generate_json(decomp_prompt)
            
            sub_queries = decomp_response.get("sub_queries", [])
            
            if debug:
                print(f"[Advanced/Decomposition] Generated {len(sub_queries)} sub-queries")
            
            if not sub_queries:
                return None
            
            # Step 2: Retrieve for each sub-query
            all_chunks = []
            sub_answers = []
            n_results = self.config["query_decomposition"]["n_results_per_query"]
            
            for i, sub_query in enumerate(sub_queries):
                if debug:
                    print(f"[Advanced/Decomposition] Processing sub-query {i+1}: {sub_query[:50]}...")
                
                results = self.vector_store.query(sub_query, n_results=n_results)
                chunks = results["documents"]
                all_chunks.extend(chunks)
                
                # Generate answer for sub-query
                if chunks:
                    context = "\n\n".join(chunks)
                    sub_prompt = BASIC_GENERATOR_PROMPT.format(
                        context=context,
                        query=sub_query
                    )
                    sub_answer = self.generate(sub_prompt)
                    sub_answers.append(f"Sub-question {i+1}: {sub_query}\nAnswer: {sub_answer}")
            
            # Step 3: Synthesize final answer
            if sub_answers:
                synthesis_prompt = DECOMPOSITION_SYNTHESIS_PROMPT.format(
                    query=query,
                    sub_answers="\n\n".join(sub_answers)
                )
                final_answer = self.generate(synthesis_prompt)
            else:
                final_answer = "Could not generate answer from decomposed queries."
            
            return {
                "answer": final_answer,
                "context_chunks": all_chunks,
                "metadata": {
                    "n_sub_queries": len(sub_queries),
                    "sub_queries": sub_queries,
                    "n_chunks": len(all_chunks)
                }
            }
        except Exception as e:
            if debug:
                print(f"[Advanced/Decomposition] Error: {str(e)}")
            return None
    
    def _hyde_retrieval(self, query: str, debug: bool = False) -> Optional[Dict[str, Any]]:
        """HyDE: Generate hypothetical answer, then retrieve similar documents"""
        try:
            # Step 1: Generate hypothetical answer
            hyde_prompt = HYDE_PROMPT.format(query=query)
            
            if debug:
                print("[Advanced/HyDE] Generating hypothetical answer...")
            
            hypothetical_answer = self.generate(hyde_prompt)
            
            if debug:
                print(f"[Advanced/HyDE] Generated hypothetical answer ({len(hypothetical_answer)} chars)")
            
            # Step 2: Embed hypothetical answer and search
            n_results = self.config["hyde"]["n_results"]
            results = self.vector_store.query(hypothetical_answer, n_results=n_results)
            retrieved_chunks = results["documents"]
            
            if debug:
                print(f"[Advanced/HyDE] Retrieved {len(retrieved_chunks)} chunks based on hypothetical answer")
            
            # Step 3: Generate grounded answer from real documents
            if retrieved_chunks:
                context = "\n\n".join(retrieved_chunks)
                generation_prompt = HYDE_GENERATION_PROMPT.format(
                    query=query,
                    context=context
                )
                answer = self.generate(generation_prompt)
            else:
                answer = "Could not find relevant documents using HyDE technique."
            
            return {
                "answer": answer,
                "context_chunks": retrieved_chunks,
                "metadata": {
                    "hypothetical_answer": hypothetical_answer[:200],  # Truncate for metadata
                    "n_chunks": len(retrieved_chunks)
                }
            }
        except Exception as e:
            if debug:
                print(f"[Advanced/HyDE] Error: {str(e)}")
            return None
    
    def _multi_query_retrieval(self, query: str, debug: bool = False) -> Optional[Dict[str, Any]]:
        """Multi-Query: Generate query variations and retrieve for each"""
        try:
            # Step 1: Generate query variations
            multi_prompt = MULTI_QUERY_PROMPT.format(query=query)
            variations_response = self.generate_json(multi_prompt)
            
            variations = variations_response.get("variations", [])
            
            if debug:
                print(f"[Advanced/Multi-Query] Generated {len(variations)} query variations")
            
            if not variations:
                # Fallback: create simple variations
                variations = [
                    query,
                    f"What is {query}?",
                    f"Tell me about {query}",
                    f"Explain {query}"
                ]
            
            # Step 2: Retrieve for each variation
            all_chunks = []
            n_results = self.config["multi_query"]["n_results_per_variation"]
            
            for i, variation in enumerate(variations):
                if debug:
                    print(f"[Advanced/Multi-Query] Retrieving for variation {i+1}: {variation[:50]}...")
                
                results = self.vector_store.query(variation, n_results=n_results)
                chunks = results["documents"]
                all_chunks.extend(chunks)
            
            # Step 3: Generate answer from combined results
            if all_chunks:
                # Remove duplicates
                unique_chunks = list(dict.fromkeys(all_chunks))  # Preserves order
                context = "\n\n".join(unique_chunks)
                
                generation_prompt = ADVANCED_GENERATION_PROMPT.format(
                    query=query,
                    context=context
                )
                answer = self.generate(generation_prompt)
            else:
                answer = "Could not find relevant documents using multi-query technique."
            
            return {
                "answer": answer,
                "context_chunks": all_chunks,
                "metadata": {
                    "n_variations": len(variations),
                    "variations": variations,
                    "n_chunks": len(all_chunks)
                }
            }
        except Exception as e:
            if debug:
                print(f"[Advanced/Multi-Query] Error: {str(e)}")
            return None

