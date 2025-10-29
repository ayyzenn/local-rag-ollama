"""Main Agentic RAG System Orchestrator"""

import os
import sys
from vector_store import VectorStore
from preprocess import preprocess_documents
from agents.basic_generator import BasicGeneratorAgent
from agents.advanced_generator import AdvancedGeneratorAgent
from agents.router_agent import RouterAgent
from config import GEMINI_API_KEY


def initialize_system(doc_folder: str = "docs", force_rebuild: bool = False):
    """Initialize the RAG system with vector store"""
    print("🚀 Initializing Agentic RAG System...")
    
    # Check API key
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        print("❌ Error: GEMINI_API_KEY not set in .env file")
        print("   Please add your API key to .env file:")
        print("   GEMINI_API_KEY=your_actual_api_key")
        sys.exit(1)
    
    # Initialize vector store
    print("📚 Loading vector store...")
    vector_store = VectorStore()
    
    # Check if collection is empty or needs rebuilding
    collection_info = vector_store.get_collection_info()
    
    if collection_info["count"] == 0 or force_rebuild:
        if force_rebuild:
            print("🔄 Rebuilding vector store...")
            vector_store.delete_collection()
            vector_store = VectorStore()
        else:
            print("📄 Processing documents...")
        
        # Load and preprocess documents
        documents_data = preprocess_documents(doc_folder)
        
        if not documents_data:
            print(f"⚠️  No documents found in '{doc_folder}' folder")
            return None, None
        
        # Extract texts and metadatas
        texts = [doc["text"] for doc in documents_data]
        metadatas = [doc["metadata"] for doc in documents_data]
        
        # Add to vector store
        print(f"💾 Storing {len(texts)} document chunks...")
        vector_store.add_documents(texts, metadatas)
        print(f"✅ Vector store initialized with {len(texts)} chunks")
    else:
        print(f"✅ Vector store ready ({collection_info['count']} chunks)")
    
    # Initialize agents
    print("🤖 Initializing agents...")
    basic_agent = BasicGeneratorAgent(vector_store)
    advanced_agent = AdvancedGeneratorAgent(vector_store)
    router_agent = RouterAgent(basic_agent, advanced_agent)
    
    print("✅ System ready!\n")
    return router_agent, vector_store


def format_output(result: dict, mode: str = "silent"):
    """Format the output based on mode"""
    if mode == "silent":
        # Only show final answer
        print("\n" + "="*60)
        print("Answer:")
        print("="*60)
        print(result["answer"])
        print("="*60 + "\n")
    
    elif mode == "verbose":
        # Show routing decision and agent used
        metadata = result["metadata"]
        routing = metadata.get("routing", {})
        
        print("\n" + "="*60)
        print("Router Decision:")
        print("="*60)
        
        if routing.get("strategy") == "basic_only":
            print("✓ Used Basic Generator Agent (answer sufficient)")
        else:
            print("✓ Used Advanced Generator Agent (Basic was insufficient)")
            if "techniques_used" in metadata:
                techniques = metadata["techniques_used"]
                print(f"  Techniques: {', '.join(techniques)}")
        
        print("\n" + "="*60)
        print("Answer:")
        print("="*60)
        print(result["answer"])
        print("="*60 + "\n")
    
    elif mode == "debug":
        # Show all steps
        metadata = result["metadata"]
        routing = metadata.get("routing", {})
        agent_used = metadata.get("agent", "unknown")
        
        print("\n" + "="*60)
        print("DEBUG INFO")
        print("="*60)
        print(f"Agent Used: {agent_used}")
        print(f"Routing Strategy: {routing.get('strategy', 'unknown')}")
        
        if "evaluation" in routing:
            eval_data = routing["evaluation"]
            print(f"\nAnswer Evaluation:")
            print(f"  - Sufficient: {eval_data.get('sufficient', False)}")
            print(f"  - Completeness: {eval_data.get('completeness_score', 0):.2f}")
            print(f"  - Confidence: {eval_data.get('confidence_score', 0):.2f}")
        
        if "technique_details" in metadata:
            print(f"\nAdvanced Techniques Details:")
            for tech, details in metadata["technique_details"].items():
                print(f"  {tech}: {details}")
        
        print(f"\nRetrieved Chunks: {len(result['retrieved_chunks'])}")
        print("\n" + "="*60)
        print("Answer:")
        print("="*60)
        print(result["answer"])
        print("="*60 + "\n")


def main():
    """Main CLI interface"""
    print("="*60)
    print("🧠 Agentic RAG System")
    print("="*60)
    print("\nAvailable output modes:")
    print("  - silent: Final answer only")
    print("  - verbose: Show routing decision")
    print("  - debug: Show all steps and details")
    print("\nPress Ctrl+C to exit\n")
    
    # Initialize system
    router_agent, vector_store = initialize_system()
    
    if router_agent is None:
        return
    
    # Ask for output mode
    mode = input("Select output mode [silent/verbose/debug] (default: verbose): ").strip().lower()
    if mode not in ["silent", "verbose", "debug"]:
        mode = "verbose"
    
    print(f"\nMode: {mode}\n")
    print("="*60)
    
    # Main query loop
    while True:
        try:
            query = input("\nAsk your question (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print()  # Empty line for better formatting
            
            # Route and generate
            result = router_agent.route_and_generate(
                query=query,
                mode=mode,
                debug=(mode == "debug")
            )
            
            # Format and display output
            format_output(result, mode=mode)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            if mode == "debug":
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()

