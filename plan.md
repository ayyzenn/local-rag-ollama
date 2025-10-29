# Agentic RAG System with 3-Agent Architecture

## Architecture Overview

```
User Query → Router Agent → Basic Generator Agent → Answer Evaluation
                                    ↓ (if insufficient)
                            Advanced Agent (Query Decomposition + HyDE + Multi-Query)
                                    ↓
                            Final Answer + Metadata
```

## Implementation Steps

### 1. Setup Gemini API Integration

**File: `config.py` (new)**

- Store Gemini API key configuration
- Define model settings (gemini-1.5-flash)
- Configure agent parameters (temperature, max_tokens)

**File: `requirements.txt`**

- Add: `google-generativeai>=0.3.0`
- Add: `python-dotenv>=1.0.0` (for .env file)

**File: `.env` (new)**

- Store Gemini API key securely

### 2. Create Agent Base Classes

**File: `agents/base_agent.py` (new)**

- Abstract Agent class with common methods
- Gemini API wrapper for LLM calls
- Response formatting utilities

### 3. Implement Router Agent

**File: `agents/router_agent.py` (new)**

- Analyzes query complexity
- Routes to Basic Generator first
- Evaluates basic answer sufficiency
- Routes to Advanced Agent if needed
- Decision criteria:
  - Answer completeness check
  - Confidence score evaluation
  - Context relevance assessment

### 4. Implement Basic Generator Agent

**File: `agents/basic_generator.py` (new)**

- Simple retrieval (top 2-3 chunks)
- Straightforward prompt to Gemini
- Fast response generation
- Similar to current `rag_local_ollama.py` but with Gemini

### 5. Implement Advanced Generator Agent

**File: `agents/advanced_generator.py` (new)**

**Technique 1: Query Decomposition**

- Detect complex/multi-part queries
- Break into sub-queries using Gemini
- Retrieve documents for each sub-query
- Synthesize combined answer

**Technique 2: HyDE (Hypothetical Document Embeddings)**

- Generate hypothetical ideal answer using Gemini
- Embed the hypothetical answer
- Search for documents similar to the hypothetical answer
- Use real documents to ground the final response

**Technique 3: Multi-Query Retrieval**

- Generate 3-5 query variations using Gemini
- Retrieve documents for each variation
- Combine and deduplicate results
- Use expanded context for generation

### 6. Create Orchestrator/Main Pipeline

**File: `agentic_rag.py` (new)**

- Initialize all three agents
- Manage agent workflow:

  1. Router receives query
  2. Basic Generator attempts answer
  3. Router evaluates result
  4. Advanced Generator activates if needed
  5. Return final answer with metadata

- Conversation flow options (configurable):
  - Silent mode: final answer only
  - Verbose mode: show routing + agent used
  - Debug mode: show all steps

### 7. Update Vector Database Module

**File: `vector_store.py` (new)**

- Refactor ChromaDB operations from `rag_local_ollama.py`
- Methods: add_documents, query, update, delete
- Support variable n_results for different agents
- Maintain existing sentence-transformers embeddings

### 8. Enhance Preprocessing

**File: `preprocess.py`**

- Keep existing functionality
- Add metadata extraction (document source, type)
- Improve chunking with overlap (e.g., 20% overlap between chunks)

### 9. Configuration & Utilities

**File: `utils/prompt_templates.py` (new)**

- Router evaluation prompts
- Basic generator prompts
- Advanced generator prompts (decomposition, HyDE, multi-query)
- Answer sufficiency check prompts

**File: `utils/evaluator.py` (new)**

- Answer quality scoring
- Completeness detection
- Confidence assessment

### 10. CLI Interface

**File: `agentic_rag.py`**

- Interactive query loop
- Display options: silent/verbose/debug
- Show which agent was used
- Display reasoning steps if requested

## Key Files to Create/Modify

**New Files:**

- `config.py` - Configuration
- `.env` - API keys
- `.gitignore` - Protect .env
- `agents/base_agent.py`
- `agents/router_agent.py`
- `agents/basic_generator.py`
- `agents/advanced_generator.py`
- `vector_store.py`
- `utils/prompt_templates.py`
- `utils/evaluator.py`
- `agentic_rag.py`

**Modified Files:**

- `requirements.txt` - Add Gemini dependencies
- `preprocess.py` - Enhanced chunking

**Preserved Files:**

- `rag_local_ollama.py` - Keep as reference/legacy
- `docs/*` - Unchanged

## Advanced Techniques Implementation Details

### Query Decomposition

```
Complex Query → Gemini analyzes → Extract sub-queries → 
Retrieve for each → Synthesize final answer
```

### HyDE

```
Query → Gemini generates hypothetical answer → 
Embed hypothetical → Search similar docs → 
Generate grounded answer from real docs
```

### Multi-Query

```
Query → Gemini generates variations → 
Retrieve for all variations → Merge results → 
Generate from enriched context
```

## Output Formats

**Option A: Silent (final answer only)**

```
Answer: [Generated response]
```

**Option B: Verbose (show routing)**

```
Router Decision: Using Basic Agent
Answer: [Generated response]
---
Router Decision: Basic insufficient, using Advanced Agent (Query Decomposition)
Answer: [Generated response]
```

**Option C: Debug (all steps)**

```
[Router] Analyzing query complexity...
[Router] Routing to Basic Agent
[Basic] Retrieved 2 chunks
[Basic] Generated answer
[Router] Evaluating answer... Insufficient
[Router] Routing to Advanced Agent
[Advanced] Technique: Query Decomposition
[Advanced] Sub-queries: Q1, Q2, Q3
[Advanced] Retrieved 6 chunks total
[Advanced] Final answer generated
Answer: [Generated response]
```