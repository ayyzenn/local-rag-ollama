# Agentic RAG System - Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Gemini API Key

1. Get your Gemini API key from: https://aistudio.google.com/app/apikey
2. Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

### 3. Run the System

```bash
python agentic_rag.py
```

## System Architecture

The system consists of 3 agents:

1. **Router Agent**: Routes queries to appropriate generator
2. **Basic Generator Agent**: Fast, simple RAG for straightforward queries
3. **Advanced Generator Agent**: Uses advanced techniques (Query Decomposition, HyDE, Multi-Query) for complex queries

## Output Modes

When running `agentic_rag.py`, you can choose from three output modes:

- **silent**: Shows only the final answer
- **verbose**: Shows routing decision and which agent was used (default)
- **debug**: Shows all steps, evaluations, and detailed reasoning

## How It Works

1. User submits a query
2. Router sends query to Basic Generator
3. Router evaluates the answer quality
4. If insufficient, Router activates Advanced Generator with:
   - Query Decomposition (breaks complex queries into sub-queries)
   - HyDE (generates hypothetical answer, then searches)
   - Multi-Query (generates query variations)
5. Final answer is returned with metadata

## Advanced Techniques

### Query Decomposition
Automatically breaks complex queries into simpler sub-queries, retrieves for each, then synthesizes a comprehensive answer.

### HyDE (Hypothetical Document Embeddings)
Generates a hypothetical ideal answer first, then uses it to find similar real documents for a more focused retrieval.

### Multi-Query Retrieval
Creates multiple phrasings of the query to capture different perspectives and improve retrieval coverage.

## Configuration

Edit `config.py` to customize:
- Agent parameters (temperature, max_tokens)
- Retrieval settings (number of chunks)
- Evaluation thresholds
- Chunking parameters

## Troubleshooting

### API Key Error
- Ensure `.env` file exists and contains `GEMINI_API_KEY=your_key`
- Verify the API key is valid at https://aistudio.google.com/app/apikey

### No Documents Found
- Ensure `docs/` folder contains `.txt` files
- The system will automatically process all `.txt` files in the `docs/` folder

### Vector Store Issues
- Delete the ChromaDB collection by restarting and it will rebuild automatically
- Check that documents are properly formatted text files

