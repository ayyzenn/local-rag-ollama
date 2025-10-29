import os
import re
from typing import List, Dict, Any
from config import CHUNK_CONFIG


def clean_text(text):
    """Remove unwanted characters and whitespace"""
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def chunk_text(text, max_words=100, overlap_words=20):
    """
    Chunk text with overlapping windows for better context preservation
    
    Args:
        text: Input text to chunk
        max_words: Maximum words per chunk
        overlap_words: Number of words to overlap between chunks
    """
    words = text.split()
    if len(words) <= max_words:
        return [' '.join(words)]
    
    chunks = []
    start = 0
    step = max_words - overlap_words  # Move forward by (max - overlap) words
    
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = ' '.join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += step
    
    return chunks


def preprocess_documents(doc_folder: str) -> List[Dict[str, Any]]:
    """
    Preprocess documents with metadata extraction
    
    Returns:
        List of dictionaries with 'text' and 'metadata' keys
    """
    all_chunks = []
    max_words = CHUNK_CONFIG["max_words"]
    overlap_words = CHUNK_CONFIG["overlap_words"]
    
    for filename in os.listdir(doc_folder):
        if filename.endswith(".txt"):
            path = os.path.join(doc_folder, filename)
            with open(path, 'r', encoding='utf-8') as f:
                raw = f.read()
                cleaned = clean_text(raw)
                chunks = chunk_text(cleaned, max_words=max_words, overlap_words=overlap_words)
                
                # Add metadata for each chunk
                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "text": chunk,
                        "metadata": {
                            "source": filename,
                            "source_path": path,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "document_type": _infer_document_type(filename)
                        }
                    })
    
    return all_chunks


def _infer_document_type(filename: str) -> str:
    """Infer document type from filename"""
    filename_lower = filename.lower()
    if "about" in filename_lower or "me" in filename_lower:
        return "personal_info"
    elif "education" in filename_lower:
        return "education"
    elif "finance" in filename_lower:
        return "finance"
    elif "health" in filename_lower or "care" in filename_lower:
        return "healthcare"
    else:
        return "general"


def preprocess_documents_simple(doc_folder: str) -> List[str]:
    """Legacy function: Returns simple list of chunks (for backward compatibility)"""
    chunks_with_metadata = preprocess_documents(doc_folder)
    return [chunk["text"] for chunk in chunks_with_metadata]

