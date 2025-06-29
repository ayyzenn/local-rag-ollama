import os
import re

def clean_text(text):
    # Remove unwanted characters and whitespace
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text, max_words=100):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def preprocess_documents(doc_folder):
    all_chunks = []
    for filename in os.listdir(doc_folder):
        if filename.endswith(".txt"):
            path = os.path.join(doc_folder, filename)
            with open(path, 'r') as f:
                raw = f.read()
                cleaned = clean_text(raw)
                chunks = chunk_text(cleaned)
                all_chunks.extend(chunks)
    return all_chunks

