# 🧠 Local RAG System with Ollama & ChromaDB

A complete **Retrieval-Augmented Generation (RAG)** system that runs entirely offline using Ollama, ChromaDB, and Python. This project demonstrates how to build a privacy-focused AI knowledge base without relying on cloud services or external APIs.

## 🌟 Features

- **🔒 Completely Offline**: No data leaves your machine
- **💰 Zero API Costs**: Uses local LLM (Llama3.1) via Ollama
- **⚡ Fast Retrieval**: Efficient semantic search with ChromaDB
- **🎯 Customizable**: Easy to add your own documents
- **🔧 Simple Setup**: Minimal dependencies and configuration

## 🏗️ Architecture

```
Documents → Preprocessing → Embeddings → ChromaDB Storage
                                              ↓
User Query → Query Embedding → Vector Search → Context Retrieval
                                              ↓
Context + Query → Ollama LLM → Generated Response
```

## 📋 Prerequisites

- **Python 3.8+**
- **Ollama** installed with llama3.1 model
- **8GB+ RAM** (for model loading)

### Install Ollama

```bash
# On Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the llama3.1 model
ollama pull llama3.1
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ayyzenn/local-rag-ollama.git
cd local-rag-ollama
```

### 2. Set Up Virtual Environment

```bash
python -m venv my_env
source my_env/bin/activate  # On Windows: my_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the RAG System

```bash
python rag_local_ollama.py
```

## 📁 Project Structure

```
rag/
├── docs/                    # Knowledge base documents
│   ├── about_me.txt        # Personal information
│   ├── education.txt       # AI in education
│   ├── finance.txt         # AI in finance
│   └── healthcare.txt      # AI in healthcare
├── preprocess.py           # Document preprocessing
├── rag_local_ollama.py     # Main RAG system
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Usage

### Running the System

```bash
python rag_local_ollama.py
```

### Example Queries

**Personal Information:**
```
Ask your question: What is Saad Ahmad's background?
```

**Domain Knowledge:**
```
Ask your question: How is AI being used in healthcare?
```

**Technical Questions:**
```
Ask your question: What technologies are used in this RAG system?
```

## 📊 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `chromadb` | >=0.4.0 | Vector database |
| `sentence-transformers` | >=2.0.0 | Text embeddings |
| `numpy` | >=1.21.0 | Numerical computations |
| `torch` | >=1.9.0 | PyTorch backend |

## 🛠️ Customization

### Adding Your Own Documents

1. Place your `.txt` files in the `docs/` directory
2. Run the system - it will automatically process new documents

### Modifying Chunk Size

Edit `preprocess.py`:
```python
def chunk_text(text, max_words=100):  # Change this value
```

### Using Different Embedding Models

Edit `rag_local_ollama.py`:
```python
model = SentenceTransformer("all-MiniLM-L6-v2")  # Replace with your model
```

## 🔍 How It Works

1. **Document Processing**: Text files are cleaned and split into chunks
2. **Embedding Generation**: Each chunk is converted to a vector using sentence-transformers
3. **Vector Storage**: Embeddings are stored in ChromaDB for fast retrieval
4. **Query Processing**: User questions are embedded and matched against stored vectors
5. **Context Retrieval**: Most relevant document chunks are retrieved
6. **Response Generation**: Ollama generates answers using retrieved context

## 🎯 Performance

- **Response Time**: ~2-5 seconds per query
- **Memory Usage**: ~4-6GB with llama3.1
- **Storage**: ~500MB for embeddings and documents
- **Accuracy**: High relevance for domain-specific queries

## 🚧 Future Enhancements

- [ ] Web interface with Gradio/Streamlit
- [ ] Support for PDF and Word documents
- [ ] Advanced chunking strategies
- [ ] Query expansion and reformulation
- [ ] Integration with LangChain
- [ ] Multi-modal support (images, tables)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Ollama** for local LLM deployment
- **ChromaDB** for vector database
- **sentence-transformers** for embeddings
- **Hugging Face** for model hosting

*Built with ❤️ for privacy-focused AI applications* 