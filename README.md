# ArXiv Research Assistant - RAG System

A Retrieval-Augmented Generation (RAG) system for querying ArXiv research papers. Built with modular architecture supporting multiple LLM providers, monitoring, and easy deployment.

## Features

- **Multiple LLM Providers**: Support for Ollama (local), HuggingFace Inference API (cloud), vLLM (GPU fast), TGI (GPU serving), and Transformers (local fallback)
- **Semantic Search**: Dense vector retrieval using BGE embeddings
- **Reranking**: Context-aware reranking for improved relevance
- **Streamlit UI**: Interactive chatbot interface
- **REST API**: FastAPI-based API with OpenAPI documentation
- **Monitoring & Metrics**: Structured logging and performance metrics
- **Docker Support**: Containerized deployment
- **CI/CD**: GitHub Actions pipeline

## Prerequisites

- Python 3.10+
- Docker & Docker Compose (optional, for containerized deployment)
- Ollama (for local LLM inference) - [Install Ollama](https://ollama.ai)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd "AI research assistant"
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env`:
```env
APP_ENV=dev
HUGGINGFACE_API_KEY=your_key_here  # Optional, for HuggingFace provider
```

### 5. Setup Ollama (for Local Inference)

If using Ollama provider (recommended for local development):

```bash
# Install Ollama from https://ollama.ai
# Then pull a model:
ollama pull llama3.2:1b
# Or use other models: mistral, qwen2.5:1.5b, etc.
```

### 6. Populate Vector Database

**IMPORTANT**: The system requires a populated vector database to work!

Before using the system, you need to ingest and index ArXiv papers:

```bash
# Run ingestion pipeline (see notebooks/01_ingestion/)
# This will download papers, chunk them, create embeddings, and store in ChromaDB

# Or use the test setup script to create a minimal test DB:
python -c "
from rag_arxiv_qa.src.utils.config import load_config
from rag_arxiv_qa.src.indexing.chroma_store import ChromaVectorStore
from rag_arxiv_qa.src.embeddings.embedder import Embedder

config = load_config()
embedder = Embedder(config)
store = ChromaVectorStore(config)

test_text = 'Transformers are neural network architectures that use attention mechanisms.'
embedding = embedder.embed_query(test_text)

store.collection.add(
    ids=['test_1'],
    embeddings=[embedding],
    documents=[test_text],
    metadatas=[{'source': 'test', 'position': '1'}]
)

print(f'Test vector DB created with {store.count()} chunks')
"
```

### 7. Verify Setup

Run the verification script to check if everything is configured correctly:

```bash
python verify_setup.py
```

This will check:
- Python version
- Dependencies
- Configuration
- Vector database
- LLM provider connectivity

**See SETUP_GUIDE.md for detailed instructions.**

## Usage

### Option 1: Streamlit UI (Recommended for Interactive Use)

```bash
streamlit run streamlit_app.py
```

Open your browser to `http://localhost:8501`

### Option 2: FastAPI Server

```bash
python -m uvicorn rag_arxiv_qa.src.api.main:app --reload
```

API will be available at:
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`

### Option 3: Python Script

```bash
python test_rag_service.py
```

### Option 4: Docker Compose

```bash
docker-compose up -d
```

Access:
- API: `http://localhost:8000`
- Streamlit: `http://localhost:8501`

## Configuration

The system uses YAML configuration files in the `config/` directory:

- `config.yaml`: Base configuration
- `dev.yaml`: Development overrides
- `prod.yaml`: Production overrides

### LLM Provider Configuration

Edit `config/config.yaml` to change the LLM provider:

**Ollama (Local - Recommended for Development):**
```yaml
generation:
  provider: "ollama"
  model: "llama3.2:1b"
  base_url: "http://localhost:11434"
```

**HuggingFace API (Cloud - Good for Production):**
```yaml
generation:
  provider: "huggingface"
  model: "mistralai/Mistral-7B-Instruct-v0.2"
  # API key from environment variable: HUGGINGFACE_API_KEY
```

**vLLM (GPU - Fast Inference):**
```yaml
generation:
  provider: "vllm"
  model: "mistralai/Mistral-7B-Instruct-v0.2"
  base_url: "http://localhost:8000"
```
*Requires vLLM server running. Start with: `python -m vllm.entrypoints.openai.api_server --model <model> --port 8000`*

**TGI (GPU - Production Serving):**
```yaml
generation:
  provider: "tgi"
  model: "mistralai/Mistral-7B-Instruct-v0.2"
  base_url: "http://localhost:8080"
```
*Requires TGI server running. See `rag_arxiv_qa/src/inference/tgi/serve.py`*

**Transformers (Local Fallback):**
```yaml
generation:
  provider: "transformers"
  model: "Qwen/Qwen2-1.5B-Instruct"
```

## Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAG Service    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────┐
│Retrieval│ │Generation│
│Pipeline │ │ (LLM)    │
└────┬────┘ └────┬─────┘
     │           │
     ▼           ▼
┌─────────┐ ┌──────────┐
│Embedding│ │  Ollama  │
│+ Vector │ │ Hugging  │
│  Search │ │  Face    │
└─────────┘ └──────────┘
```

## Project Structure

```
.
├── config/                 # Configuration files
│   ├── config.yaml        # Base config
│   ├── dev.yaml           # Dev overrides
│   └── prod.yaml          # Prod overrides
├── data/                   # Data storage
│   ├── raw/               # Raw ArXiv papers
│   ├── processed/         # Processed data
│   └── index/             # Vector database
├── rag_arxiv_qa/          # Main package
│   └── src/
│       ├── api/           # FastAPI application
│       ├── chunking/      # Text chunking
│       ├── embeddings/    # Embedding models
│       ├── generation/   # LLM generation
│       ├── indexing/      # Vector store
│       ├── ingestion/    # Data ingestion
│       ├── retrieval/     # Retrieval pipeline
│       ├── services/     # RAG service
│       └── utils/        # Utilities (config, logger, metrics)
├── notebooks/             # Jupyter notebooks for data processing
├── tests/                 # Unit tests
├── streamlit_app.py      # Streamlit UI
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
└── requirements.txt      # Python dependencies
```

## API Endpoints

### Query Endpoint

```bash
POST /api/v1/query
Content-Type: application/json

{
  "query": "What are transformers in language models?"
}
```

Response:
```json
{
  "answer": "...",
  "citations": [1, 2],
  "confidence_score": 0.85,
  "metadata": {
    "request_id": "...",
    "retrieved_chunks": 10,
    "retrieval_time_sec": 0.123,
    "generation_time_sec": 1.456,
    "total_time_sec": 1.579
  }
}
```

### Health Check

```bash
GET /health
```

### Metrics

```bash
GET /metrics
```

## Deployment

### Free Deployment Options

#### Option 1: HuggingFace Spaces (Recommended)

1. Create a HuggingFace Space
2. Push your code
3. Configure environment variables
4. Deploy automatically

**Pros**: Free, automatic deployments, built-in GPU support

#### Option 2: Railway

1. Connect GitHub repository
2. Railway auto-detects Docker
3. Set environment variables
4. Deploy

**Pros**: Free tier available, easy setup

#### Option 3: Render

1. Create new Web Service
2. Connect repository
3. Use Dockerfile
4. Deploy

**Pros**: Free tier, simple deployment

### Production Deployment Checklist

- [ ] Set `APP_ENV=prod` in environment
- [ ] Configure HuggingFace API key (if using HuggingFace provider)
- [ ] Ensure vector database is populated
- [ ] Set up monitoring and alerting
- [ ] Configure CORS appropriately
- [ ] Set up log aggregation
- [ ] Enable HTTPS
- [ ] Set resource limits

## Monitoring

The system includes built-in metrics:

- Request counts and success/error rates
- Latency metrics (retrieval, generation, total)
- Retrieved chunks count
- Confidence scores

Access metrics via:
- API: `GET /metrics`
- Logs: Structured JSON logs in `logs/app.log`

## Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=rag_arxiv_qa
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License


## Acknowledgments

- ArXiv for research papers
- HuggingFace for models and infrastructure
- Ollama for local LLM inference
- ChromaDB for vector storage

## Contact


---

**Note**: For local development, Ollama is recommended as it's free, open-source, and doesn't require API keys. For production deployment, HuggingFace Inference API offers a free tier and is easy to use.
