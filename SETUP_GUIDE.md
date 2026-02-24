# Complete Setup Guide

This guide will help you get the RAG system running from scratch.

## Prerequisites Check

Before starting, ensure you have:
- Python 3.10+
- Git
- (Optional) Docker for containerized deployment
- (Optional) GPU with CUDA for vLLM/TGI

## Step 1: Clone and Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd "AI research assistant"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Choose Your LLM Provider

You have 5 options. Choose ONE based on your setup:

### Option A: Ollama (Easiest - CPU/GPU, Local)

**Best for**: Quick local testing, no API keys needed

```bash
# 1. Install Ollama from https://ollama.ai
# 2. Pull a model
ollama pull llama3.2:1b

# 3. Verify it's running
ollama list
```

**Config**: Already set in `config/config.yaml` (provider: "ollama")

### Option B: HuggingFace API (Easiest - Cloud)

**Best for**: Production, no local GPU needed

```bash
# 1. Get free API key from https://huggingface.co/settings/tokens
# 2. Set environment variable
export HUGGINGFACE_API_KEY=your_key_here  # Windows: set HUGGINGFACE_API_KEY=your_key_here

# Or add to .env file:
echo "HUGGINGFACE_API_KEY=your_key_here" >> .env
```

**Config**: Update `config/config.yaml`:
```yaml
generation:
  provider: "huggingface"
  model: "mistralai/Mistral-7B-Instruct-v0.2"
```

### Option C: vLLM (Fastest - GPU Required)

**Best for**: GPU users who want maximum speed

```bash
# 1. Ensure you have CUDA and vLLM installed
# (vLLM is in requirements.txt but requires CUDA)

# 2. Start vLLM server in a separate terminal
python -m rag_arxiv_qa.src.inference.vllm.serve \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --port 8000

# Or use vLLM directly:
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --port 8000
```

**Config**: Update `config/config.yaml`:
```yaml
generation:
  provider: "vllm"
  model: "mistralai/Mistral-7B-Instruct-v0.2"
  base_url: "http://localhost:8000"
```

### Option D: TGI (Production Serving - GPU Required)

**Best for**: Production GPU serving with advanced features

```bash
# 1. Using Docker (recommended)
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id mistralai/Mistral-7B-Instruct-v0.2

# Or use the helper script:
python -m rag_arxiv_qa.src.inference.tgi.serve \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --use-docker
```

**Config**: Update `config/config.yaml`:
```yaml
generation:
  provider: "tgi"
  model: "mistralai/Mistral-7B-Instruct-v0.2"
  base_url: "http://localhost:8080"
```

### Option E: Transformers (Fallback - CPU/GPU)

**Best for**: When other options don't work

**Config**: Update `config/config.yaml`:
```yaml
generation:
  provider: "transformers"
  model: "Qwen/Qwen2-1.5B-Instruct"
```

## Step 3: Populate Vector Database

**IMPORTANT**: The system needs a populated vector database to work!

### Option 1: Use Existing Data (If Available)

If you have existing processed data:
```bash
# Check if vector DB exists and has data
python -c "
from rag_arxiv_qa.src.utils.config import load_config
from rag_arxiv_qa.src.indexing.chroma_store import ChromaVectorStore
config = load_config()
store = ChromaVectorStore(config)
print(f'Vector DB has {store.count()} chunks')
"
```

### Option 2: Run Ingestion Pipeline

If vector DB is empty or doesn't exist:

```bash
# Run your ingestion notebook or script
# See notebooks/01_ingestion/ for ingestion pipeline
# This will:
# 1. Download ArXiv papers
# 2. Chunk them
# 3. Create embeddings
# 4. Store in ChromaDB
```

**Quick test**: Create a minimal test vector DB:
```python
# test_setup.py
from rag_arxiv_qa.src.utils.config import load_config
from rag_arxiv_qa.src.indexing.chroma_store import ChromaVectorStore
from rag_arxiv_qa.src.embeddings.embedder import Embedder

config = load_config()
embedder = Embedder(config)
store = ChromaVectorStore(config)

# Add a test document
test_text = "Transformers are neural network architectures that use attention mechanisms."
embedding = embedder.embed_query(test_text)

store.collection.add(
    ids=["test_1"],
    embeddings=[embedding],
    documents=[test_text],
    metadatas=[{"source": "test", "position": "1"}]
)

print(f"Test vector DB created with {store.count()} chunks")
```

## Step 4: Verify Setup

```bash
# Test the RAG service
python test_rag_service.py
```

Expected output: A response with answer, citations, and metadata.

## Step 5: Run the Application

### Streamlit UI (Recommended)

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### FastAPI Server

```bash
python -m uvicorn rag_arxiv_qa.src.api.main:app --reload
```

Open `http://localhost:8000/docs` for API documentation.

## Troubleshooting

### Issue: "Vector database not found" or "No chunks retrieved"

**Solution**: Run Step 3 to populate the vector database.

### Issue: "Cannot connect to Ollama"

**Solution**: 
```bash
# Start Ollama
ollama serve

# Verify model is available
ollama list
```

### Issue: "Cannot connect to vLLM/TGI server"

**Solution**: Make sure the inference server is running in a separate terminal before starting the RAG app.

### Issue: "HuggingFace API error"

**Solution**: 
- Check your API key is set: `echo $HUGGINGFACE_API_KEY`
- Verify the model name is correct
- Check your internet connection

### Issue: "Out of memory" with Transformers provider

**Solution**: 
- Use a smaller model
- Switch to Ollama or HuggingFace API
- Use vLLM/TGI if you have GPU

## Quick Start Summary

For fastest setup:

1. **Install Ollama** and pull a model
2. **Run ingestion** to populate vector DB
3. **Run Streamlit**: `streamlit run streamlit_app.py`
4. **Done!**

For production:

1. **Get HuggingFace API key**
2. **Set environment variable**
3. **Update config** to use HuggingFace provider
4. **Deploy** (see DEPLOYMENT.md)

## Next Steps

- See `README.md` for full documentation
- See `DEPLOYMENT.md` for production deployment
- See `QUICKSTART.md` for quick reference
