# Quick Start Guide

Get your RAG system running in 5 minutes!

## Fast Setup (Local with Ollama)

### 1. Install Ollama
```bash
# Visit https://ollama.ai and install
# Or use:
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull a Model
```bash
ollama pull llama3.2:1b
# This downloads a small, fast model (~1.3GB)
```

### 3. Setup Python Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run Streamlit UI
```bash
streamlit run streamlit_app.py
```

**That's it!** Open `http://localhost:8501` and start asking questions.

## What Changed?

### What We Built

1. **Multi-Provider LLM Support**
   - Ollama (local, free) - Recommended for development
   - HuggingFace API (cloud, free tier) - Recommended for production
   - Transformers (local fallback)

2. **Streamlit UI**
   - Chatbot interface
   - Citation display
   - Performance metrics

3. **FastAPI Backend**
   - REST API with `/api/v1/query`
   - Health checks
   - Metrics endpoint

4. **Monitoring & Logging**
   - Structured JSON logs
   - Performance metrics
   - Error tracking

5. **Docker Support**
   - Dockerfile
   - docker-compose.yml

6. **CI/CD**
   - GitHub Actions pipeline

7. **Documentation**
   - Comprehensive README
   - Deployment guide
   - Quick start guide

### What Stayed the Same

- **Ingestion pipeline** - No changes needed
- **Chunking** - No changes needed
- **Embeddings** - No changes needed
- **Vector database** - No changes needed
- **Retrieval** - No changes needed

**All your existing data and pipelines work as-is!**

## Next Steps

### For Local Development:
1. Use Ollama (already set up above)
2. Test with Streamlit UI
3. Populate vector DB if not done already

### For Production Deployment:
1. Get HuggingFace API key (free): https://huggingface.co/settings/tokens
2. Set `HUGGINGFACE_API_KEY` environment variable
3. Update `config/prod.yaml` (already done)
4. Deploy to HuggingFace Spaces (see DEPLOYMENT.md)

## Common Questions

**Q: Do I need to change my ingestion/chunking/embeddings?**  
A: No! Everything stays the same. Only the LLM generation layer changed.

**Q: Will this work for other people cloning my repo?**  
A: Yes! They just need to:
1. Install Ollama
2. Pull a model: `ollama pull llama3.2:1b`
3. Run `streamlit run streamlit_app.py`

**Q: Can I deploy this for free?**  
A: Yes! HuggingFace Spaces is free and perfect for this. See DEPLOYMENT.md.

**Q: What about the infrastructure issues?**  
A: Solved! Using Ollama (local) or HuggingFace API (cloud) eliminates the need to download/run large models locally.

## Troubleshooting

**Issue: "Cannot connect to Ollama"**
- Solution: Make sure Ollama is running: `ollama serve` (usually auto-starts)

**Issue: "Model not found"**
- Solution: Pull the model: `ollama pull llama3.2:1b`

**Issue: "Vector database not found"**
- Solution: Run your ingestion pipeline first to populate the vector DB

**Issue: Slow responses**
- Solution: Use a smaller model or switch to HuggingFace API

## More Information

- Full README: See `README.md`
- Deployment: See `DEPLOYMENT.md`
- API Docs: Run FastAPI and visit `/docs`

---

**Ready to deploy?** Check out `DEPLOYMENT.md` for free hosting options!
