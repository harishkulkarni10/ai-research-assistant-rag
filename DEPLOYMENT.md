# Deployment Guide

This guide covers deploying the ArXiv Research Assistant RAG system to various platforms.

## Deployment Strategy

### Local Development
- **LLM Provider**: Ollama (free, local)
- **Vector DB**: ChromaDB (local filesystem)
- **UI**: Streamlit (localhost)

### Production Deployment
- **LLM Provider**: HuggingFace Inference API (free tier available)
- **Vector DB**: ChromaDB (persisted in container volume)
- **UI**: Streamlit or FastAPI (deployed service)

## Free Deployment Options

### 1. HuggingFace Spaces (Recommended)

**Why**: Free, automatic deployments, built-in GPU support, easy sharing

**Steps**:

1. **Create a HuggingFace Space**:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Docker" as SDK
   - Name it (e.g., `arxiv-rag-assistant`)

2. **Prepare for Deployment**:
   ```bash
   # Create app.py for HuggingFace Spaces
   # This will be the entry point
   ```

3. **Create `app.py`** (for HuggingFace Spaces):
   ```python
   import streamlit as st
   import sys
   from pathlib import Path
   
   sys.path.insert(0, str(Path(__file__).parent))
   
   # Import and run your Streamlit app
   exec(open("streamlit_app.py").read())
   ```

4. **Create `README.md` for Space**:
   ```markdown
   ---
   title: ArXiv Research Assistant
   colorFrom: blue
   colorTo: purple
   sdk: docker
   pinned: false
   ---
   ```

5. **Set Environment Variables**:
   - In Space settings, add:
     - `HUGGINGFACE_API_KEY`: Your API key
     - `APP_ENV`: `prod`

6. **Push to HuggingFace**:
   ```bash
   git clone https://huggingface.co/spaces/your-username/arxiv-rag-assistant
   cd arxiv-rag-assistant
   # Copy your files
   git add .
   git commit -m "Initial deployment"
   git push
   ```

**Limitations**: 
- Free tier: CPU only (can upgrade to GPU)
- 16GB storage
- Public by default

### 2. Railway

**Why**: Free tier, easy GitHub integration, automatic deployments

**Steps**:

1. **Sign up**: https://railway.app
2. **Create New Project**: "Deploy from GitHub repo"
3. **Select Repository**: Your RAG project
4. **Configure**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port $PORT`
5. **Set Environment Variables**:
   - `HUGGINGFACE_API_KEY`
   - `APP_ENV=prod`
   - `PORT=8501` (Railway sets this automatically)
6. **Deploy**: Railway auto-detects and deploys

**Limitations**:
- Free tier: $5 credit/month
- Sleeps after inactivity (can upgrade)

### 3. Render

**Why**: Free tier, simple setup, good for demos

**Steps**:

1. **Sign up**: https://render.com
2. **New Web Service**: Connect GitHub repo
3. **Configure**:
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
4. **Set Environment Variables**:
   - `HUGGINGFACE_API_KEY`
   - `APP_ENV=prod`
5. **Deploy**: Click "Create Web Service"

**Limitations**:
- Free tier: Sleeps after 15 min inactivity
- Limited resources

### 4. Fly.io

**Why**: Free tier, global deployment, good performance

**Steps**:

1. **Install Fly CLI**: `curl -L https://fly.io/install.sh | sh`
2. **Login**: `fly auth login`
3. **Create App**: `fly launch`
4. **Configure `fly.toml`**:
   ```toml
   app = "your-app-name"
   primary_region = "iad"
   
   [build]
   
   [http_service]
     internal_port = 8501
     force_https = true
     auto_stop_machines = true
     auto_start_machines = true
     min_machines_running = 0
   
   [[services]]
     protocol = "tcp"
     internal_port = 8501
   ```
5. **Set Secrets**:
   ```bash
   fly secrets set HUGGINGFACE_API_KEY=your_key
   fly secrets set APP_ENV=prod
   ```
6. **Deploy**: `fly deploy`

**Limitations**:
- Free tier: 3 shared VMs
- Limited storage

## Production Configuration

### Environment Variables

Create a `.env` file or set in deployment platform:

```env
APP_ENV=prod
HUGGINGFACE_API_KEY=your_key_here
LOG_LEVEL=INFO
```

### Update Config for Production

Ensure `config/prod.yaml` is configured:

```yaml
generation:
  provider: "huggingface"
  model: "mistralai/Mistral-7B-Instruct-v0.2"
```

### Pre-populate Vector Database

Before deployment, ensure your vector database is populated:

1. Run ingestion pipeline locally
2. Commit `data/index/` to repository (if small) OR
3. Upload to cloud storage and download on first run

## Monitoring in Production

### Logs

All platforms provide log viewing:
- **HuggingFace Spaces**: Built-in logs tab
- **Railway**: Logs in dashboard
- **Render**: Logs in dashboard
- **Fly.io**: `fly logs`

### Metrics

Access metrics endpoint:
```bash
curl https://your-app-url/metrics
```

### Health Checks

```bash
curl https://your-app-url/health
```

## Docker Deployment

For any platform supporting Docker:

```bash
# Build
docker build -t rag-assistant .

# Run
docker run -p 8501:8501 \
  -e HUGGINGFACE_API_KEY=your_key \
  -e APP_ENV=prod \
  rag-assistant
```

## Security Considerations

1. **API Keys**: Never commit to repository
2. **CORS**: Configure appropriately for production
3. **Rate Limiting**: Consider adding rate limits
4. **Input Validation**: Already implemented in FastAPI
5. **HTTPS**: Use HTTPS in production (most platforms provide)

## Cost Estimation

### Free Tier Options:
- **HuggingFace Spaces**: Free (CPU), $0.60/hour (GPU)
- **Railway**: $5 credit/month
- **Render**: Free (with limitations)
- **Fly.io**: Free tier available

### HuggingFace API:
- **Free tier**: 1000 requests/month
- **Pay-as-you-go**: Very affordable

## Recommended Setup for Resume/Portfolio

**Best Option**: HuggingFace Spaces

**Why**:
- Free
- Easy to share (just a link)
- Professional appearance
- Automatic deployments
- Built-in monitoring

**Setup**:
1. Deploy to HuggingFace Spaces
2. Get public URL: `https://your-username-arxiv-rag-assistant.hf.space`
3. Add to resume/GitHub README
4. Share with recruiters

## Deployment Checklist

- [ ] Code is tested and working locally
- [ ] Vector database is populated
- [ ] Environment variables are set
- [ ] Production config is correct
- [ ] Health check endpoint works
- [ ] Metrics endpoint accessible
- [ ] Logs are being captured
- [ ] CORS is configured
- [ ] HTTPS is enabled
- [ ] Documentation is updated with deployment URL

## Troubleshooting

### Issue: Model not loading
- **Solution**: Check HuggingFace API key and model name

### Issue: Vector database not found
- **Solution**: Ensure `data/index/` is included or populated on first run

### Issue: Slow responses
- **Solution**: Use smaller models or upgrade to GPU tier

### Issue: Out of memory
- **Solution**: Reduce batch sizes, use smaller models

---

**Need Help?** Open an issue on GitHub!
