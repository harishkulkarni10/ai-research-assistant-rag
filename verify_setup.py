# Setup verification script
# Run this to check if your RAG system is ready to use

import sys
from pathlib import Path

print("=" * 60)
print("RAG System Setup Verification")
print("=" * 60)

errors = []
warnings = []

# Check 1: Python version
print("\n[1] Checking Python version...")
if sys.version_info < (3, 10):
    errors.append(f"Python 3.10+ required. Found: {sys.version}")
    print("  ❌ FAILED")
else:
    print(f"  ✅ Python {sys.version_info.major}.{sys.version_info.minor}")

# Check 2: Dependencies
print("\n[2] Checking dependencies...")
try:
    import torch
    import transformers
    import chromadb
    import streamlit
    import fastapi
    import requests
    print("  ✅ Core dependencies installed")
except ImportError as e:
    errors.append(f"Missing dependency: {e}")
    print(f"  ❌ FAILED: {e}")

# Check 3: Configuration
print("\n[3] Checking configuration...")
try:
    from rag_arxiv_qa.src.utils.config import load_config
    config = load_config()
    print("  ✅ Configuration loaded")
    
    provider = config.get("generation", {}).get("provider", "unknown")
    model = config.get("generation", {}).get("model", "unknown")
    print(f"  → LLM Provider: {provider}")
    print(f"  → Model: {model}")
except Exception as e:
    errors.append(f"Configuration error: {e}")
    print(f"  ❌ FAILED: {e}")

# Check 4: Vector Database
print("\n[4] Checking vector database...")
try:
    from rag_arxiv_qa.src.indexing.chroma_store import ChromaVectorStore
    store = ChromaVectorStore(config)
    count = store.count()
    
    if count == 0:
        warnings.append("Vector database is empty. You need to run ingestion pipeline.")
        print(f"  ⚠️  WARNING: Vector DB exists but is empty ({count} chunks)")
        print("     → Run ingestion pipeline to populate it")
    else:
        print(f"  ✅ Vector DB has {count} chunks")
except Exception as e:
    errors.append(f"Vector DB error: {e}")
    print(f"  ❌ FAILED: {e}")

# Check 5: LLM Provider
print("\n[5] Checking LLM provider...")
try:
    from rag_arxiv_qa.src.generation.llm_providers import get_llm_provider
    provider_instance = get_llm_provider(config)
    print(f"  ✅ LLM provider '{provider}' initialized")
    
    # Test connection for API-based providers
    if provider == "ollama":
        import requests
        try:
            response = requests.get(f"{config['generation'].get('base_url', 'http://localhost:11434')}/api/tags", timeout=5)
            if response.status_code == 200:
                print("  ✅ Ollama server is running")
            else:
                warnings.append("Ollama server may not be running")
                print("  ⚠️  WARNING: Cannot connect to Ollama server")
        except:
            warnings.append("Ollama server not accessible. Make sure it's running.")
            print("  ⚠️  WARNING: Cannot connect to Ollama. Run: ollama serve")
    
    elif provider == "huggingface":
        import os
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            warnings.append("HUGGINGFACE_API_KEY not set. API calls may be rate-limited.")
            print("  ⚠️  WARNING: HUGGINGFACE_API_KEY not set")
        else:
            print("  ✅ HuggingFace API key found")
    
    elif provider in ["vllm", "tgi"]:
        base_url = config["generation"].get("base_url", "http://localhost:8000" if provider == "vllm" else "http://localhost:8080")
        import requests
        try:
            if provider == "vllm":
                response = requests.get(f"{base_url}/health", timeout=5)
            else:
                response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"  ✅ {provider.upper()} server is running")
            else:
                warnings.append(f"{provider.upper()} server may not be running")
                print(f"  ⚠️  WARNING: Cannot connect to {provider.upper()} server at {base_url}")
        except:
            warnings.append(f"{provider.upper()} server not accessible. Make sure it's running.")
            print(f"  ⚠️  WARNING: Cannot connect to {provider.upper()}. Start the server first.")
    
except Exception as e:
    errors.append(f"LLM provider error: {e}")
    print(f"  ❌ FAILED: {e}")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

if errors:
    print(f"\n❌ Found {len(errors)} error(s):")
    for error in errors:
        print(f"   - {error}")
    print("\n⚠️  Please fix these errors before running the application.")
    sys.exit(1)

if warnings:
    print(f"\n⚠️  Found {len(warnings)} warning(s):")
    for warning in warnings:
        print(f"   - {warning}")
    print("\n⚠️  The application may not work correctly with these warnings.")
else:
    print("\n✅ All checks passed! Your system is ready to use.")
    print("\nNext steps:")
    print("  1. Run: streamlit run streamlit_app.py")
    print("  2. Or: python -m uvicorn rag_arxiv_qa.src.api.main:app --reload")

print("=" * 60)
