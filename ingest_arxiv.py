# ArXiv Ingestion Script
# Downloads and ingests ArXiv papers into the vector database

from datasets import load_dataset
import pandas as pd
from rag_arxiv_qa.src.utils.config import load_config
from rag_arxiv_qa.src.ingestion.ingestion import IngestionPipeline
from tqdm import tqdm
import time

def filter_ai_ml_papers(example):
    """Filter papers that are AI/ML related."""
    ai_ml_keywords = [
        "machine learning", "deep learning", "neural network", "transformer",
        "large language model", "llm", "gpt", "bert", "artificial intelligence",
        "computer vision", "natural language processing", "nlp", "reinforcement learning",
        "generative ai", "diffusion model", "rag",   "retrieval augmented"
    ]
    text = (example.get('article', '') + ' ' + example.get('abstract', '')).lower()
    return any(keyword in text for keyword in ai_ml_keywords)

def prepare_documents(dataset, max_papers=None):
    """Convert dataset to document format for ingestion."""
    documents = []
    
    for idx, paper in enumerate(tqdm(dataset, desc="Preparing documents")):
        if max_papers and idx >= max_papers:
            break
            
        doc_id = paper.get('id', f"paper_{idx}")
        full_text = paper.get('article', paper.get('abstract', ''))
        abstract = paper.get('abstract', '')
        
        documents.append({
            "doc_id": doc_id,
            "text": full_text if full_text else abstract,
            "metadata": {
                "source": doc_id,
                "title": abstract[:200] + "..." if len(abstract) > 200 else abstract,
                "abstract": abstract[:500] if abstract else "",
            }
        })
    
    return documents

def load_from_parquet(parquet_path, max_papers=None):
    """Load documents from existing parquet file."""
    from pathlib import Path
    
    path = Path(parquet_path)
    if not path.exists():
        return None
    
    print(f"Loading from parquet file: {path}")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} papers from parquet file")
    
    documents = []
    for idx, row in df.head(max_papers if max_papers else len(df)).iterrows():
        doc_id = str(row.get('paper_id', idx))
        text = row.get('full_text', '')
        if not text or len(text.strip()) == 0:
            text = row.get('abstract', '')
        
        if not text or len(text.strip()) == 0:
            continue
        
        documents.append({
            "doc_id": doc_id,
            "text": text,
            "metadata": {
                "source": doc_id,
                "title": str(row.get('title', ''))[:200],
                "abstract": str(row.get('abstract', ''))[:500] if 'abstract' in row else "",
            }
        })
    
    return documents

def main():
    print("=" * 60)
    print("ArXiv Paper Ingestion")
    print("=" * 60)
    
    config = load_config()
    
    # Check if parquet file exists
    from pathlib import Path
    parquet_path = Path("rag_arxiv_qa/data/processed/arxiv_ai_ml_corpus.parquet")
    use_parquet = parquet_path.exists()
    
    if use_parquet:
        print(f"\nFound existing parquet file: {parquet_path}")
        print("Will use existing parquet file instead of downloading.")
    else:
        print("\nNo existing parquet file found.")
        print("Will download from HuggingFace dataset.")
    
    print("\nStorage & Time Estimates:")
    print("-" * 60)
    print("10 papers:    ~50-100 chunks,   ~50-100 MB,   ~2-5 minutes")
    print("100 papers:   ~500-1000 chunks, ~500 MB-1 GB, ~15-30 minutes")
    print("500 papers:   ~2.5K-5K chunks,   ~2-5 GB,      ~1-2 hours")
    print("1000 papers:  ~5K-10K chunks,   ~5-10 GB,     ~2-4 hours")
    print("5000 papers:  ~25K-50K chunks,  ~25-50 GB,    ~10-20 hours")
    print("-" * 60)
    
    print("\nChoose dataset size:")
    print("1. Small test: 10 papers")
    print("2. Medium: 100 papers")
    print("3. Large: 500 papers")
    print("4. Very Large: 1000 papers")
    print("5. All papers: All papers available")
    print("6. Custom: Enter your own number")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    size_map = {"1": 10, "2": 100, "3": 500, "4": 1000}
    
    if choice in size_map:
        max_papers = size_map[choice]
    elif choice == "5":
        max_papers = None  # All papers
    elif choice == "6":
        try:
            max_papers = int(input("Enter number of papers: "))
        except:
            print("Invalid input. Using default: 100 papers")
            max_papers = 100
    else:
        print("Invalid choice. Using default: 100 papers")
        max_papers = 100
    
    print(f"\nWill process {max_papers if max_papers else 'all'} papers")
    print("\nStarting ingestion...")
    
    start_time = time.time()
    
    # Load documents
    if use_parquet:
        print("\n[1/3] Loading from parquet file...")
        documents = load_from_parquet(parquet_path, max_papers)
        if documents is None:
            print("Error loading parquet file. Falling back to HuggingFace download.")
            use_parquet = False
        else:
            print(f"Prepared {len(documents)} documents from parquet file")
    
    if not use_parquet:
        # Step 1: Load dataset from HuggingFace
        print("\n[1/4] Loading ArXiv dataset from HuggingFace...")
        print("(This downloads ~2GB dataset on first run)")
        try:
            ds = load_dataset("ccdv/arxiv-summarization", split="train")
            print(f"Loaded {len(ds):,} papers from dataset")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Make sure you have 'datasets' installed: pip install datasets")
            return
        
        # Step 2: Filter AI/ML papers
        print("\n[2/4] Filtering AI/ML papers...")
        filtered_ds = ds.filter(filter_ai_ml_papers)
        print(f"Found {len(filtered_ds):,} AI/ML papers")
        
        # Step 3: Limit and prepare documents
        print(f"\n[3/4] Preparing {max_papers if max_papers else 'all'} documents...")
        if max_papers:
            limited_ds = filtered_ds.shuffle(seed=42).select(range(min(max_papers, len(filtered_ds))))
        else:
            limited_ds = filtered_ds.shuffle(seed=42)
        documents = prepare_documents(limited_ds, max_papers=max_papers)
        print(f"Prepared {len(documents)} documents")
    
    # Ingest into vector DB
    step_num = "[4/4]" if not use_parquet else "[2/3]"
    print(f"\n{step_num} Ingesting into vector database...")
    print("(This includes chunking, embedding, and storing - may take a while)")
    pipeline = IngestionPipeline(config)
    pipeline.ingest(documents)
    
    elapsed = time.time() - start_time
    
    # Summary
    from rag_arxiv_qa.src.indexing.chroma_store import ChromaVectorStore
    store = ChromaVectorStore(config)
    chunk_count = store.count()
    
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)
    print(f"Papers processed: {len(documents)}")
    print(f"Total chunks created: {chunk_count:,}")
    print(f"Time taken: {elapsed/60:.2f} minutes ({elapsed/3600:.2f} hours)")
    if len(documents) > 0:
        print(f"Average: {elapsed/len(documents):.2f} seconds per paper")
    
    # Estimate storage
    import os
    chroma_path = config["vector_db"]["persist_directory"]
    if os.path.exists(chroma_path):
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(chroma_path)
            for filename in filenames
        )
        print(f"Vector DB size: {total_size / (1024**3):.2f} GB")
    
    print("\nYour vector database is now ready!")
    print("You can now run: streamlit run streamlit_app.py")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nIngestion cancelled by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
