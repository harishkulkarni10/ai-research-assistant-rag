# Streamlit UI for RAG ArXiv Research Assistant
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag_arxiv_qa.src.utils.logger import setup_logging
from rag_arxiv_qa.src.retrieval.pipeline import RetrievalPipeline
from rag_arxiv_qa.src.generation.generator import Generator
from rag_arxiv_qa.src.services.rag_service import RAGService
from rag_arxiv_qa.src.utils.config import load_config

st.set_page_config(
    page_title="ArXiv Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
setup_logging(level="INFO")

# Initialize session state
if "rag_service" not in st.session_state:
    with st.spinner("Loading RAG system..."):
        try:
            config = load_config()
            retrieval_pipeline = RetrievalPipeline(config)
            generator = Generator(config)
            st.session_state.rag_service = RAGService(retrieval_pipeline, generator)
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {e}")
            st.error("Please ensure:")
            st.error("1. Vector database is populated (run ingestion pipeline)")
            st.error("2. Ollama is running (if using Ollama provider)")
            st.error("3. Configuration is correct")
            st.session_state.initialized = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0


def display_metadata(metadata: dict):
    with st.expander("Response Metrics", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Request ID", metadata.get("request_id", "N/A")[:8])
        with col2:
            st.metric("Retrieved Chunks", metadata.get("retrieved_chunks", 0))
        with col3:
            st.metric("Retrieval Time", f"{metadata.get('retrieval_time_sec', 0):.3f}s")
        with col4:
            st.metric("Generation Time", f"{metadata.get('generation_time_sec', 0):.3f}s")
        
        st.metric("Total Time", f"{metadata.get('total_time_sec', 0):.3f}s")
        st.caption(f"Confidence Score: {metadata.get('confidence_score', 0):.2f}")


def main():
    with st.sidebar:
        st.title("ArXiv Research Assistant")
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This RAG system answers questions about ArXiv research papers.
        
        Features:
        - Semantic search over paper corpus
        - Context-aware responses
        - Citation tracking
        - Performance metrics
        """)
        
        st.markdown("---")
        st.markdown("### Configuration")
        if st.session_state.get("initialized", False):
            st.success("System Ready")
        else:
            st.error("System Not Ready")
        
        st.markdown("---")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    st.title("ArXiv Research Assistant")
    st.markdown("Ask questions about research papers in the corpus.")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "citations" in message:
                citations = message.get("citations", [])
                if citations:
                    with st.expander("Citations", expanded=False):
                        for i, citation in enumerate(citations, 1):
                            st.markdown(f"**[{i}]** {citation}")
            
            if message["role"] == "assistant" and "metadata" in message:
                display_metadata(message["metadata"])
    
    if prompt := st.chat_input("Ask a question about the research papers..."):
        if not st.session_state.get("initialized", False):
            st.error("System not initialized. Please check the error messages above.")
            st.stop()
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_service.answer(prompt)
                    
                    st.markdown(response.answer)
                    
                    if response.citations:
                        with st.expander("Citations", expanded=False):
                            for i, citation in enumerate(response.citations, 1):
                                st.markdown(f"**[{i}]** {citation}")
                    
                    display_metadata(response.metadata)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "citations": response.citations,
                        "metadata": response.metadata,
                    })
                    
                    st.session_state.query_count += 1
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })


if __name__ == "__main__":
    main()
