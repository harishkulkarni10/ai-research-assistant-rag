from rag_arxiv_qa.src.utils.logger import setup_logging
from rag_arxiv_qa.src.retrieval.pipeline import RetrievalPipeline
from rag_arxiv_qa.src.generation.generator import Generator
from rag_arxiv_qa.src.services.rag_service import RAGService
from rag_arxiv_qa.src.utils.config import load_config


setup_logging(level="INFO")


def main():
    config = load_config()

    retrieval_pipeline = RetrievalPipeline(config)
    generator = Generator(config)

    rag_service = RAGService(retrieval_pipeline, generator)

    response = rag_service.answer(
        "What are transformers in the world of language models?"
    )

    print("\nFinal Response Object:\n")
    print(response)


if __name__ == "__main__":
    main()