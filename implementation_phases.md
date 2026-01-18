1. Big Picture Flow (Lock This In)

Final system, end state:

User Query
↓
FastAPI API
↓
LangGraph Orchestrator
↓
Dense Retriever (ChromaDB + BGE)
↓
Re-ranker (Cross-Encoder)
↓
Context Assembler (metadata + citations)
↓
LLM (vLLM inference)
↓
Post-checks (hallucination guard)
↓
Answer + Sources

Everything you build maps to one box here.
If something doesn’t fit a box, it doesn’t belong.

2. Phased Implementation Plan (7 Days, Realistic)
   Phase 0 — Cleanup & Baseline (Day 0)

Goal: One repo, one truth, zero confusion.

What happens

Remove duplicate root-level folders

Keep only rag-arxiv-qa/ as the real project

Commit once

Tech

Git

No code logic changes

This is hygiene. Seniors do this instinctively.

Phase 1 — Configuration & Contracts (Day 1)

Goal: Make the system configurable, reproducible, non-fragile.

What you implement

config.yaml (paths, models, DB choice, chunk size, top_k, etc.)

Central config loader (src/utils/config.py)

Environment-based overrides (dev, prod)

Why
Hardcoded values = junior code.
Config-driven systems = production systems.

Tech stack

Python

PyYAML

dotenv

Phase 2 — Vector Store & Indexing (Day 1–2)

Goal: Persistent, scalable retrieval layer.

What you implement

ChromaDB as the primary vector DB

FAISS only as backend index (hidden)

Metadata stored with vectors (paper_id, chunk_id, section, source)

Why

FAISS alone = in-memory toy

Chroma = persistence + filtering + production ergonomics

Tech stack

ChromaDB

SentenceTransformers (BGE)

FAISS (internal)

Output

No more parquet-based retrieval

Vectors live in DB, not notebooks

Phase 3 — Retrieval + Re-ranking (Day 2)

Goal: High recall and high precision.

What you implement

Retriever

Dense search

top_k = 50

Re-ranker

Cross-encoder

Re-score top 50 → keep top 5–10

Why
Dense retrieval is fast but fuzzy.
Re-ranking is slow but accurate.
Together = enterprise standard.

Tech stack

BGE embeddings

Cross-encoder (bge-reranker or MiniLM)

Torch (GPU-aware)

Folder

src/retrieval/
retriever.py
reranker.py
pipeline.py

Phase 4 — LangGraph Orchestration (Day 3)

Goal: Move from linear RAG to agentic RAG.

What you implement

LangGraph DAG:

Query node

Retrieval node

Re-ranking node

Generation node

Optional retry / reformulation node

Why

Shows architectural maturity

Enables self-correction later

Very marketable skill right now

Tech stack

LangGraph

LangChain (light usage, not everywhere)

Important: LangChain is a glue, not your core logic.

Phase 5 — Generation (Day 4)

Goal: Reliable, grounded answers.

What you implement

vLLM-based inference server

Open-source LLM (Mistral / Qwen / Llama)

Strict prompt with:

Retrieved context

Source grounding

No hallucination encouragement

Why

vLLM is the industry inference standard

Transformers-in-notebook ≠ production

Tech stack

vLLM

Hugging Face models

Prompt templates

Phase 6 — Evaluation & Guardrails (Day 5)

Goal: Measure quality, not vibes.

What you implement

RAGAS metrics:

Context recall

Faithfulness

LLM-as-judge hallucination check

Store results in MLflow

Why
If you can’t measure it, you can’t defend it.

Tech stack

RAGAS

MLflow

Pandas

Phase 7 — API, CI/CD, Deployment (Day 6–7)

Goal: Make it shippable.

What you implement

FastAPI serving layer

Dockerfile

docker-compose

GitHub Actions:

lint

tests

build

Optional:

Kubernetes manifests (basic)

Tech stack

FastAPI

Docker

GitHub Actions

(Optional) K8s
