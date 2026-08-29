import os
import time
import platform
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from app.core.config import settings
from app.core.logging import logger
from app.database.session import init_db
from app.api.v1.router import api_router


# ─── OpenAPI Metadata ─────────────────────────────────────────────────────────

OPENAPI_TAGS = [
    {"name": "Documents", "description": "Upload, list, retrieve, and preprocess research documents (PDF, DOCX, TXT, CSV)."},
    {"name": "Data Science & EDA", "description": "Corpus analytics, vocabulary richness, n-gram analysis, and tabular EDA."},
    {"name": "ML Classification", "description": "TF-IDF + traditional ML classification pipeline (LR, RF, SVM) with benchmarking."},
    {"name": "Deep Learning", "description": "PyTorch BiLSTM classifier with training loop, evaluation, and ML vs DL comparison."},
    {"name": "NLP Analytics & Transformers", "description": "Research NER, keyphrase extraction, cosine similarity, and abstractive summarization."},
    {"name": "Embeddings & Vector Database", "description": "Sentence-Transformers chunking, FAISS vector indexing, and semantic similarity search."},
    {"name": "RAG — Retrieval-Augmented Generation", "description": "Grounded question answering with source citations. Supports OpenAI, Gemini, Ollama, Mock providers."},
    {"name": "GenAI — Research Intelligence Features", "description": "Research Gap Finder, Multi-Document Comparator, Literature Review Generator, Citation Network."},
    {"name": "Document Clustering & Topic Modeling", "description": "LDA topic discovery and KMeans document partitioning."},
    {"name": "Health", "description": "System health check and API info endpoints."},
]


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.PROJECT_NAME} in [{settings.ENVIRONMENT}] mode...")

    os.makedirs(settings.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(settings.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)

    await init_db()
    logger.info("ResearchMind server ready.")

    yield

    logger.info("Shutting down ResearchMind server...")


# ─── App Factory ──────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "**ResearchMind** is an AI-Powered Research Intelligence & RAG Platform demonstrating "
        "end-to-end production ML/AI engineering:\n\n"
        "- **Data Science**: Corpus analytics, EDA, n-gram analysis\n"
        "- **ML**: TF-IDF + Logistic Regression / Random Forest / SVM classification\n"
        "- **Deep Learning**: PyTorch BiLSTM NLP classifier\n"
        "- **NLP**: Hugging Face Transformers abstractive summarization, NER, Cosine Similarity\n"
        "- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`) + FAISS vector database\n"
        "- **RAG**: Grounded retrieval-augmented generation with source page citations\n"
        "- **GenAI**: Research Gap Finder, Literature Review Generator, Document Comparator, Citation Network\n"
        "- **Clustering**: LDA Topic Modeling + KMeans document partitioning\n"
    ),
    version="1.0.0",
    contact={
        "name": "ResearchMind Platform",
        "url": "https://github.com/researchmind",
    },
    license_info={"name": "MIT License"},
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
)

# ─── Middleware ────────────────────────────────────────────────────────────────

# GZip compression for large responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    process_ms = round((time.time() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(process_ms)
    return response


# ─── Router ───────────────────────────────────────────────────────────────────

app.include_router(api_router, prefix=settings.API_V1_STR)


# ─── Root ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"], summary="Welcome")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "platform": settings.PROJECT_NAME,
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": f"{settings.API_V1_STR}/health",
        "openapi": f"{settings.API_V1_STR}/openapi.json",
    }


@app.get(f"{settings.API_V1_STR}/health/detailed", tags=["Health"], summary="Detailed system health")
async def detailed_health():
    """
    Detailed system health — reports Python version, environment, LLM provider,
    vector store status, and configured embedding model.
    """
    import torch
    from app.services.embeddings.vector_store import VectorStoreFactory
    store = VectorStoreFactory.get_vector_store()
    stats = store.get_stats()
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "embedding_dimension": settings.EMBEDDING_DIMENSION,
        "vector_store": stats["vector_store_type"],
        "indexed_vectors": stats["total_vectors"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
