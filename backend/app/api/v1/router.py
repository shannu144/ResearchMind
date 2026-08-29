from fastapi import APIRouter
from app.core.config import settings
from app.api.v1.documents import router as documents_router
from app.api.v1.data_science import router as data_science_router
from app.api.v1.ml import router as ml_router
from app.api.v1.deep_learning import router as deep_learning_router
from app.api.v1.nlp import router as nlp_router
from app.api.v1.embeddings import router as embeddings_router
from app.api.v1.rag import router as rag_router
from app.api.v1.genai import router as genai_router
from app.api.v1.clustering import router as clustering_router

api_router = APIRouter()

# Register routes
api_router.include_router(documents_router)
api_router.include_router(data_science_router)
api_router.include_router(ml_router)
api_router.include_router(deep_learning_router)
api_router.include_router(nlp_router)
api_router.include_router(embeddings_router)
api_router.include_router(rag_router)
api_router.include_router(genai_router)
api_router.include_router(clustering_router)


@api_router.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint returning application status and environment info.
    """
    return {
        "status": "healthy",
        "app": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "llm_provider": settings.LLM_PROVIDER,
        "vector_store": settings.VECTOR_STORE_TYPE,
    }
