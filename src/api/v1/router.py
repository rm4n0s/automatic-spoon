from fastapi import APIRouter

from .aimodels.endpoints import router as aimodels_router

api_router = APIRouter()
api_router.include_router(aimodels_router, prefix="/aimodels", tags=["aimodels"])
