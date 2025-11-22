from fastapi import APIRouter

from .aimodels.endpoints import router as aimodels_router
from .engines.endpoints import router as engine_router
from .generators.endpoints import router as generator_router
from .gpus.endpoints import router as gpus_router
from .images.endpoints import router as image_router
from .info.endpoints import router as info_router
from .jobs.endpoints import router as job_router

api_router = APIRouter()
api_router.include_router(info_router, prefix="/info", tags=["info"])
api_router.include_router(gpus_router, prefix="/gpus", tags=["gpus"])
api_router.include_router(aimodels_router, prefix="/aimodels", tags=["aimodels"])
api_router.include_router(engine_router, prefix="/engines", tags=["engines"])
api_router.include_router(generator_router, prefix="/generators", tags=["generators"])
api_router.include_router(job_router, prefix="/jobs", tags=["jobs"])
api_router.include_router(image_router, prefix="/images", tags=["images"])
