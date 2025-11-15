from fastapi import APIRouter

from src.core.utils.list_gpus import GPU, list_gpus

router = APIRouter()


@router.get("/", response_model=list[GPU])
async def get_gpus():
    return list_gpus()
