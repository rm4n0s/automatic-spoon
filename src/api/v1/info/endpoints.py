from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter

from src.core.config import Config

router = APIRouter()


@router.get("/")
@inject
async def info(config: FromDishka[Config]):
    is_on_mem = config.db_path == ":memory:"
    return {"is_db_on_memory": is_on_mem}
