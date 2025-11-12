from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, status

from .repositories import EngineRepo
from .schemas import EngineSchema, EngineSchemaAsUserInput
from .services import EngineService

router = APIRouter()


@router.get("/", response_model=list[EngineSchema])
@inject
async def get_engines(repo: FromDishka[EngineRepo]):
    return await repo.get_all()


@router.get("/{id}", response_model=EngineSchema)
@inject
async def get_one_engine(id: int, repo: FromDishka[EngineRepo]):
    return await repo.get_one(id)


@router.post("/", response_model=EngineSchema, status_code=status.HTTP_201_CREATED)
@inject
async def create_engine(
    payload: EngineSchemaAsUserInput,
    svc: FromDishka[EngineService],
):
    return await svc.create(payload)
