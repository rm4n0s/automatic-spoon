from fastapi import APIRouter, Depends, status

from src.db.repositories.aimodel_repo import AIModelRepo
from src.db.repositories.engine_repo import EngineRepo
from src.schemas.engine_schemas import EngineSchema

from .schemas import EngineSchemaAsUserInput
from .services import EngineService

router = APIRouter()


def get_aimodel_repo() -> AIModelRepo:
    return AIModelRepo()


def get_engine_repo() -> EngineRepo:
    return EngineRepo()


def get_engine_service(
    engine_repo: EngineRepo = Depends(get_engine_repo),
    aimodel_repo: AIModelRepo = Depends(get_aimodel_repo),
) -> EngineService:
    return EngineService(engine_repo, aimodel_repo)


@router.get("/", response_model=list[EngineSchema])
async def get_engines(repo: EngineRepo = Depends(get_engine_repo)):
    return await repo.get_all()


@router.get("/{id}", response_model=EngineSchema)
async def get_one_engine(id: int, repo: EngineRepo = Depends(get_engine_repo)):
    return await repo.get_one(id)


@router.post("/", response_model=EngineSchema, status_code=status.HTTP_201_CREATED)
async def create_engine(
    payload: EngineSchemaAsUserInput,
    svc: EngineService = Depends(get_engine_service),
):
    return await svc.create(payload)
