from fastapi import APIRouter, Depends, status

from src.api.v1.aimodels.services import AIModelService
from src.db.repositories.aimodel_repo import AIModelRepo
from src.schemas.aimodel_schemas import AIModelSchema, AIModelSchemaAsUserInput

router = APIRouter()


def get_repo() -> AIModelRepo:
    return AIModelRepo()


def get_service(repo: AIModelRepo = Depends(get_repo)) -> AIModelService:
    return AIModelService(repo)


@router.get("/", response_model=list[AIModelSchema])
async def get_aimodels(repo: AIModelRepo = Depends(get_repo)):
    return await repo.get_all()


@router.get("/{id}", response_model=AIModelSchema)
async def get_one_aimodel(id: int, repo: AIModelRepo = Depends(get_repo)):
    return await repo.get_one(id)


@router.post("/", response_model=AIModelSchema, status_code=status.HTTP_201_CREATED)
async def create_aimodel(
    payload: AIModelSchemaAsUserInput, svc: AIModelService = Depends(get_service)
):
    return await svc.create(payload)
