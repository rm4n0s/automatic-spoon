from fastapi import APIRouter, Depends, status

from .repositories import AIModelRepo
from .schemas import AIModelSchema, AIModelSchemaAsUserInput
from .services import AIModelService

router = APIRouter()


def get_aimodel_repo() -> AIModelRepo:
    return AIModelRepo()


def get_aimodel_service(
    repo: AIModelRepo = Depends(get_aimodel_repo),
) -> AIModelService:
    return AIModelService(repo)


@router.get("/", response_model=list[AIModelSchema])
async def get_aimodels(repo: AIModelRepo = Depends(get_aimodel_repo)):
    return await repo.get_all()


@router.get("/{id}", response_model=AIModelSchema)
async def get_one_aimodel(id: int, repo: AIModelRepo = Depends(get_aimodel_repo)):
    return await repo.get_one(id)


@router.post("/", response_model=AIModelSchema, status_code=status.HTTP_201_CREATED)
async def create_aimodel(
    payload: AIModelSchemaAsUserInput,
    svc: AIModelService = Depends(get_aimodel_service),
):
    return await svc.create(payload)
