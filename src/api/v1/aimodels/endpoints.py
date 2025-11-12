from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, status

from .repositories import AIModelRepo
from .schemas import AIModelSchema, AIModelSchemaAsUserInput
from .services import AIModelService

router = APIRouter()


@router.get("/", response_model=list[AIModelSchema])
@inject
async def get_aimodels(repo: FromDishka[AIModelRepo]):
    return await repo.get_all()


@router.get("/{id}", response_model=AIModelSchema)
@inject
async def get_one_aimodel(id: int, repo: FromDishka[AIModelRepo]):
    return await repo.get_one(id)


@router.post("/", response_model=AIModelSchema, status_code=status.HTTP_201_CREATED)
@inject
async def create_aimodel(
    payload: AIModelSchemaAsUserInput,
    svc: FromDishka[AIModelService],
):
    return await svc.create(payload)
