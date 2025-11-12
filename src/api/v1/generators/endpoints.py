from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, Depends, status
from starlette.status import HTTP_202_ACCEPTED

from src.api.v1.generators.repositories import GeneratorRepo
from src.api.v1.generators.schemas import GeneratorSchema, GeneratorSchemaAsUserInput
from src.api.v1.generators.services import GeneratorService

router = APIRouter()


@router.get("/", response_model=list[GeneratorSchema])
@inject
async def get_generators(repo: FromDishka[GeneratorRepo]):
    return await repo.get_all()


@router.get("/{id}", response_model=GeneratorSchema)
@inject
async def get_generator(id: int, repo: FromDishka[GeneratorRepo]):
    return await repo.get_one(id)


@router.post("/", response_model=GeneratorSchema, status_code=status.HTTP_201_CREATED)
@inject
async def create_generator(
    payload: GeneratorSchemaAsUserInput,
    svc: FromDishka[GeneratorService],
):
    return await svc.create(payload)


@router.patch(
    "/{id}/start", response_model=GeneratorSchema, status_code=status.HTTP_202_ACCEPTED
)
@inject
async def start_generator(id: int, svc: FromDishka[GeneratorService]):
    return await svc.start(id)


@router.patch(
    "/{id}/stop", response_model=GeneratorSchema, status_code=status.HTTP_202_ACCEPTED
)
@inject
async def stop_generator(id: int, svc: FromDishka[GeneratorService]):
    return await svc.stop(id)


@router.patch("/{id}/stop/force")
async def stop_force_generator():
    pass


@router.delete("/{id}")
async def delete_generator():
    pass


@router.delete("/{id}/force")
async def delete_force_generator():
    pass
