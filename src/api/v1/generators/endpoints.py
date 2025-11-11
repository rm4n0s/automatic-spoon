from fastapi import APIRouter, Depends, status
from starlette.status import HTTP_202_ACCEPTED

from src.api.v1.engines.repositories import EngineRepo
from src.api.v1.generators.repositories import GeneratorRepo
from src.api.v1.generators.schemas import GeneratorSchema, GeneratorSchemaAsUserInput
from src.api.v1.generators.services import GeneratorService

router = APIRouter()


def get_generator_repo() -> GeneratorRepo:
    return GeneratorRepo()


def get_engine_repo() -> EngineRepo:
    return EngineRepo()


def get_generator_service(
    generator_repo: GeneratorRepo = Depends(get_generator_repo),
    engine_repo: EngineRepo = Depends(get_engine_repo),
) -> GeneratorService:
    return GeneratorService(generator_repo, engine_repo)


@router.get("/", response_model=list[GeneratorSchema])
async def get_generators(repo: GeneratorRepo = Depends(get_generator_repo)):
    return await repo.get_all()


@router.get("/{id}", response_model=GeneratorSchema)
async def get_generator(id: int, repo: GeneratorRepo = Depends(get_generator_repo)):
    return await repo.get_one(id)


@router.post("/", response_model=GeneratorSchema, status_code=status.HTTP_201_CREATED)
async def create_generator(
    payload: GeneratorSchemaAsUserInput,
    svc: GeneratorService = Depends(get_generator_service),
):
    return await svc.create(payload)


@router.patch(
    "/{id}/start", response_model=GeneratorSchema, status_code=status.HTTP_202_ACCEPTED
)
async def start_generator(
    id: int, svc: GeneratorService = Depends(get_generator_service)
):
    return await svc.start(id)


@router.patch(
    "/{id}/stop", response_model=GeneratorSchema, status_code=status.HTTP_202_ACCEPTED
)
async def stop_generator(
    id: int, svc: GeneratorService = Depends(get_generator_service)
):
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
