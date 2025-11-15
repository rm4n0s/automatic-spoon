from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, status

from .repositories import JobRepo
from .schemas import JobSchema
from .user_inputs import JobUserInput

router = APIRouter()


@router.get("/", response_model=list[JobSchema])
@inject
async def get_jobs(repo: FromDishka[JobRepo]):
    return await repo.get_all()


@router.post("/", response_model=JobSchema, status_code=status.HTTP_201_CREATED)
@inject
async def add_job(payload: JobUserInput, repo: FromDishka[JobRepo]):
    return await repo.get_all()
