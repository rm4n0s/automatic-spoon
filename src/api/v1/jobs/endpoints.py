# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, status

from src.api.v1.jobs.services import JobService
from src.core.config import Config

from .repositories import JobRepo
from .schemas import JobSchema
from .user_inputs import JobUserInput

router = APIRouter()


@router.get("", response_model=list[JobSchema])
@inject
async def get_jobs(repo: FromDishka[JobRepo]):
    return await repo.get_all()


@router.post("", response_model=JobSchema, status_code=status.HTTP_201_CREATED)
@inject
async def add_job(
    payload: JobUserInput, svc: FromDishka[JobService], config: FromDishka[Config]
):
    return await svc.create_job(config, payload)


@router.get("/{id}", response_model=JobSchema)
@inject
async def get_job(id: int, repo: FromDishka[JobRepo]):
    return await repo.get_one(id)


@router.delete("/{id}", response_model=None, status_code=status.HTTP_204_NO_CONTENT)
@inject
async def delete_job(
    id: int,
    svc: FromDishka[JobService],
):
    _ = await svc.delete_job(id)
    return None
