# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, status
from starlette.status import HTTP_202_ACCEPTED

from .repositories import GeneratorRepo
from .schemas import GeneratorSchema
from .services import GeneratorService
from .user_inputs import GeneratorUserInput

router = APIRouter()


@router.get("", response_model=list[GeneratorSchema])
@inject
async def get_generators(repo: FromDishka[GeneratorRepo]):
    return await repo.get_all()


@router.get("/{id}", response_model=GeneratorSchema)
@inject
async def get_generator(id: int, repo: FromDishka[GeneratorRepo]):
    return await repo.get_one(id)


@router.post("", response_model=GeneratorSchema, status_code=status.HTTP_201_CREATED)
@inject
async def create_generator(
    payload: GeneratorUserInput,
    svc: FromDishka[GeneratorService],
):
    return await svc.create(payload)


@router.patch(
    "/{id}/start", response_model=GeneratorSchema, status_code=status.HTTP_202_ACCEPTED
)
@inject
async def start_generator(id: int, svc: FromDishka[GeneratorService]):
    return await svc.start_generator(id)


@router.patch(
    "/{id}/close", response_model=GeneratorSchema, status_code=status.HTTP_202_ACCEPTED
)
@inject
async def close_generator(id: int, svc: FromDishka[GeneratorService]):
    return await svc.close_generator(id)


@router.delete("/{id}", response_model=None, status_code=status.HTTP_204_NO_CONTENT)
@inject
async def delete_generator(
    id: int,
    svc: FromDishka[GeneratorService],
):
    _ = await svc.delete_generator(id)
    return None
