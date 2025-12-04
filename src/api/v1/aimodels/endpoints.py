# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, status

from .repositories import AIModelRepo
from .schemas import AIModelSchema
from .services import AIModelService
from .user_inputs import AIModelUserInput

router = APIRouter()


@router.get("", response_model=list[AIModelSchema])
@inject
async def get_aimodels(repo: FromDishka[AIModelRepo]):
    return await repo.get_all()


@router.get("/{id}", response_model=AIModelSchema)
@inject
async def get_one_aimodel(id: int, repo: FromDishka[AIModelRepo]):
    return await repo.get_one(id)


@router.post("", response_model=AIModelSchema, status_code=status.HTTP_201_CREATED)
@inject
async def create_aimodel(
    payload: AIModelUserInput,
    svc: FromDishka[AIModelService],
):
    return await svc.create(payload)


@router.delete("/{id}", response_model=None, status_code=status.HTTP_204_NO_CONTENT)
@inject
async def delete(id: int, svc: FromDishka[AIModelService]):
    _ = await svc.delete(id)
    return None
