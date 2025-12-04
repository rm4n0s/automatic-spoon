# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, status

from .repositories import EngineRepo
from .schemas import EngineSchema
from .services import EngineService
from .user_inputs import EngineUserInput

router = APIRouter()


@router.get("", response_model=list[EngineSchema])
@inject
async def get_engines(repo: FromDishka[EngineRepo]):
    return await repo.get_all()


@router.get("/{id}", response_model=EngineSchema)
@inject
async def get_one_engine(id: int, repo: FromDishka[EngineRepo]):
    return await repo.get_one(id)


@router.post("", response_model=EngineSchema, status_code=status.HTTP_201_CREATED)
@inject
async def create_engine(
    payload: EngineUserInput,
    svc: FromDishka[EngineService],
):
    print("payload", payload)
    return await svc.create(payload)


@router.delete("/{id}", response_model=None, status_code=status.HTTP_204_NO_CONTENT)
@inject
async def delete(id: int, svc: FromDishka[EngineService]):
    _ = await svc.delete(id)
    return None
