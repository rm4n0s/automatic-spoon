# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter

from .schemas import GPUSchema
from .services import GPUService

router = APIRouter()


@router.get("", response_model=list[GPUSchema])
@inject
async def get_gpus(svc: FromDishka[GPUService]):
    return svc.list_gpus()
