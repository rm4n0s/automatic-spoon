# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: MIT

from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter

from src.core.config import Config

from .schemas import InfoSchema

router = APIRouter()


@router.get("", response_model=InfoSchema)
@inject
async def info(config: FromDishka[Config]):
    info = InfoSchema(
        db_path=config.db_path,
        images_path=config.images_path,
        hugging_face_path=config.hugging_face_path,
    )
    return info
