# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from pydantic import BaseModel

from src.api.v1.images.schemas import ImageSchema
from src.core.enums import JobStatus


class JobSchema(BaseModel):
    id: int | None
    generator_id: int
    images: list[ImageSchema]
    status: JobStatus
