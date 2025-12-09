# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: MIT

from pydantic import BaseModel, ConfigDict, Field

from src.api.v1.engines.schemas import EngineSchema
from src.core.enums import GeneratorStatus


class GeneratorSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: int | None = Field(default=None)
    name: str
    gpu_id: int = Field(default=0)
    engine: EngineSchema
    status: GeneratorStatus
