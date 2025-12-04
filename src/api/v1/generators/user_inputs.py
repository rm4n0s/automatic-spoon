# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from pydantic import BaseModel, Field


class GeneratorUserInput(BaseModel):
    name: str
    engine_id: int
    gpu_id: int = Field(default=0)
