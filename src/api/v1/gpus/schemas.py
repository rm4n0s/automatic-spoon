# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: MIT

from pydantic import BaseModel


class GPUSchema(BaseModel):
    id: int
    name: str
    total_vram_gb: float
