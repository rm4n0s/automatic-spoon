# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from pydantic import BaseModel


class GPUSchema(BaseModel):
    id: int
    name: str
    total_vram_gb: float
