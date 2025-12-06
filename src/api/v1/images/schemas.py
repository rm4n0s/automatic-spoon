# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from pydantic import BaseModel, Field

from src.api.v1.aimodels.schemas import AIModelSchema
from src.core.enums import FileImageType


class ControlNetImageSchema(BaseModel):
    aimodel: AIModelSchema | None
    image_file_path: str
    controlnet_conditioning_scale: float | None
    canny_low_threshold: int | None = None
    canny_high_threshold: int | None = None


class ImageSchema(BaseModel):
    id: int | None
    job_id: int
    generator_id: int
    prompt: str
    negative_prompt: str
    ready: bool
    file_path: str
    name: str | None = Field(default=None)
    seed: int | None = Field(default=None)
    guidance_scale: float | None = Field(default=None)
    width: int | None = Field(default=None)
    height: int | None = Field(default=None)
    steps: int | None = Field(default=None)
    control_images: list[ControlNetImageSchema] = Field(default=[])
    file_type: FileImageType = Field(default=FileImageType.PNG)
    control_guidance_start: float | None = None
    control_guidance_end: float | None = None
