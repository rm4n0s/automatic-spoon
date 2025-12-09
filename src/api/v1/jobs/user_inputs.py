# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: MIT

from typing import Any

from pydantic import BaseModel, Field

from src.core.enums import FileImageType


class ControlNetImageInput(BaseModel):
    aimodel_id: (
        int | None
    )  # if none is image reference to be used from engine's controlnets
    data_base64: str
    controlnet_conditioning_scale: float | None
    canny_low_threshold: int | None = None
    canny_high_threshold: int | None = None


class ImageUserInput(BaseModel):
    prompt: str
    negative_prompt: str
    name: str | None = Field(default=None)
    seed: int | None = Field(default=None)
    guidance_scale: float | None = Field(default=None)
    width: int | None = Field(default=None)
    height: int | None = Field(default=None)
    steps: int | None = Field(default=None)
    control_images: list[ControlNetImageInput] = Field(default=[])
    file_type: FileImageType = Field(default=FileImageType.PNG)
    control_guidance_start: float | None = None
    control_guidance_end: float | None = None


class JobUserInput(BaseModel):
    generator_id: int
    images: list[ImageUserInput]
    ip_adapter_config: dict[str, Any] | None = None
