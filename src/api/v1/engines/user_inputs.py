# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.enums import (
    LongPromptTechnique,
    PipeType,
    Scheduler,
)


class LoraIDAndWeightInput(BaseModel):
    lora_model_id: int
    weight: int


class EngineUserInput(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    name: str
    checkpoint_model_id: int
    scheduler: Scheduler
    guidance_scale: float
    seed: int
    width: int
    height: int
    steps: int
    pipe_type: PipeType
    lora_model_ids: list[LoraIDAndWeightInput] = Field(default=[])
    conrol_net_model_ids: list[int] = Field(default=[])
    embedding_model_ids: list[int] = Field(default=[])
    long_prompt_technique: LongPromptTechnique | None = None
    vae_model_id: int | None = None
    scaling_factor_enabled: bool | None = None
    scheduler_config: dict[str, Any] | None = None
    controlnet_conditioning_scale: float | None = None
    control_guidance_start: float | None = None
    control_guidance_end: float | None = None
    clip_skip: int | None = None
