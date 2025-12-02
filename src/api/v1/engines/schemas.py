from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.api.v1.aimodels.schemas import AIModelSchema
from src.core.enums import (
    LongPromptTechnique,
    PipeType,
    Scheduler,
)


class LoraAndWeight(BaseModel):
    aimodel: AIModelSchema
    weight: float


class EngineSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: int | None = Field(default=None)
    name: str
    checkpoint_model: AIModelSchema
    lora_models: list[LoraAndWeight]
    control_net_models: list[AIModelSchema]
    embedding_models: list[AIModelSchema]
    scheduler: Scheduler
    guidance_scale: float
    seed: int
    width: int
    height: int
    steps: int
    pipe_type: PipeType
    long_prompt_technique: LongPromptTechnique | None = None
    scaling_factor_enabled: bool | None = None
    scheduler_config: dict[str, Any] | None = None
    vae_model: AIModelSchema | None = None
    controlnet_conditioning_scale: float | None = None
    control_guidance_start: float | None = None
    control_guidance_end: float | None = None
    clip_skip: int | None = None
