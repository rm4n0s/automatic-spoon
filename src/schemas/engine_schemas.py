from pydantic import BaseModel, ConfigDict, Field

from .aimodel_schemas import AIModelSchema
from .enums import (
    EngineCommandType,
    EngineResultType,
    LongPromptTechnique,
    Scheduler,
)
from .job_schemas import JobSchema


class EngineCommand(BaseModel):
    command: EngineCommandType
    value: JobSchema | None


class EngineResult(BaseModel):
    result: EngineResultType
    value: JobSchema | str | None


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
    long_prompt_technique: LongPromptTechnique | None = None
    vae_model: AIModelSchema | None = None
    controlnet_conditioning_scale: float | None = None
    control_guidance_start: float | None = None
    control_guidance_end: float | None = None
