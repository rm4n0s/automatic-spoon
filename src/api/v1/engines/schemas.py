from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticCustomError
from pytsterrors import TSTError

from src.core.tags.user_errors import AIMODEL_NOT_FOUND_ERROR, user_error_responses
from src.db.repositories.aimodel_repo import AIModelRepo
from src.schemas.aimodel_schemas import AIModelSchema
from src.schemas.enums import (
    AIModelType,
    LongPromptTechnique,
    Scheduler,
)


class LoraIDAndWeight(BaseModel):
    lora_id: int
    weight: int


class EngineSchemaAsUserInput(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    name: str
    checkpoint_model_id: int
    scheduler: Scheduler
    guidance_scale: float
    seed: int
    width: int
    height: int
    steps: int
    lora_model_ids: list[LoraIDAndWeight] = Field(default=[])
    conrol_net_model_ids: list[int] = Field(default=[])
    embedding_model_ids: list[int] = Field(default=[])
    long_prompt_technique: LongPromptTechnique | None = None
    vae_model_id: int | None = None
    controlnet_conditioning_scale: float | None = None
    control_guidance_start: float | None = None
    control_guidance_end: float | None = None
