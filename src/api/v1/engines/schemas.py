from typing import LiteralString

from pydantic import BaseModel, ConfigDict, field_validator
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


class EngineSchemaAsUserInput(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    name: str
    checkpoint_model_id: int
    lora_model_ids: list[int]
    conrol_net_model_ids: list[int]
    embeddint_model_ids: list[int]
    scheduler: Scheduler
    guidance_scale: float
    seed: int
    width: int
    height: int
    steps: int
    long_prompt_technique: LongPromptTechnique | None = None
    vae_model_id: int | None = None
    controlnet_conditioning_scale: float | None = None
    control_guidance_start: float | None = None
    control_guidance_end: float | None = None

    @field_validator("checkpoint_model_id")
    @classmethod
    async def checkpoint_must_exist(cls, v: int, info) -> int:
        aimodel_repo: AIModelRepo = info.context.get("aimodel_repo")
        try:
            aimodel = await aimodel_repo.get_one(v)
            if aimodel.model_type != AIModelType.CHECKPOINT:
                raise PydanticCustomError(
                    "not-checkpoint",
                    "The AI model is not type of checkpoint",
                    {"checkpoint_model_id": v},
                )
        except TSTError as ex:
            if ex.tag() == AIMODEL_NOT_FOUND_ERROR:
                resp = user_error_responses[AIMODEL_NOT_FOUND_ERROR]
                raise PydanticCustomError(
                    AIMODEL_NOT_FOUND_ERROR, resp.response, {"checkpoint_model_id": v}
                )
        return v
