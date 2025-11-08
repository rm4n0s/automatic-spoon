from pydantic import BaseModel, Field

from .aimodel_schemas import AIModelSchema
from .enums import (
    ControlNetPose,
    EngineCommandEnums,
    EngineResultEnums,
    EngineStatus,
    FileImageType,
    LongPromptTechnique,
    Scheduler,
)


class Lora(BaseModel):
    model: AIModelSchema
    weight: float


class Engine(BaseModel):
    id: int
    name: str
    status: EngineStatus
    checkpoint_model: AIModelSchema
    lora_models: list[Lora]
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


class PoseImage(BaseModel):
    path: str | None
    control_net_pose: ControlNetPose
    scale: float


class Job(BaseModel):
    id: int
    prompt: str
    negative_prompt: str
    save_file_path: str
    pose_images: list[PoseImage] = Field(default_factory=list)
    reference_image_path: str | None = None
    image_file_type: FileImageType = FileImageType.PNG
    scheduler: Scheduler | None = None
    guidance_scale: float | None = None
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    controlnet_conditioning_scale: float | None = None
    control_guidance_start: float | None = None
    control_guidance_end: float | None = None


class EngineCommand(BaseModel):
    command: EngineCommandEnums
    value: Job | None


class EngineResult(BaseModel):
    result: EngineResultEnums
    value: Job | str | None
