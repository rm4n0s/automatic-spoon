from pydantic import BaseModel, Field

from .aimodel_schemas import AIModelSchema
from .enums import (
    FileImageType,
    Scheduler,
)


class PoseAndScale(BaseModel):
    aimodel: AIModelSchema
    scale: float


class JobSchema(BaseModel):
    id: int
    prompt: str
    negative_prompt: str
    save_file_path: str
    pose_images: list[PoseAndScale] = Field(default_factory=list)
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
