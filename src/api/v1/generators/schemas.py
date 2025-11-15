from pydantic import BaseModel, ConfigDict, Field

from src.api.v1.aimodels.schemas import AIModelSchema
from src.api.v1.engines.schemas import EngineSchema
from src.core.enums import FileImageType, GeneratorStatus


class GeneratorSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: int | None = Field(default=None)
    name: str
    gpu_id: int = Field(default=0)
    engine: EngineSchema
    status: GeneratorStatus


class ControlImageSchema(BaseModel):
    aimodel: AIModelSchema | None
    image_file_path: str
    controlnet_conditioning_scale: float


class ImageSchema(BaseModel):
    id: int | None
    job_id: int
    generator_id: int
    prompt: str
    negative_prompt: str
    ready: bool
    file_path: str
    seed: int | None = Field(default=None)
    guidance_scale: float | None = Field(default=None)
    width: int | None = Field(default=None)
    height: int | None = Field(default=None)
    steps: int | None = Field(default=None)
    control_images: list[ControlImageSchema] = Field(default=[])
    result_file_type: FileImageType = Field(default=FileImageType.PNG)
    control_guidance_start: float | None = None
    control_guidance_end: float | None = None


class JobSchema(BaseModel):
    id: int | None
    generator_id: int
    images: list[ImageSchema]
