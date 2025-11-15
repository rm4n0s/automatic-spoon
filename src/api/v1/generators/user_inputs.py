from pydantic import BaseModel, Field

from src.core.enums import FileImageType


class GeneratorUserInput(BaseModel):
    name: str
    engine_id: int
    gpu_id: int = Field(default=0)


class ControlImageInput(BaseModel):
    aimodel_id: (
        int | None
    )  # if none is image reference to be used from engine's controlnets
    data_base64: str
    controlnet_conditioning_scale: float | None = None
    control_guidance_start: float | None = None
    control_guidance_end: float | None = None


class ImageUserInput(BaseModel):
    prompt: str
    negative: str
    seed: int | None = Field(default=None)
    guidance_scale: float | None = Field(default=None)
    width: int | None = Field(default=None)
    height: int | None = Field(default=None)
    steps: int | None = Field(default=None)
    control_images: list[ControlImageInput] = Field(default=[])
    result_file_type: FileImageType = Field(default=FileImageType.PNG)


class JobUserInput(BaseModel):
    generator_id: int
    images: list[ImageUserInput]
