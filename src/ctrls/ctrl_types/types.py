from dataclasses import dataclass
from datetime import datetime
from .enums import (
    AIModelStatus,
    PathType,
    Variant,
    AIModelType,
    AIModelBase,
    EngineStatus,
    LongPromptTechnique,
    ControlNetPose,
    Scheduler,
    FileImageType,
)


@dataclass
class Model:
    id: int
    name: str
    status: AIModelStatus
    error: str
    path: str
    trigger_words: str
    path_type: PathType
    variant: Variant
    model_type: AIModelType
    model_base: AIModelBase
    tags: str


@dataclass
class Lora:
    model: Model
    weight: float


@dataclass
class Engine:
    id: int
    name: str
    status: EngineStatus
    long_prompt_technique: LongPromptTechnique | None
    controlnet: ControlNetPose | None
    checkpoint_model: Model
    lora_models: list[Lora]
    vae_model: Model | None
    control_net_models: list[Model]
    embedding_models: list[Model]
    rembg_model: Model | None
    scheduler: Scheduler
    guidance_scale: float
    seed: int
    width: int
    height: int
    steps: int
    controlnet_conditioning_scale: float
    control_guidance_start: float
    control_guidance_end: float


@dataclass
class PoseImage:
    path: str | None
    control_net_pose: ControlNetPose
    scale: float


@dataclass
class Job:
    id: int
    prompt: str
    negative_prompt: str
    reference_image_path: str | None
    pose_images: list[PoseImage]
    image_file_type: FileImageType
    rembgModel: Model | None
    scheduler: Scheduler | None
    guidance_scale: float | None
    seed: int | None
    width: int | None
    height: int | None
    steps: int | None
    controlnet_conditioning_scale: float | None
    control_guidance_start: float | None
    control_guidance_end: float | None
