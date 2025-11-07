from dataclasses import dataclass, field

from .enums import (
    AIModelBase,
    AIModelStatus,
    AIModelType,
    ControlNetPose,
    EngineCommandEnums,
    EngineResultEnums,
    EngineStatus,
    FileImageType,
    LongPromptTechnique,
    PathType,
    Scheduler,
    Variant,
)


@dataclass
class Model:
    id: int
    name: str
    status: AIModelStatus
    path: str
    path_type: PathType
    variant: Variant
    model_type: AIModelType
    model_base: AIModelBase
    tags: str
    trigger_pos_words: str | None = None
    trigger_neg_words: str | None = None
    error: str | None = None


@dataclass
class Lora:
    model: Model
    weight: float


@dataclass
class Engine:
    id: int
    name: str
    status: EngineStatus
    checkpoint_model: Model
    lora_models: list[Lora]
    control_net_models: list[Model]
    embedding_models: list[Model]
    scheduler: Scheduler
    guidance_scale: float
    seed: int
    width: int
    height: int
    steps: int
    long_prompt_technique: LongPromptTechnique | None = None
    vae_model: Model | None = None
    controlnet_conditioning_scale: float | None = None
    control_guidance_start: float | None = None
    control_guidance_end: float | None = None


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
    save_file_path: str
    pose_images: list[PoseImage] = field(default_factory=list)
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


@dataclass
class EngineCommand:
    command: EngineCommandEnums
    value: Job | None


@dataclass
class EngineResult:
    result: EngineResultEnums
    value: Job | str | None
