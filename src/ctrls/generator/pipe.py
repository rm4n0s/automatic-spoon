from pytsterrors import TSTError
from src.models import Scheduler, AIModel, PathType, Variant, AIModelBase
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionControlNetPipeline,
    DiffusionPipeline,
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)


def create_controlnets(cnet_models: list[AIModel]) -> list[ControlNetModel]:
    cnets = []
    for v in cnet_models:
        variant = str(v.variant)
        torch_dtype = torch.float16
        if v.variant == Variant.FP32:
            torch_dtype = torch.float32

        match v.path_type:
            case PathType.FILE:
                cnm = ControlNetModel.from_single_file(
                    v.path,
                    torch_dtype=torch_dtype,
                    variant=variant,
                )
                cnets.append(cnm)
            case PathType.HUGGING_FACE:
                cnm = ControlNetModel.from_pretrained(
                    v.path,
                    torch_dtype=torch_dtype,
                    variant=variant,
                )
                cnets.append(cnm)
    return cnets

def create_vae(vae: AIModel) -> AutoencoderKL:
    variant = str(vae.variant)
    torch_dtype = torch.float16
    if vae.variant == Variant.FP32:
        torch_dtype = torch.float32
    
    match vae.path_type:
        case PathType.FILE:
            return AutoencoderKL.from_single_file(
                vae.path,
                torch_dtype=torch_dtype,
                variant=variant,
            )
        case PathType.HUGGING_FACE:
            return AutoencoderKL.from_pretrained(
                vae.path,
                torch_dtype=torch_dtype,
                variant=variant,
            )

def create_pipe(
    checkpoint: AIModel, vae: AutoencoderKL | None, cnets: list[ControlNetModel]
) -> DiffusionPipeline:
    variant = str(checkpoint.variant)
    torch_dtype = torch.float16
    if checkpoint.variant == Variant.FP32:
        torch_dtype = torch.float32

    # WITHOUT CONTROLL NET
    if (
        checkpoint.model_base == AIModelBase.SD
        and checkpoint.path_type == PathType.FILE
        and len(cnets) == 0
    ):
        pipe = StableDiffusionPipeline.from_single_file(
            checkpoint.path,
            vae=vae,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant=variant,
        )
    elif (
        checkpoint.model_base == AIModelBase.SDXL
        and checkpoint.path_type == PathType.FILE
        and len(cnets) == 0
    ):
        pipe = StableDiffusionXLPipeline.from_single_file(
            checkpoint.path,
            vae=vae,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant=variant,
        )
    elif (
        checkpoint.model_base == AIModelBase.SD
        and checkpoint.path_type == PathType.HUGGING_FACE
        and len(cnets) == 0
    ):
        pipe = StableDiffusionPipeline.from_pretrained(
            checkpoint.path,
            vae=vae,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant=variant,
        )
    elif (
        checkpoint.model_base == AIModelBase.SDXL
        and checkpoint.path_type == PathType.HUGGING_FACE
        and len(cnets) == 0
    ):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            checkpoint.path,
            vae=vae,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant=variant,
        )

    # WITH CONTROLL NET
    if (
        checkpoint.model_base == AIModelBase.SD
        and checkpoint.path_type == PathType.FILE
        and len(cnets) > 0
    ):
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            checkpoint.path,
            vae=vae,
            controlnet=cnets,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant=variant,
        )
    elif (
        checkpoint.model_base == AIModelBase.SDXL
        and checkpoint.path_type == PathType.FILE
        and len(cnets) > 0
    ):
        pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            checkpoint.path,
            vae=vae,
            controlnet=cnets,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant=variant,
        )
    elif (
        checkpoint.model_base == AIModelBase.SD
        and checkpoint.path_type == PathType.HUGGING_FACE
        and len(cnets) > 0
    ):
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            checkpoint.path,
            vae=vae,
            controlnet=cnets,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant=variant,
        )
    elif (
        checkpoint.model_base == AIModelBase.SDXL
        and checkpoint.path_type == PathType.HUGGING_FACE
        and len(cnets) > 0
    ):
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            checkpoint.path,
            vae=vae,
            controlnet=cnets,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant=variant,
        )
    else:
        raise TSTError(
            "combination-not-found",
            "Combination of checkpoint model base, path type and control nets not found while creating the pipe",
        )

    return pipe


def set_scheduler(pipe: DiffusionPipeline, scheduler_enum: Scheduler):
    match scheduler_enum:
        case Scheduler.EULERA:
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipe.scheduler.config
            )
        case Scheduler.EULER:
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        case Scheduler.LMS:
            pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        case Scheduler.HEUN:
            pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
        case Scheduler.DPM2:
            pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
        case Scheduler.DPM2A:
            pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(
                pipe.scheduler.config
            )
        case Scheduler.DPM2SA:
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
                pipe.scheduler.config
            )
        case Scheduler.DPM2M:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config
            )
        case Scheduler.DPMSDE:
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
                pipe.scheduler.config, algorithm_type="sde-dpmsolver++"
            )
        case Scheduler.LMSKARRAS:
            pipe.scheduler = LMSDiscreteScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True
            )
        case Scheduler.DPM2KARRAS:
            pipe.scheduler = KDPM2DiscreteScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True
            )
        case Scheduler.DPM2AKARRAS:
            pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True
            )
        case Scheduler.DPM2SAKARRAS:
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True
            )
        case Scheduler.DPM2MKARRAS:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True
            )
        case Scheduler.DPMSDEKARRAS:
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="sde-dpmsolver++",
            )
        case Scheduler.DDIM:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        case Scheduler.PLMS:
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        case Scheduler.UNIPC:
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        case Scheduler.LCM:
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        case Scheduler.DDPM:
            pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        case Scheduler.DEIS:
            pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
