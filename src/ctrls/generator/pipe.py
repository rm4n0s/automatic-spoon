import torch
from compel import CompelForSD, CompelForSDXL
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DiffusionPipeline,
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
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    UniPCMultistepScheduler,
)
from pytsterrors import TSTError
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl

from src.ctrls.ctrl_types import (
    AIModelBase,
    Engine,
    Job,
    LongPromptTechnique,
    Lora,
    Model,
    PathType,
    Scheduler,
    Variant,
)
from src.ctrls.ctrl_types.enums import ControlNetPose

from .pose import prepare_pose_images


def create_controlnets(cnet_models: list[Model]) -> list[ControlNetModel]:
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


def create_vae(vae: Model) -> AutoencoderKL:
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
    checkpoint: Model, vae: AutoencoderKL | None, cnets: list[ControlNetModel]
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


def load_loras(pipe: DiffusionPipeline, loras: list[Lora]):
    for lora in loras:
        print(lora)
        lora_path = lora.model.path  # Replace or comment out
        lora_weight = lora.weight
        if lora_path:
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=lora_weight)


def load_embeddings(pipe: DiffusionPipeline, embeddings: list[Model]):
    for embed in embeddings:
        pipe.load_textual_inversion(embed.path, token=embed.trigger_words)


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


def run_pipe(pipe, engine: Engine, job: Job):  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    seed = job.seed or engine.seed
    guidance_scale = job.guidance_scale or engine.guidance_scale
    num_inference_steps = job.steps or engine.steps
    control_guidance_start = job.control_guidance_start or engine.control_guidance_start
    control_guidance_end = job.control_guidance_end or engine.control_guidance_end
    height = job.height or engine.height
    width = job.width or engine.width

    prompt = job.prompt
    negative_prompt = job.negative_prompt

    prompt_embeds = None
    prompt_neg_embeds = None
    pooled_prompt_embeds = None
    negative_pooled_prompt_embeds = None

    conditioning_images, controlnet_conditioning_scale = prepare_pose_images(job)

    if engine.long_prompt_technique is not None:
        match engine.long_prompt_technique:
            case LongPromptTechnique.COMPEL:
                if engine.checkpoint_model.model_base == AIModelBase.SDXL:
                    compel = CompelForSDXL(
                        pipe,
                    )
                    conditioning = compel(
                        main_prompt=prompt, negative_prompt=negative_prompt
                    )
                    prompt_embeds = conditioning.embeds
                    pooled_prompt_embeds = conditioning.pooled_embeds
                    prompt_neg_embeds = conditioning.negative_embeds
                    negative_pooled_prompt_embeds = conditioning.negative_pooled_embeds

                if engine.checkpoint_model.model_base == AIModelBase.SD:
                    compel = CompelForSD(
                        pipe,
                    )
                    conditioning = compel(
                        prompt=prompt, negative_prompt=negative_prompt
                    )
                    prompt_embeds = conditioning.embeds
                    pooled_prompt_embeds = conditioning.pooled_embeds
                    prompt_neg_embeds = conditioning.negative_embeds
                    negative_pooled_prompt_embeds = conditioning.negative_pooled_embeds

            case LongPromptTechnique.SDEMBED:
                (
                    prompt_embeds,
                    prompt_neg_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = get_weighted_text_embeddings_sdxl(
                    pipe, prompt=prompt, neg_prompt=negative_prompt
                )

    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=prompt_neg_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        image=conditioning_images,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        control_guidance_start=control_guidance_start,
        control_guidance_end=control_guidance_end,
        height=height,
        width=width,
        generator=generator,
    ).images[0]

    output = "temporary.png"
    image.save(output)
