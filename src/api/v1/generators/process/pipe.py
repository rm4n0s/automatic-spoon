from dataclasses import dataclass
from typing import Any

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

from src.api.v1.aimodels.schemas import AIModelSchema
from src.api.v1.engines.schemas import (
    EngineSchema,
    LoraAndWeight,
)
from src.api.v1.jobs.schemas import ImageSchema
from src.core.enums import (
    AIModelBase,
    LongPromptTechnique,
    PathType,
    Scheduler,
    Variant,
)

from .pose import prepare_pose_images


def create_controlnets(cnet_models: list[AIModelSchema]) -> list[ControlNetModel]:
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


def create_vae(vae: AIModelSchema) -> AutoencoderKL:
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
    checkpoint: AIModelSchema, vae: AutoencoderKL | None, cnets: list[ControlNetModel]
) -> DiffusionPipeline:
    print(vae)
    variant = str(checkpoint.variant)
    torch_dtype = torch.float16
    if checkpoint.variant == Variant.FP32:
        torch_dtype = torch.float32

    kwargs: dict[str, Any] = {  # pyright: ignore[reportExplicitAny]
        "torch_dtype": torch_dtype,
        "use_safetensors": True,
        "variant": variant,
    }
    # WITHOUT CONTROLL NET
    if (
        checkpoint.model_base == AIModelBase.SD
        and checkpoint.path_type == PathType.FILE
        and len(cnets) == 0
    ):
        if vae:
            kwargs["vae"] = vae
        pipe = StableDiffusionPipeline.from_single_file(checkpoint.path, **kwargs)
    elif (
        checkpoint.model_base == AIModelBase.SDXL
        and checkpoint.path_type == PathType.FILE
        and len(cnets) == 0
    ):
        if vae:
            kwargs["vae"] = vae
        pipe = StableDiffusionXLPipeline.from_single_file(checkpoint.path, **kwargs)
    elif (
        checkpoint.model_base == AIModelBase.SD
        and checkpoint.path_type == PathType.HUGGING_FACE
        and len(cnets) == 0
    ):
        if vae:
            kwargs["vae"] = vae
        pipe = StableDiffusionPipeline.from_pretrained(checkpoint.path, **kwargs)
    elif (
        checkpoint.model_base == AIModelBase.SDXL
        and checkpoint.path_type == PathType.HUGGING_FACE
        and len(cnets) == 0
    ):
        if vae:
            kwargs["vae"] = vae
        pipe = StableDiffusionXLPipeline.from_pretrained(checkpoint.path, **kwargs)

    # WITH CONTROLL NET
    elif (
        checkpoint.model_base == AIModelBase.SD
        and checkpoint.path_type == PathType.FILE
        and len(cnets) > 0
    ):
        kwargs["controlnet"] = cnets
        if vae:
            kwargs["vae"] = vae
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            checkpoint.path, **kwargs
        )
    elif (
        checkpoint.model_base == AIModelBase.SDXL
        and checkpoint.path_type == PathType.FILE
        and len(cnets) > 0
    ):
        kwargs["controlnet"] = cnets
        if vae:
            kwargs["vae"] = vae
        pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            checkpoint.path, **kwargs
        )
    elif (
        checkpoint.model_base == AIModelBase.SD
        and checkpoint.path_type == PathType.HUGGING_FACE
        and len(cnets) > 0
    ):
        kwargs["controlnet"] = cnets
        if vae:
            kwargs["vae"] = vae
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            checkpoint.path, **kwargs
        )
    elif (
        checkpoint.model_base == AIModelBase.SDXL
        and checkpoint.path_type == PathType.HUGGING_FACE
        and len(cnets) > 0
    ):
        kwargs["controlnet"] = cnets
        if vae:
            kwargs["vae"] = vae
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            checkpoint.path, **kwargs
        )
    else:
        raise TSTError(
            "combination-not-found",
            "The selected combination of checkpoint model base, path type and control nets not found while creating the pipe",
        )

    return pipe


def load_loras(pipe: DiffusionPipeline, loras: list[LoraAndWeight]):
    for lora in loras:
        print(lora)
        lora_path = lora.aimodel.path  # Replace or comment out
        lora_weight = lora.weight
        if lora_path:
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=lora_weight)


def load_embeddings(pipe: DiffusionPipeline, embeddings: list[AIModelSchema]):
    for embed in embeddings:
        trigger = ""
        if embed.trigger_neg_words is not None:
            trigger = embed.trigger_neg_words

        if embed.trigger_pos_words is not None:
            trigger = embed.trigger_pos_words

        pipe.load_textual_inversion(embed.path, token=trigger)


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


@dataclass
class PromptEmbeds:
    prompt_embeds: torch.Tensor | None
    prompt_neg_embeds: torch.Tensor | None
    pooled_prompt_embeds: torch.Tensor | None
    negative_pooled_prompt_embeds: torch.Tensor | None


def enable_long_prompt(
    pipe, prompt: str, negative_prompt: str, engine: EngineSchema
) -> PromptEmbeds | None:
    if engine.long_prompt_technique is None:
        return None

    emb = PromptEmbeds(
        prompt_embeds=None,
        prompt_neg_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
    )
    match engine.long_prompt_technique:
        case LongPromptTechnique.COMPEL:
            if engine.checkpoint_model.model_base == AIModelBase.SDXL:
                compel = CompelForSDXL(
                    pipe,
                )
                conditioning = compel(
                    main_prompt=prompt, negative_prompt=negative_prompt
                )
                emb.prompt_embeds = conditioning.embeds
                emb.pooled_prompt_embeds = conditioning.pooled_embeds
                emb.prompt_neg_embeds = conditioning.negative_embeds
                emb.negative_pooled_prompt_embeds = conditioning.negative_pooled_embeds

            if engine.checkpoint_model.model_base == AIModelBase.SD:
                compel = CompelForSD(
                    pipe,
                )
                conditioning = compel(prompt=prompt, negative_prompt=negative_prompt)
                emb.prompt_embeds = conditioning.embeds
                emb.pooled_prompt_embeds = conditioning.pooled_embeds
                emb.prompt_neg_embeds = conditioning.negative_embeds
                emb.negative_pooled_prompt_embeds = conditioning.negative_pooled_embeds

        case LongPromptTechnique.SDEMBED:
            (
                emb.prompt_embeds,
                emb.prompt_neg_embeds,
                emb.pooled_prompt_embeds,
                emb.negative_pooled_prompt_embeds,
            ) = get_weighted_text_embeddings_sdxl(
                pipe, prompt=prompt, neg_prompt=negative_prompt
            )

    return emb


def run_pipe(pipe, engine: EngineSchema, img_sch: ImageSchema):  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
    seed = img_sch.seed or engine.seed
    guidance_scale = img_sch.guidance_scale or engine.guidance_scale
    num_inference_steps = img_sch.steps or engine.steps
    control_guidance_start = (
        img_sch.control_guidance_start or engine.control_guidance_start
    )
    control_guidance_end = img_sch.control_guidance_end or engine.control_guidance_end
    height = img_sch.height or engine.height
    width = img_sch.width or engine.width

    prompt = img_sch.prompt
    negative_prompt = img_sch.negative_prompt

    conditioning_images, controlnet_conditioning_scale = prepare_pose_images(
        engine, img_sch
    )

    prompt_embeddings = enable_long_prompt(pipe, prompt, negative_prompt, engine)
    kwargs = {}

    if prompt_embeddings is not None:
        kwargs["prompt_embeds"] = prompt_embeddings.prompt_embeds
        kwargs["prompt_neg_embeds"] = prompt_embeddings.prompt_neg_embeds
        kwargs["pooled_prompt_embeds"] = prompt_embeddings.pooled_prompt_embeds
        kwargs["negative_pooled_prompt_embeds"] = (
            prompt_embeddings.negative_pooled_prompt_embeds
        )
    else:
        kwargs["prompt"] = prompt
        kwargs["negative_prompt"] = negative_prompt

    generator = torch.Generator(device="cuda").manual_seed(seed)
    kwargs["generator"] = generator
    kwargs["height"] = height
    kwargs["width"] = width
    kwargs["guidance_scale"] = guidance_scale
    kwargs["num_inference_steps"] = num_inference_steps

    if conditioning_images is not None:
        kwargs["image"] = conditioning_images
        kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale
        kwargs["control_guidance_start"] = control_guidance_start
        kwargs["control_guidance_end"] = control_guidance_end

    image = pipe(**kwargs).images[0]
    print(f"saved image to {img_sch.file_path}")
    image.save(img_sch.file_path)
