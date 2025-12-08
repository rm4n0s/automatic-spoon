# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from dataclasses import dataclass
from typing import Any

import torch
from compel import Compel, CompelForSD, CompelForSDXL, ReturnedEmbeddingsType
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
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    UniPCMultistepScheduler,
)
from pytsterrors import TSTError

# from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file
from sd_embed.embedding_funcs import (
    get_weighted_text_embeddings_sd15,
    get_weighted_text_embeddings_sdxl,
)

from src.api.v1.aimodels.schemas import AIModelSchema
from src.api.v1.engines.schemas import (
    EngineSchema,
    LoraAndWeight,
)
from src.api.v1.images.schemas import ImageSchema
from src.core.enums import (
    AIModelBase,
    LongPromptTechnique,
    PathType,
    PipeType,
    Scheduler,
    Variant,
)

from .pose import prepare_pose_images


def create_controlnets(cnet_models: list[AIModelSchema]) -> list[ControlNetModel]:
    print("cnet_models ", cnet_models)
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
            if variant in vae.path:
                variant = None
            return AutoencoderKL.from_pretrained(
                vae.path,
                torch_dtype=torch_dtype,
                variant=variant,
            )


def create_pipe(
    engine: EngineSchema, vae: AutoencoderKL | None, cnets: list[ControlNetModel]
) -> DiffusionPipeline:
    checkpoint = engine.checkpoint_model
    variant = str(checkpoint.variant)
    torch_dtype = torch.float16
    if checkpoint.variant == Variant.FP32:
        torch_dtype = torch.float32

    pipeline = None
    match engine.pipe_type:
        case PipeType.TXT2IMG:
            if checkpoint.model_base == AIModelBase.SD:
                if len(cnets) == 0:
                    pipeline = StableDiffusionPipeline
                else:
                    pipeline = StableDiffusionControlNetPipeline
            elif checkpoint.model_base == AIModelBase.SDXL:
                if len(cnets) == 0:
                    pipeline = StableDiffusionXLPipeline
                else:
                    pipeline = StableDiffusionXLControlNetPipeline

        case PipeType.IMG2IMG:
            if checkpoint.model_base == AIModelBase.SD:
                pipeline = StableDiffusionImg2ImgPipeline
            elif checkpoint.model_base == AIModelBase.SDXL:
                pipeline = StableDiffusionXLImg2ImgPipeline

    assert pipeline is not None
    kwargs: dict[str, Any] = {  # pyright: ignore[reportExplicitAny]
        "torch_dtype": torch_dtype,
        "use_safetensors": True,
        "variant": variant,
    }
    if vae:
        kwargs["vae"] = vae

    if len(cnets) > 0:
        if len(cnets) == 1:
            kwargs["controlnet"] = cnets[0]
        else:
            kwargs["controlnet"] = cnets

    pipe = None
    if checkpoint.path_type == PathType.FILE:
        pipe = pipeline.from_single_file(checkpoint.path, **kwargs)  # pyright: ignore[reportOptionalMemberAccess]
    elif checkpoint.path_type == PathType.HUGGING_FACE:
        pipe = pipeline.from_pretrained(checkpoint.path, **kwargs)  # pyright: ignore[reportOptionalMemberAccess]

    assert pipe is not None

    return pipe


def load_loras(pipe: DiffusionPipeline, loras: list[LoraAndWeight]):
    for lora in loras:
        lora_path = lora.aimodel.path  # Replace or comment out
        lora_weight = lora.weight
        if lora_path:
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=lora_weight)


def load_sdxl_embedding(pipe, embed_path: str, token: str):
    # Load the safetensors file
    state_dict = load_file(embed_path)

    # Extract the embeddings for the two text encoders
    if "clip_l" not in state_dict or "clip_g" not in state_dict:
        raise KeyError(
            "Embedding file missing 'clip_l' or 'clip_g' keys - verify it's an SDXL-compatible embedding."
        )

    clip_l_emb = state_dict["clip_l"].to(pipe.device).to(pipe.dtype)
    clip_g_emb = state_dict["clip_g"].to(pipe.device).to(pipe.dtype)

    # Add the token to both tokenizers
    pipe.tokenizer.add_tokens([token])
    pipe.tokenizer_2.add_tokens([token])

    # Resize the token embeddings for both text encoders
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    pipe.text_encoder_2.resize_token_embeddings(len(pipe.tokenizer_2))

    # Get the token IDs
    token_id_l = pipe.tokenizer.convert_tokens_to_ids(token)
    token_id_g = pipe.tokenizer_2.convert_tokens_to_ids(token)

    # Inject into text_encoder (CLIP-L)
    orig_embeds_l = pipe.text_encoder.get_input_embeddings()
    new_embeds_l = torch.nn.Embedding(
        orig_embeds_l.num_embeddings,
        orig_embeds_l.embedding_dim,
        padding_idx=orig_embeds_l.padding_idx,
        device=pipe.device,
        dtype=pipe.dtype,
    )
    new_embeds_l.weight.data[:] = orig_embeds_l.weight.data[:]
    if clip_l_emb.dim() == 2:  # Multi-vector: average to single vector
        new_embeds_l.weight.data[token_id_l] = clip_l_emb.mean(dim=0)
    else:
        new_embeds_l.weight.data[token_id_l] = clip_l_emb
    pipe.text_encoder.set_input_embeddings(new_embeds_l)

    # Inject into text_encoder_2 (CLIP-G)
    orig_embeds_g = pipe.text_encoder_2.get_input_embeddings()
    new_embeds_g = torch.nn.Embedding(
        orig_embeds_g.num_embeddings,
        orig_embeds_g.embedding_dim,
        padding_idx=orig_embeds_g.padding_idx,
        device=pipe.device,
        dtype=pipe.dtype,
    )
    new_embeds_g.weight.data[:] = orig_embeds_g.weight.data[:]
    if clip_g_emb.dim() == 2:  # Multi-vector: average to single vector
        new_embeds_g.weight.data[token_id_g] = clip_g_emb.mean(dim=0)
    else:
        new_embeds_g.weight.data[token_id_g] = clip_g_emb
    pipe.text_encoder_2.set_input_embeddings(new_embeds_g)

    print(f"Loaded SDXL embedding for token '{token}' from {embed_path}")


def load_embeddings(pipe: DiffusionPipeline, embeddings: list[AIModelSchema]):
    for embed in embeddings:
        trigger = ""
        if embed.trigger_neg_words is not None:
            trigger += embed.trigger_neg_words + " "

        if embed.trigger_pos_words is not None:
            trigger += embed.trigger_pos_words + " "

        if embed.model_base == AIModelBase.SDXL:
            load_sdxl_embedding(pipe, embed.path, trigger)
        else:
            pipe.load_textual_inversion(embed.path, token=trigger)


def set_scheduler(
    pipe: DiffusionPipeline, scheduler_enum: Scheduler, scheduler_config: dict[str, Any]
):
    match scheduler_enum:
        case Scheduler.EULERA:
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipe.scheduler.config, **scheduler_config
            )
        case Scheduler.EULER:
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config, **scheduler_config
            )
        case Scheduler.LMS:
            pipe.scheduler = LMSDiscreteScheduler.from_config(
                pipe.scheduler.config, **scheduler_config
            )
        case Scheduler.HEUN:
            pipe.scheduler = HeunDiscreteScheduler.from_config(
                pipe.scheduler.config, **scheduler_config
            )
        case Scheduler.DPM2:
            pipe.scheduler = KDPM2DiscreteScheduler.from_config(
                pipe.scheduler.config, **scheduler_config
            )
        case Scheduler.DPM2A:
            pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(
                pipe.scheduler.config, **scheduler_config
            )
        case Scheduler.DPM2SA:
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
                pipe.scheduler.config, **scheduler_config
            )

        case Scheduler.DPM2M:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, **scheduler_config
            )
        case Scheduler.DDIM:
            pipe.scheduler = DDIMScheduler.from_config(
                pipe.scheduler.config, **scheduler_config
            )
        case Scheduler.PLMS:
            pipe.scheduler = PNDMScheduler.from_config(
                pipe.scheduler.config, **scheduler_config
            )
        case Scheduler.UNIPC:
            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config, **scheduler_config
            )
        case Scheduler.LCM:
            pipe.scheduler = LCMScheduler.from_config(
                pipe.scheduler.config, **scheduler_config
            )
        case Scheduler.DDPM:
            pipe.scheduler = DDPMScheduler.from_config(
                pipe.scheduler.config, **scheduler_config
            )
        case Scheduler.DEIS:
            pipe.scheduler = DEISMultistepScheduler.from_config(
                pipe.scheduler.config, **scheduler_config
            )


@dataclass
class PromptEmbeds:
    prompt_embeds: torch.Tensor | None
    prompt_neg_embeds: torch.Tensor | None
    pooled_prompt_embeds: torch.Tensor | None
    negative_pooled_prompt_embeds: torch.Tensor | None


def build_long_prompt_embeds_for_sdxl(
    pipe,
    compel_instance,
    prompt: str,
    max_tokens: int = 75,
    verbose: bool = True,
):
    """
    Automatically split a long prompt into chunks and build embeddings.
    Works only for SDXL.

    Args:
        pipe: The pipe
        compel_instance: Compel object SDXL
        prompt: The full prompt string (can be very long)
        max_tokens: Max tokens per chunk (default 75 to stay safe under 77)
        verbose: Print chunking information

    Returns:
        tuple of (conditioning tensor, pooled embeddings)
    """

    tokenizer = pipe.tokenizer

    # Tokenize the full prompt to check length
    full_tokens = tokenizer(prompt, truncation=False, add_special_tokens=False)
    token_count = len(full_tokens["input_ids"])

    if token_count <= max_tokens:
        if verbose:
            print(f"✓ Prompt fits in {token_count} tokens (under {max_tokens})")

        result = compel_instance(prompt)  # Returns (conditioning, pooled)
        return result.embeds, result.pooled_embeds

    if verbose:
        print(f"⚠ Prompt is {token_count} tokens (over {max_tokens})")
        print("→ Splitting into chunks...")

    # Split prompt by commas for intelligent chunking
    parts = [p.strip() for p in prompt.split(",")]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for part in parts:
        # Count tokens for this part
        part_tokens = tokenizer(part, truncation=False, add_special_tokens=False)
        part_token_count = len(part_tokens["input_ids"])

        # If adding this part would exceed limit, save current chunk
        if current_tokens + part_token_count > max_tokens and current_chunk:
            chunk_text = ", ".join(current_chunk)
            chunks.append(chunk_text)
            if verbose:
                chunk_len = len(tokenizer(chunk_text, truncation=False)["input_ids"])
                print(f"  Chunk {len(chunks)}: {chunk_len} tokens")
            current_chunk = [part]
            current_tokens = part_token_count
        else:
            current_chunk.append(part)
            current_tokens += part_token_count

    # Add the last chunk
    if current_chunk:
        chunk_text = ", ".join(current_chunk)
        chunks.append(chunk_text)
        if verbose:
            chunk_len = len(tokenizer(chunk_text, truncation=False)["input_ids"])
            print(f"  Chunk {len(chunks)}: {chunk_len} tokens")

    if verbose:
        print(f"✓ Created {len(chunks)} chunks\n")

    # SDXL returns (conditioning, pooled) for each chunk
    chunk_results = [compel_instance(chunk) for chunk in chunks]
    # CompelForSDXL returns LabelledConditioning objects
    conditioning_list = [result.embeds for result in chunk_results]
    pooled_list = [result.pooled_embeds for result in chunk_results]

    if hasattr(compel_instance, "pad_conditioning_tensors_to_same_length"):
        padded_conditionings = compel_instance.pad_conditioning_tensors_to_same_length(
            conditioning_list
        )
        final_conditioning = torch.cat(padded_conditionings, dim=1)
    else:
        # Legacy concat method or manual concat
        if hasattr(compel_instance, "concat_conditioning"):
            final_conditioning = compel_instance.concat_conditioning(conditioning_list)
        else:
            # Manual concatenation as fallback
            final_conditioning = torch.cat(conditioning_list, dim=-2)

    # For pooled embeddings, use the last chunk's pooled embeddings
    final_pooled = pooled_list[-1]

    return (final_conditioning, final_pooled)


def enable_long_prompt(
    pipe, prompt: str, negative_prompt: str, engine: EngineSchema
) -> PromptEmbeds:
    emb = PromptEmbeds(
        prompt_embeds=None,
        prompt_neg_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
    )
    assert engine.long_prompt_technique is not None
    match engine.long_prompt_technique:
        case LongPromptTechnique.COMPEL:
            if engine.checkpoint_model.model_base == AIModelBase.SDXL:
                compel = CompelForSDXL(pipe)

                prompt_embeds, pooled_prompt_embeds = build_long_prompt_embeds_for_sdxl(
                    pipe, compel, prompt
                )
                prompt_neg_embeds, negative_pooled_prompt_embeds = (
                    build_long_prompt_embeds_for_sdxl(pipe, compel, negative_prompt)
                )

                emb.prompt_embeds = prompt_embeds
                emb.pooled_prompt_embeds = pooled_prompt_embeds  # pyright: ignore[reportAttributeAccessIssue]
                emb.prompt_neg_embeds = prompt_neg_embeds
                emb.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds  # pyright: ignore[reportAttributeAccessIssue]

            if engine.checkpoint_model.model_base == AIModelBase.SD:
                compel = Compel(
                    tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder
                )
                prompt_embeds = compel.build_conditioning_tensor(prompt)
                prompt_neg_embeds = compel.build_conditioning_tensor(negative_prompt)
                emb.prompt_embeds = prompt_embeds  # pyright: ignore[reportAttributeAccessIssue]
                emb.prompt_neg_embeds = prompt_neg_embeds  # pyright: ignore[reportAttributeAccessIssue]

        case LongPromptTechnique.SDEMBED:
            if engine.checkpoint_model.model_base == AIModelBase.SDXL:
                (
                    emb.prompt_embeds,
                    emb.prompt_neg_embeds,
                    emb.pooled_prompt_embeds,
                    emb.negative_pooled_prompt_embeds,
                ) = get_weighted_text_embeddings_sdxl(
                    pipe, prompt=prompt, neg_prompt=negative_prompt
                )

            if engine.checkpoint_model.model_base == AIModelBase.SD:
                (
                    emb.prompt_embeds,
                    emb.prompt_neg_embeds,
                ) = get_weighted_text_embeddings_sd15(
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
    kwargs = {}

    if len(img_sch.control_images) > 0:
        conditioning_images, controlnet_conditioning_scale = prepare_pose_images(
            engine, img_sch
        )

        if len(conditioning_images) == 1:
            kwargs["image"] = conditioning_images[0]
        else:
            kwargs["image"] = conditioning_images

        if len(controlnet_conditioning_scale) == 1:
            kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale[0]
        else:
            kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale

        kwargs["control_guidance_start"] = control_guidance_start
        kwargs["control_guidance_end"] = control_guidance_end

    if engine.long_prompt_technique is not None:
        prompt_embeddings = enable_long_prompt(pipe, prompt, negative_prompt, engine)

        kwargs["prompt_embeds"] = prompt_embeddings.prompt_embeds
        kwargs["prompt_neg_embeds"] = prompt_embeddings.prompt_neg_embeds
        if prompt_embeddings.pooled_prompt_embeds is not None:
            kwargs["pooled_prompt_embeds"] = prompt_embeddings.pooled_prompt_embeds

        if prompt_embeddings.negative_pooled_prompt_embeds is not None:
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

    image = pipe(**kwargs).images[0]
    print(f"saved image to {img_sch.file_path}")
    image.save(img_sch.file_path)
