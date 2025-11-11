from pytsterrors import TSTError

from src.api.v1.aimodels.schemas import AIModelSchema
from src.core.enums import AIModelType
from src.core.tags.user_errors import ENGINE_NOT_FOUND_ERROR
from src.db.models import AIModel, AIModelForEngine, Engine

from .schemas import EngineSchema, LoraAndWeight


class EngineRepo:
    async def create(self, input: EngineSchema) -> EngineSchema:
        e = await Engine.create(
            name=input.name,
            long_prompt_technique=input.long_prompt_technique,
            scheduler=input.scheduler,
            guidance_scale=input.guidance_scale,
            seed=input.seed,
            width=input.width,
            height=input.height,
            steps=input.steps,
            controlnet_conditioning_scale=input.controlnet_conditioning_scale,
            control_guidance_start=input.control_guidance_start,
            control_guidance_end=input.control_guidance_end,
        )
        input.id = e.id
        _ = await AIModelForEngine.create(
            engine_id=e.id,
            model_id=input.checkpoint_model.id,
            model_type=AIModelType.CHECKPOINT,
        )

        if input.vae_model:
            _ = await AIModelForEngine.create(
                engine_id=e.id,
                model_id=input.vae_model.id,
                model_type=AIModelType.VAE,
            )

        for v in input.embedding_models:
            _ = await AIModelForEngine.create(
                engine_id=e.id,
                model_id=v.id,
                model_type=AIModelType.EMBEDDING,
            )

        for v in input.lora_models:
            _ = await AIModelForEngine.create(
                engine_id=e.id,
                model_id=v.aimodel.id,
                weight=v.weight,
                model_type=AIModelType.LORA,
            )

        for v in input.control_net_models:
            _ = await AIModelForEngine.create(
                engine_id=e.id,
                model_id=v.id,
                model_type=AIModelType.CONTROLNET,
            )

        return input

    async def get_all(self) -> list[EngineSchema]:
        es = await Engine.all()
        res = []
        for e in es:
            res.append(await serialize_engine(e))
        return res

    async def get_one(self, id: int) -> EngineSchema:
        e = await Engine.get_or_none(id=id)
        if not e:
            raise TSTError(ENGINE_NOT_FOUND_ERROR, f"Engine with ID {id} not found")

        return await serialize_engine(e)

    async def exists(self, id: int) -> bool:
        return await Engine.exists(id=id)


async def serialize_engine(e: Engine) -> EngineSchema:
    aoes = await AIModelForEngine.filter(engine_id=e.id)

    checkpoint_model = None
    vae_model = None
    lora_models = []
    embedding_models = []
    controlnet_models = []
    for v in aoes:
        match v.model_type:
            case AIModelType.CHECKPOINT:
                checkpoint = await AIModel.get_or_none(id=v.id)
                if checkpoint:
                    checkpoint_model = AIModelSchema.model_validate(checkpoint)
            case AIModelType.EMBEDDING:
                embedding = await AIModel.get_or_none(id=v.id)
                if embedding:
                    embedding_model = AIModelSchema.model_validate(embedding)
                    embedding_models.append(embedding_model)
            case AIModelType.VAE:
                vae = await AIModel.get_or_none(id=v.id)
                if vae:
                    vae_model = AIModelSchema.model_validate(vae)
            case AIModelType.LORA:
                lora = await AIModel.get_or_none(id=v.id)
                if lora:
                    lora_model = AIModelSchema.model_validate(lora)
                    lw = LoraAndWeight(aimodel=lora_model, weight=v.weight)
                    lora_models.append(lw)
            case AIModelType.CONTROLNET:
                controlnet = await AIModel.get_or_none(id=v.id)
                if controlnet:
                    controlnet_model = AIModelSchema.model_validate(controlnet)
                    controlnet_models.append(controlnet_model)

    if checkpoint_model is None:
        raise TSTError(
            "checkpoint-not-found",
            f"Checkpoint that should exist for Engine with ID {e.id}, is not found",
        )

    return EngineSchema(
        id=e.id,
        name=e.name,
        checkpoint_model=checkpoint_model,
        vae_model=vae_model,
        lora_models=lora_models,
        embedding_models=embedding_models,
        control_net_models=controlnet_models,
        scheduler=e.scheduler,
        guidance_scale=e.guidance_scale,
        seed=e.seed,
        width=e.width,
        height=e.height,
        steps=e.steps,
        long_prompt_technique=e.long_prompt_technique,
        controlnet_conditioning_scale=e.controlnet_conditioning_scale,
        control_guidance_start=e.control_guidance_start,
        control_guidance_end=e.control_guidance_end,
    )
