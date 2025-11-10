from pytsterrors.exception import TSTError

from src.core.tags.user_errors import WRONG_INPUT
from src.db.repositories.aimodel_repo import AIModelRepo
from src.db.repositories.engine_repo import EngineRepo
from src.schemas.engine_schemas import EngineSchema, LoraAndWeight
from src.schemas.enums import AIModelType

from .schemas import EngineSchemaAsUserInput


class EngineService:
    engine_repo: EngineRepo
    aimodel_repo: AIModelRepo

    def __init__(self, engine_repo: EngineRepo, aimodel_repo: AIModelRepo):
        self.engine_repo = engine_repo
        self.aimodel_repo = aimodel_repo

    async def _validate(self, input: EngineSchemaAsUserInput) -> list[dict[str, str]]:
        res = []
        ok = await self.aimodel_repo.exists(
            id=input.checkpoint_model_id, model_type=AIModelType.CHECKPOINT
        )
        if not ok:
            res.append(
                {
                    "field": "checkpoint_id",
                    "error": f"the ID {input.checkpoint_model_id} doesn't exit in aimodels",
                }
            )

        return res

    async def create(self, input: EngineSchemaAsUserInput) -> EngineSchema | None:
        errs = await self._validate(input)
        if len(errs) > 0:
            raise TSTError(WRONG_INPUT, "", metadata={"error_per_field": errs})

        checkpoint = await self.aimodel_repo.get_one(input.checkpoint_model_id)
        vae = None
        if input.vae_model_id:
            vae = await self.aimodel_repo.get_one(input.vae_model_id)

        control_nets = []
        for v in input.conrol_net_model_ids:
            cn = await self.aimodel_repo.get_one(v)
            control_nets.append(cn)

        embeddings = []
        for v in input.embedding_model_ids:
            em = await self.aimodel_repo.get_one(v)
            embeddings.append(em)

        loras = []
        for v in input.lora_model_ids:
            aimodel = await self.aimodel_repo.get_one(v.lora_id)
            lora = LoraAndWeight(aimodel=aimodel, weight=v.weight)
            loras.append(lora)

        engine = EngineSchema(
            name=input.name,
            checkpoint_model=checkpoint,
            vae_model=vae,
            control_net_models=control_nets,
            lora_models=loras,
            embedding_models=embeddings,
            scheduler=input.scheduler,
            guidance_scale=input.guidance_scale,
            steps=input.steps,
            seed=input.seed,
            width=input.width,
            height=input.height,
            long_prompt_technique=input.long_prompt_technique,
            controlnet_conditioning_scale=input.controlnet_conditioning_scale,
            control_guidance_start=input.control_guidance_start,
            control_guidance_end=input.control_guidance_end,
        )
        res = await self.engine_repo.create(engine)
        return res
