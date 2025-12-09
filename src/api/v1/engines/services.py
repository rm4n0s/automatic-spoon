# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: MIT

from pytsterrors.exception import TSTError

from src.api.v1.aimodels.repositories import AIModelRepo
from src.core.enums import AIModelType

from .repositories import EngineRepo
from .schemas import EngineSchema, LoraAndWeight
from .user_inputs import EngineUserInput


class EngineService:
    engine_repo: EngineRepo
    aimodel_repo: AIModelRepo

    def __init__(self, engine_repo: EngineRepo, aimodel_repo: AIModelRepo):
        self.engine_repo = engine_repo
        self.aimodel_repo = aimodel_repo

    async def _validate(self, input: EngineUserInput) -> list[dict[str, str]]:
        res = []
        if input.name == "":
            res.append(
                {
                    "field": "name",
                    "error": "name can't be empty",
                }
            )

        ok = await self.aimodel_repo.exists(
            id=input.checkpoint_model_id, model_type=AIModelType.CHECKPOINT
        )
        if not ok:
            res.append(
                {
                    "field": "checkpoint_model_id",
                    "error": f"checkpoint model with id {input.checkpoint_model_id} doesn't exist",
                }
            )

        if input.vae_model_id:
            ok = await self.aimodel_repo.exists(
                id=input.vae_model_id, model_type=AIModelType.VAE
            )
            if not ok:
                res.append(
                    {
                        "field": "vae_model_id",
                        "error": f"VAE model with id {input.vae_model_id} doesn't exist",
                    }
                )

        for lw in input.lora_model_ids:
            id = lw.lora_model_id
            ok = await self.aimodel_repo.exists(id=id, model_type=AIModelType.LORA)
            if not ok:
                res.append(
                    {
                        "field": "lora_model_ids",
                        "error": f"LORA model with id {id} doesn't exist",
                    }
                )

        for id in input.conrol_net_model_ids:
            ok = await self.aimodel_repo.exists(
                id=id, model_type=AIModelType.CONTROLNET
            )
            if not ok:
                res.append(
                    {
                        "field": "conrol_net_model_ids",
                        "error": f"control net model with id {id} doesn't exist",
                    }
                )

        if len(input.conrol_net_model_ids) > 0:
            if input.controlnet_conditioning_scale is None:
                res.append(
                    {
                        "field": "controlnet_conditioning_scale",
                        "error": "control net scale can't be empty when control_net_model_ids is not",
                    }
                )

            if input.control_guidance_start is None:
                res.append(
                    {
                        "field": "control_guidance_start",
                        "error": "control guidance start can't be empty when control_net_model_ids is not",
                    }
                )

            if input.control_guidance_end is None:
                res.append(
                    {
                        "field": "control_guidance_end",
                        "error": "control guidance end can't be empty when control_net_model_ids is not",
                    }
                )

        for id in input.embedding_model_ids:
            ok = await self.aimodel_repo.exists(id=id, model_type=AIModelType.EMBEDDING)
            if not ok:
                res.append(
                    {
                        "field": "embedding_model_ids",
                        "error": f"embedding model with id {id} doesn't exist",
                    }
                )

        return res

    async def create(self, input: EngineUserInput) -> EngineSchema | None:
        errs = await self._validate(input)
        if len(errs) > 0:
            raise TSTError(
                "incorrect-input",
                "Incorrect input",
                metadata={"error_per_field": errs, "status_code": 400},
            )

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
            aimodel = await self.aimodel_repo.get_one(v.lora_model_id)
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
            pipe_type=input.pipe_type,
            long_prompt_technique=input.long_prompt_technique,
            scaling_factor_enabled=input.scaling_factor_enabled,
            scheduler_config=input.scheduler_config,
            controlnet_conditioning_scale=input.controlnet_conditioning_scale,
            control_guidance_start=input.control_guidance_start,
            control_guidance_end=input.control_guidance_end,
            clip_skip=input.clip_skip,
        )
        res = await self.engine_repo.create(engine)
        return res

    async def delete(self, id: int) -> str | None:
        if await self.engine_repo.is_used_by_generator(id):
            raise TSTError(
                "generator-is-using-engine",
                f"Engine with {id} can't be deleted because it is used by a generator",
                metadata={"status_code": 400},
            )

        await self.engine_repo.delete(id)
