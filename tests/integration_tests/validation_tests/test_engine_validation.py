# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from automatic_spoon_client_sync import (
    AIModelBase,
    AIModelCaller,
    AIModelType,
    AIModelUserInput,
    CreationError,
    EngineCaller,
    EngineUserInput,
    LoraIDAndWeightInput,
    PathType,
    PipeType,
    Scheduler,
    Variant,
)
from pytsterrors import TSTError


def test_engine_validate_empty_fields():
    host = "http://localhost:8080"
    aimodel_caller = AIModelCaller(host)
    engine_caller = EngineCaller(host)
    aimodel_input = AIModelUserInput(
        name="stabilityai",
        path="stabilityai/stable-diffusion-xl-base-1.0",
        path_type=PathType.HUGGING_FACE,
        variant=Variant.FP16,
        model_base=AIModelBase.SDXL,
        model_type=AIModelType.CHECKPOINT,
    )
    checkpoint_model = aimodel_caller.create_aimodel(aimodel_input)
    assert checkpoint_model.id
    aimodel_input = AIModelUserInput(
        name="stabilityai",
        path="madebyollin/sdxl-vae-fp16-fix",
        path_type=PathType.HUGGING_FACE,
        variant=Variant.FP16,
        model_base=AIModelBase.SDXL,
        model_type=AIModelType.VAE,
    )
    vae_model = aimodel_caller.create_aimodel(aimodel_input)
    assert vae_model.id

    engine_input = EngineUserInput(
        name="",
        checkpoint_model_id=checkpoint_model.id,
        scheduler=Scheduler.EULERA,
        guidance_scale=7,
        seed=3312,
        width=1024,
        height=1024,
        steps=25,
        pipe_type=PipeType.TXT2IMG,
        lora_model_ids=[],
        conrol_net_model_ids=[],
        embedding_model_ids=[],
    )
    threw_exception = False
    try:
        engine_caller.create_engine(engine_input)
    except CreationError as exc:
        threw_exception = True
        errs = {}
        for errf in exc.error_fields:
            errs[errf.field] = errf.error

        assert "name" in errs.keys() and errs["name"] == "name can't be empty"

    assert threw_exception

    aimodel_caller.delete_aimodel(checkpoint_model.id)
    aimodel_caller.delete_aimodel(vae_model.id)


def test_engine_validate_aimodels_dont_exist():
    host = "http://localhost:8080"
    engine_caller = EngineCaller(host)

    engine_input = EngineUserInput(
        name="new engine",
        checkpoint_model_id=1,
        vae_model_id=1,
        scheduler=Scheduler.EULERA,
        guidance_scale=7,
        seed=3312,
        width=1024,
        height=1024,
        steps=25,
        pipe_type=PipeType.TXT2IMG,
        lora_model_ids=[LoraIDAndWeightInput(lora_model_id=1, weight=3)],
        conrol_net_model_ids=[1],
        embedding_model_ids=[1],
    )
    threw_exception = False
    try:
        engine_caller.create_engine(engine_input)
    except CreationError as exc:
        threw_exception = True
        errs = {}
        for errf in exc.error_fields:
            errs[errf.field] = errf.error

        print(errs)
        assert (
            "checkpoint_model_id" in errs.keys()
            and errs["checkpoint_model_id"]
            == "checkpoint model with id 1 doesn't exist"
        )
        assert (
            "vae_model_id" in errs.keys()
            and errs["vae_model_id"] == "VAE model with id 1 doesn't exist"
        )
        assert (
            "lora_model_ids" in errs.keys()
            and errs["lora_model_ids"] == "LORA model with id 1 doesn't exist"
        )
        assert (
            "conrol_net_model_ids" in errs.keys()
            and errs["conrol_net_model_ids"]
            == "control net model with id 1 doesn't exist"
        )
        assert (
            "embedding_model_ids" in errs.keys()
            and errs["embedding_model_ids"] == "embedding model with id 1 doesn't exist"
        )

        assert (
            "controlnet_conditioning_scale" in errs.keys()
            and errs["controlnet_conditioning_scale"]
            == "control net scale can't be empty when control_net_model_ids is not"
        )

        assert (
            "control_guidance_start" in errs.keys()
            and errs["control_guidance_start"]
            == "control guidance start can't be empty when control_net_model_ids is not"
        )

        assert (
            "control_guidance_end" in errs.keys()
            and errs["control_guidance_end"]
            == "control guidance end can't be empty when control_net_model_ids is not"
        )
    except TSTError as exc:
        print(exc.metadata())

    assert threw_exception
