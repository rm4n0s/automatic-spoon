# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

import os
import time

from automatic_spoon_client_sync import (
    AIModelBase,
    AIModelCaller,
    AIModelType,
    AIModelUserInput,
    EngineCaller,
    EngineUserInput,
    GeneratorCaller,
    GeneratorStatus,
    GeneratorUserInput,
    ImageCaller,
    ImageUserInput,
    InfoCaller,
    JobCaller,
    JobStatus,
    JobUserInput,
    PathType,
    PipeType,
    Scheduler,
    Variant,
)

from src.utils import read_test_config


def test_sdxl_image_creation():
    host = "http://localhost:8080"
    info_caller = InfoCaller(host)
    aimodel_caller = AIModelCaller(host)
    engine_caller = EngineCaller(host)
    generator_caller = GeneratorCaller(host)
    job_caller = JobCaller(host)
    img_caller = ImageCaller(host)
    info = info_caller.get_info()
    assert info.db_path == ":memory:"

    test_cfg = read_test_config("tests/test-config.yaml")

    ls = aimodel_caller.get_list_aimodels()
    assert len(ls) == 0
    assert test_cfg.checkpoint_sdxl.file_path is not None
    assert test_cfg.vae_sdxl.hugging_face is not None

    aimodel_input = AIModelUserInput(
        name="new model",
        path=test_cfg.checkpoint_sdxl.file_path,
        path_type=PathType.FILE,
        variant=Variant.FP16,
        model_type=AIModelType.CHECKPOINT,
        model_base=AIModelBase.SDXL,
        tags="anime",
    )

    checkpoint_model = aimodel_caller.create_aimodel(aimodel_input)
    assert checkpoint_model.name == aimodel_input.name
    assert checkpoint_model.id is not None

    aimodel_input = AIModelUserInput(
        name="vae model",
        path=test_cfg.vae_sdxl.hugging_face,
        path_type=PathType.HUGGING_FACE,
        variant=Variant.FP16,
        model_type=AIModelType.VAE,
        model_base=AIModelBase.SDXL,
        tags="anime",
    )

    vae_model = aimodel_caller.create_aimodel(aimodel_input)
    assert vae_model.id

    engine_input = EngineUserInput(
        name="new engine",
        checkpoint_model_id=checkpoint_model.id,
        vae_model_id=vae_model.id,
        scheduler=Scheduler.EULERA,
        guidance_scale=7,
        seed=44,
        width=1024,
        height=1024,
        steps=25,
        clip_skip=2,
        pipe_type=PipeType.TXT2IMG,
    )

    engine = engine_caller.create_engine(engine_input)
    assert engine.id is not None

    gen_input = GeneratorUserInput(name="new generator", engine_id=engine.id, gpu_id=0)

    gen = generator_caller.create_generator(gen_input)
    assert gen.id is not None

    gen = generator_caller.start_generator(gen.id)
    assert gen.id is not None
    assert gen.status == GeneratorStatus.STARTING

    time.sleep(20)

    gen = generator_caller.get_generator(gen.id)
    assert gen.id is not None
    assert gen.status == GeneratorStatus.READY
    prompt = "masterpiece,1girl,waitress,(white background:1.5)"
    job_input = JobUserInput(
        generator_id=gen.id,
        images=[
            ImageUserInput(
                prompt=prompt, negative_prompt="bad quality", name="waitress1"
            )
        ],
    )

    job = job_caller.create_job(job_input)
    assert job.id is not None
    print(job)

    time.sleep(20)
    job = job_caller.get_job(job.id)
    assert job.id is not None
    assert job.status == JobStatus.FINISHED

    imgs = img_caller.get_list_images()
    assert len(imgs) == 1
    assert job.images[0].id is not None
    img = img_caller.get_image(job.images[0].id)
    assert img.id is not None
    assert img.prompt == prompt
    assert img.ready

    img_path = "/tmp/test_sdxl_image_creation.png"
    img_caller.download_image(img.id, img_path)
    assert os.path.isfile(img_path)

    _ = generator_caller.close_generator(gen.id)
    time.sleep(10)
    job_caller.delete_job(job.id)
    assert not os.path.exists(img.file_path)
    for cni in img.control_images:
        assert not os.path.exists(cni.image_file_path)

    generator_caller.delete_generator(gen.id)
    engine_caller.delete_engine(engine.id)
    aimodel_caller.delete_aimodel(checkpoint_model.id)
    aimodel_caller.delete_aimodel(vae_model.id)
