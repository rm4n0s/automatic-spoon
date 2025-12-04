# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

import logging
import multiprocessing
import os
import time
import uuid
from multiprocessing.queues import Queue

from src.api.v1.aimodels.schemas import AIModelSchema
from src.api.v1.engines.schemas import (
    EngineSchema,
)
from src.api.v1.generators.process.generator import start_generator
from src.api.v1.generators.process.types import (
    GeneratorCommand,
    GeneratorResult,
)
from src.api.v1.generators.schemas import GeneratorSchema
from src.api.v1.images.schemas import ControlNetImageSchema, ImageSchema
from src.api.v1.jobs.schemas import JobSchema
from src.core.config import enable_hugging_face_envs, read_config
from src.core.enums import (
    AIModelBase,
    AIModelStatus,
    AIModelType,
    ControlNetType,
    GeneratorCommandType,
    GeneratorResultType,
    GeneratorStatus,
    JobStatus,
    LongPromptTechnique,
    PathType,
    PipeType,
    Scheduler,
    Variant,
)
from src.utils import read_test_config


def test_sd_compel():
    logging.basicConfig(level=logging.DEBUG)
    multiprocessing.set_start_method("spawn")
    config = read_config("config.yaml")
    enable_hugging_face_envs(config)
    cfg = read_test_config("tests/test-config.yaml")
    assert cfg.checkpoint_sdxl.file_path is not None
    assert cfg.vae_sdxl.file_path is not None
    assert cfg.canny_sdxl.file_path is not None
    assert cfg.canny_sdxl.low_threshold is not None
    assert cfg.canny_sdxl.high_threshold is not None

    sd_model = AIModelSchema(
        id=1,
        name="sdxl model",
        status=AIModelStatus.READY,
        path=cfg.checkpoint_sdxl.file_path,
        path_type=PathType.FILE,
        variant=Variant.FP16,
        model_type=AIModelType.CHECKPOINT,
        model_base=AIModelBase.SDXL,
        tags="anime",
    )

    canny_pose_model = AIModelSchema(
        id=2,
        name="canny pose model",
        status=AIModelStatus.READY,
        path=cfg.canny_sdxl.file_path,
        path_type=PathType.FILE,
        variant=Variant.FP16,
        model_type=AIModelType.CONTROLNET,
        control_net_type=ControlNetType.CANNY,
        model_base=AIModelBase.SDXL,
        tags="anime",
    )

    vae_model = AIModelSchema(
        id=3,
        name="vae_model",
        status=AIModelStatus.READY,
        path=cfg.vae_sdxl.file_path,
        path_type=PathType.FILE,
        variant=Variant.FP16,
        model_type=AIModelType.VAE,
        model_base=AIModelBase.SDXL,
        tags="anime",
    )

    engine = EngineSchema(
        id=1,
        name="test sdxl compel",
        long_prompt_technique=LongPromptTechnique.COMPEL,
        checkpoint_model=sd_model,
        vae_model=vae_model,
        lora_models=[],
        control_net_models=[canny_pose_model],
        embedding_models=[],
        scheduler=Scheduler.EULERA,
        guidance_scale=7.0,
        seed=223123135,
        width=1024,
        height=1024,
        steps=25,
        pipe_type=PipeType.TXT2IMG,
        control_guidance_start=0.0,
        control_guidance_end=0.8,
    )

    gen = GeneratorSchema(
        id=1, name="simple", gpu_id=0, engine=engine, status=GeneratorStatus.READY
    )

    commandq: Queue[GeneratorCommand] = multiprocessing.Queue()
    resultq: Queue[GeneratorResult] = multiprocessing.Queue()

    p = multiprocessing.Process(
        target=start_generator,
        args=(
            gen.name,
            gen.id,
            gen.gpu_id,
            gen.engine,
            commandq,
            resultq,
        ),
    )
    p.start()

    time.sleep(6)

    res = resultq.get()
    assert res.result == GeneratorResultType.READY

    pose_img_ref = ControlNetImageSchema(
        aimodel=None,
        image_file_path="tests/test_data/reference-pose-img.png",
        controlnet_conditioning_scale=0.9,
        canny_low_threshold=cfg.canny_sdxl.low_threshold,
        canny_high_threshold=cfg.canny_sdxl.high_threshold,
    )
    img_id = str(uuid.uuid4())
    image_king = ImageSchema(
        id=1,
        job_id=1,
        generator_id=1,
        prompt="(masterpiece:1.2), (best quality:1.1),1boy, old wizard, standing up, blue hair, (white background:1.5)",
        negative_prompt="(worst quality:1.5), (low quality:1.5), blurry, deformed, extra limbs",
        ready=False,
        file_path=f"/tmp/wizard-{img_id}.png",
        control_images=[pose_img_ref],
    )
    logging.debug("send job")
    job = JobSchema(id=1, generator_id=1, images=[image_king], status=JobStatus.WAITING)

    start = time.time()
    commandq.put(GeneratorCommand(command=GeneratorCommandType.JOB, value=job))
    res = resultq.get()
    assert res.result == GeneratorResultType.IMAGE_FINISHED
    res = resultq.get()
    assert res.result == GeneratorResultType.JOB_FINISHED
    end = time.time()
    print(f"first job took {end - start}")
    assert os.path.isfile(job.images[0].file_path)

    commandq.put(GeneratorCommand(command=GeneratorCommandType.CLOSE, value=None))
    res = resultq.get()
    assert res.result == GeneratorResultType.CLOSED

    logging.debug(res)
