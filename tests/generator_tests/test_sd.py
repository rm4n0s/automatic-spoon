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
from src.api.v1.jobs.schemas import ImageSchema, JobSchema
from src.core.config import enable_hugging_face_envs, read_config
from src.core.enums import (
    AIModelBase,
    AIModelStatus,
    AIModelType,
    GeneratorCommandType,
    GeneratorResultType,
    GeneratorStatus,
    JobStatus,
    LongPromptTechnique,
    PathType,
    Scheduler,
    Variant,
)
from tests.utils import read_test_config


def test_sd_compel():
    logging.basicConfig(level=logging.DEBUG)
    multiprocessing.set_start_method("spawn")
    config = read_config("config.yaml")
    enable_hugging_face_envs(config)
    cfg = read_test_config("tests/test-config.yaml")
    assert cfg.checkpoint_sd.file_path is not None
    assert cfg.vae_sd.file_path is not None

    sd_model = AIModelSchema(
        id=1,
        name="sd_model",
        status=AIModelStatus.READY,
        path=cfg.checkpoint_sd.file_path,
        path_type=PathType.FILE,
        variant=Variant.FP16,
        model_type=AIModelType.CHECKPOINT,
        model_base=AIModelBase.SD,
        tags="anime",
    )

    vae_model = AIModelSchema(
        id=1,
        name="vae_model",
        status=AIModelStatus.READY,
        path=cfg.vae_sd.file_path,
        path_type=PathType.FILE,
        variant=Variant.FP16,
        model_type=AIModelType.VAE,
        model_base=AIModelBase.SD,
        tags="anime",
    )

    engine = EngineSchema(
        id=1,
        name="test sd compel",
        long_prompt_technique=LongPromptTechnique.COMPEL,
        checkpoint_model=sd_model,
        vae_model=vae_model,
        lora_models=[],
        control_net_models=[],
        embedding_models=[],
        scheduler=Scheduler.DPM2AKARRAS,
        guidance_scale=7.0,
        seed=10,
        width=512,
        height=512,
        steps=30,
        clip_skip=2,
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

    img_id = str(uuid.uuid4())
    image_king = ImageSchema(
        id=1,
        job_id=1,
        generator_id=1,
        prompt="a king, blue hair, (white background:1.5)",
        negative_prompt="bad quality",
        ready=False,
        file_path=f"/tmp/king-{img_id}.png",
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

    logging.debug("send job")
    image_queen = ImageSchema(
        id=2,
        job_id=2,
        generator_id=1,
        prompt="masterpiece, a queen, red hair, (white background:1.5)",
        negative_prompt="bad quality",
        ready=False,
        file_path=f"/tmp/queen-{img_id}.png",
    )

    image_prince = ImageSchema(
        id=3,
        job_id=2,
        generator_id=1,
        prompt="a prince, blonde hair, (white background:1.5)",
        negative_prompt="bad quality",
        ready=False,
        file_path=f"/tmp/prince-{img_id}.png",
    )

    job = JobSchema(
        id=2,
        generator_id=1,
        images=[image_queen, image_prince],
        status=JobStatus.WAITING,
    )

    start = time.time()
    commandq.put(GeneratorCommand(command=GeneratorCommandType.JOB, value=job))
    res = resultq.get()
    assert res.result == GeneratorResultType.IMAGE_FINISHED
    res = resultq.get()
    assert res.result == GeneratorResultType.IMAGE_FINISHED
    res = resultq.get()
    assert res.result == GeneratorResultType.JOB_FINISHED
    end = time.time()
    print(f"second job took {end - start}")
    for img in job.images:
        assert os.path.isfile(img.file_path)

    commandq.put(GeneratorCommand(command=GeneratorCommandType.CLOSE, value=None))
    res = resultq.get()
    assert res.result == GeneratorResultType.CLOSED

    logging.debug(res)
