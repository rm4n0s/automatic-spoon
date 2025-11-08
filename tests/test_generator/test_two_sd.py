import logging
import multiprocessing
import os
import time
import uuid
from multiprocessing.queues import Queue

from src.api.v1.generators.generator import start_generator
from src.schemas.aimodel_schemas import AIModelSchema
from src.schemas.enums import (
    AIModelBase,
    AIModelStatus,
    AIModelType,
    EngineCommandEnums,
    EngineResultEnums,
    EngineStatus,
    LongPromptTechnique,
    PathType,
    Scheduler,
    Variant,
)
from src.schemas.types import (
    Engine,
    EngineCommand,
    EngineResult,
    Job,
)
from tests.utils import read_test_config


def test_sd_compel():
    logging.basicConfig(level=logging.DEBUG)
    multiprocessing.set_start_method("spawn")
    cfg = read_test_config("test-config.yaml")
    print(cfg)
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
        name="sd_model",
        status=AIModelStatus.READY,
        path=cfg.vae_sd.file_path,
        path_type=PathType.FILE,
        variant=Variant.FP16,
        model_type=AIModelType.VAE,
        model_base=AIModelBase.SD,
        tags="anime",
    )

    engine1 = Engine(
        id=1,
        name="test sd compel",
        status=EngineStatus.READY,
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
    )

    engine2 = Engine(
        id=2,
        name="test sd compel 2",
        status=EngineStatus.READY,
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
    )

    commandq1: Queue[EngineCommand] = multiprocessing.Queue()
    resultq1: Queue[EngineResult] = multiprocessing.Queue()

    commandq2: Queue[EngineCommand] = multiprocessing.Queue()
    resultq2: Queue[EngineResult] = multiprocessing.Queue()

    p1 = multiprocessing.Process(
        target=start_generator,
        args=(
            engine1,
            commandq1,
            resultq1,
        ),
    )
    p1.start()

    p2 = multiprocessing.Process(
        target=start_generator,
        args=(
            engine2,
            commandq2,
            resultq2,
        ),
    )
    p2.start()

    time.sleep(60)
    id = uuid.uuid4()
    logging.debug("send job")
    job1 = Job(
        id=1,
        prompt="a princess, (white background:1.5)",
        negative_prompt="bad quality",
        save_file_path=f"/tmp/{id}.png",
    )
    id = uuid.uuid4()
    logging.debug("send job to second process")
    job2 = Job(
        id=2,
        prompt="a prince, (white background:1.5)",
        negative_prompt="bad quality",
        save_file_path=f"/tmp/{id}.png",
    )
    commandq1.put(EngineCommand(command=EngineCommandEnums.JOB, value=job1))
    commandq2.put(EngineCommand(command=EngineCommandEnums.JOB, value=job2))

    res = resultq1.get()
    assert res.result == EngineResultEnums.JOB

    assert os.path.isfile(job1.save_file_path)
    print(f"finished at {job1.save_file_path}")

    res = resultq2.get()
    assert res.result == EngineResultEnums.JOB

    assert os.path.isfile(job2.save_file_path)
    print(f"finished at {job2.save_file_path}")

    commandq1.put(EngineCommand(command=EngineCommandEnums.CLOSE, value=None))
    commandq2.put(EngineCommand(command=EngineCommandEnums.CLOSE, value=None))

    res = resultq1.get()
    assert res.result == EngineResultEnums.CLOSED

    res = resultq2.get()
    assert res.result == EngineResultEnums.CLOSED
