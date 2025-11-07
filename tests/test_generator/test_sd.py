import logging
import multiprocessing
import os
import time
import uuid
from multiprocessing.queues import Queue

from src.ctrls.ctrl_types import (
    AIModelBase,
    AIModelStatus,
    AIModelType,
    Engine,
    EngineCommand,
    EngineCommandEnums,
    EngineResult,
    EngineResultEnums,
    EngineStatus,
    Job,
    LongPromptTechnique,
    Model,
    PathType,
    Scheduler,
    Variant,
)
from src.ctrls.generator.generator import start_generator
from tests.utils import read_test_config


def test_sd_compel_rembg():
    logging.basicConfig(level=logging.DEBUG)
    multiprocessing.set_start_method("spawn")
    cfg = read_test_config("test-config.yaml")
    print(cfg)
    assert cfg.checkpoint_sd.file_path is not None
    assert cfg.vae_sd.file_path is not None

    sd_model = Model(
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

    vae_model = Model(
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

    engine = Engine(
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

    commandq: Queue[EngineCommand] = multiprocessing.Queue()
    resultq: Queue[EngineResult] = multiprocessing.Queue()

    p = multiprocessing.Process(
        target=start_generator,
        args=(
            engine,
            commandq,
            resultq,
        ),
    )
    p.start()

    time.sleep(60)
    id = uuid.uuid4()
    logging.debug("send job")
    job = Job(
        id=1,
        prompt="a king, (white background:1.5)",
        negative_prompt="bad quality",
        save_file_path=f"/tmp/{id}.png",
    )

    commandq.put(EngineCommand(EngineCommandEnums.JOB, job))

    res = resultq.get()
    assert res.result == EngineResultEnums.JOB

    assert os.path.isfile(job.save_file_path)
    print(f"finished at {job.save_file_path}")

    id = uuid.uuid4()
    logging.debug("send job")
    job = Job(
        id=1,
        prompt="a queen, (white background:1.5)",
        negative_prompt="bad quality",
        save_file_path=f"/tmp/{id}.png",
    )

    commandq.put(EngineCommand(EngineCommandEnums.JOB, job))

    res = resultq.get()
    assert res.result == EngineResultEnums.JOB

    assert os.path.isfile(job.save_file_path)
    print(f"finished at {job.save_file_path}")

    commandq.put(EngineCommand(EngineCommandEnums.CLOSE, None))

    res = resultq.get()
    assert res.result == EngineResultEnums.CLOSED

    logging.debug(res)
