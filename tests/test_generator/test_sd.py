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
        name="vae_model",
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

    start = time.time()
    commandq.put(EngineCommand(command=EngineCommandEnums.JOB, value=job))
    res = resultq.get()
    assert res.result == EngineResultEnums.JOB
    end = time.time()
    print(f"first job took {end - start}")
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

    start = time.time()
    commandq.put(EngineCommand(command=EngineCommandEnums.JOB, value=job))
    res = resultq.get()
    assert res.result == EngineResultEnums.JOB
    end = time.time()
    print(f"second job took {end - start}")
    assert os.path.isfile(job.save_file_path)
    print(f"finished at {job.save_file_path}")

    commandq.put(EngineCommand(command=EngineCommandEnums.CLOSE, value=None))
    res = resultq.get()
    assert res.result == EngineResultEnums.CLOSED

    logging.debug(res)
