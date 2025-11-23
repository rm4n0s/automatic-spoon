import logging
import multiprocessing
import os
import time
import uuid
from multiprocessing.queues import Queue

from src.api.v1.aimodels.schemas import AIModelSchema
from src.api.v1.engines.schemas import (
    EngineSchema,
    LoraAndWeight,
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
    Scheduler,
    Variant,
)
from src.utils import read_test_config


def test_sdxl_lora_embedding():
    logging.basicConfig(level=logging.DEBUG)
    multiprocessing.set_start_method("spawn")
    config = read_config("config.yaml")
    enable_hugging_face_envs(config)
    cfg = read_test_config("tests/test-config.yaml")
    assert cfg.checkpoint_sdxl.file_path is not None
    assert cfg.vae_sdxl.file_path is not None
    assert cfg.openpose_sdxl.file_path is not None
    assert cfg.midas_sdxl.file_path is not None

    sd_model = AIModelSchema(
        id=1,
        name="sd_model",
        status=AIModelStatus.READY,
        path=cfg.checkpoint_sdxl.file_path,
        path_type=PathType.FILE,
        variant=Variant.FP16,
        model_type=AIModelType.CHECKPOINT,
        model_base=AIModelBase.SDXL,
        tags="anime",
    )

    # openpose_pose_model = AIModelSchema(
    #     id=2,
    #     name="openpose pose model",
    #     status=AIModelStatus.READY,
    #     path=cfg.openpose_sdxl.file_path,
    #     path_type=PathType.FILE,
    #     variant=Variant.FP16,
    #     model_type=AIModelType.CONTROLNET,
    #     control_net_type=ControlNetType.OPENPOSE,
    #     model_base=AIModelBase.SDXL,
    #     tags="anime",
    # )

    midas_pose_model = AIModelSchema(
        id=2,
        name="midas pose model",
        status=AIModelStatus.READY,
        path=cfg.midas_sdxl.file_path,
        path_type=PathType.FILE,
        variant=Variant.FP16,
        model_type=AIModelType.CONTROLNET,
        control_net_type=ControlNetType.MIDAS,
        model_base=AIModelBase.SDXL,
        tags="anime",
    )

    vae_model = AIModelSchema(
        id=1,
        name="vae_model",
        status=AIModelStatus.READY,
        path=cfg.vae_sdxl.file_path,
        path_type=PathType.FILE,
        variant=Variant.FP16,
        model_type=AIModelType.VAE,
        model_base=AIModelBase.SDXL,
        tags="anime",
    )

    input_lora: list[LoraAndWeight] = []
    lora_triggers_pos = ""
    for i, lora in enumerate(cfg.loras_sdxl):
        assert lora.file_path is not None
        lora_model = AIModelSchema(
            id=1,
            name="lora_model_" + str(i),
            status=AIModelStatus.READY,
            path=lora.file_path,
            path_type=PathType.FILE,
            variant=Variant.FP16,
            model_type=AIModelType.LORA,
            model_base=AIModelBase.SDXL,
            tags="anime",
        )
        assert lora.trigger_pos is not None
        lora_triggers_pos += lora.trigger_pos + ","

        assert lora.weight is not None
        lw = LoraAndWeight(aimodel=lora_model, weight=lora.weight)
        input_lora.append(lw)

    input_embeds: list[AIModelSchema] = []
    embed_trigger_neg = ""
    embed_trigger_pos = ""
    for i, embed in enumerate(cfg.embeddings_sdxl):
        assert embed.file_path is not None
        trigger_pos = embed.trigger_pos or ""
        trigger_neg = embed.trigger_neg or ""
        if trigger_neg:
            embed_trigger_neg += trigger_neg + ","

        if trigger_pos:
            embed_trigger_pos += trigger_pos + ","
        embed_model = AIModelSchema(
            id=1,
            name="embed_model_" + str(i),
            status=AIModelStatus.READY,
            path=embed.file_path,
            path_type=PathType.FILE,
            variant=Variant.FP16,
            model_type=AIModelType.EMBEDDING,
            model_base=AIModelBase.SDXL,
            trigger_pos_words=trigger_pos,
            trigger_neg_words=trigger_neg,
            tags="",
        )

        # embed_trigger_neg += embed.trigger_pos + ","
        input_embeds.append(embed_model)

    engine = EngineSchema(
        id=1,
        name="test sdxl compel",
        long_prompt_technique=LongPromptTechnique.COMPEL,
        checkpoint_model=sd_model,
        vae_model=vae_model,
        lora_models=input_lora,
        control_net_models=[midas_pose_model],
        embedding_models=input_embeds,
        scheduler=Scheduler.EULERA,
        guidance_scale=7.0,
        seed=223123135,
        width=1024,
        height=1024,
        steps=50,
        clip_skip=2,
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
    )

    img_id = str(uuid.uuid4())
    image_king = ImageSchema(
        id=1,
        job_id=1,
        generator_id=1,
        prompt="masterpiece,best quality,1boy, 30 years old, king, blue hair, (white background:1.5),"
        + embed_trigger_pos
        + lora_triggers_pos,
        negative_prompt=embed_trigger_neg,
        control_images=[pose_img_ref],
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
        prompt="masterpiece,best quality,1girl, a queen, red hair,sits, full body view (white background:1.5),"
        + embed_trigger_pos
        + lora_triggers_pos,
        negative_prompt=embed_trigger_neg,
        ready=False,
        file_path=f"/tmp/queen-{img_id}.png",
        control_images=[pose_img_ref],
    )

    image_prince = ImageSchema(
        id=3,
        job_id=2,
        generator_id=1,
        prompt="masterpiece,best quality,1 boy, a prince, blonde hair, (white background:1.5),"
        + embed_trigger_pos
        + lora_triggers_pos,
        negative_prompt=embed_trigger_neg,
        ready=False,
        file_path=f"/tmp/prince-{img_id}.png",
        control_images=[pose_img_ref],
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
