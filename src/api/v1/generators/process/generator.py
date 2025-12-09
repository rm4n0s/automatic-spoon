# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: MIT

import logging
from multiprocessing.queues import Queue

import torch
from diffusers import DiffusionPipeline
from PIL import Image
from pytsterrors import TSTError

from src.api.v1.engines.schemas import EngineSchema
from src.core.enums import (
    GeneratorCommandType,
    GeneratorEventType,
)

from .pipe import (
    create_controlnets,
    create_pipe,
    create_vae,
    load_embeddings,
    load_ip_adapter,
    load_loras,
    run_pipe,
    set_ip_adapter_scale,
    set_scheduler,
    unload_ip_adapter,
)
from .types import GeneratorCommand, GeneratorEvent, ImageFinished, JobFinished


class GeneratorProcess:
    _name: str
    _generator_id: int
    _command_queue: Queue[GeneratorCommand]
    _event_queue: Queue[GeneratorEvent]
    _engine: EngineSchema
    _gpu_id: int

    def __init__(
        self,
        generator_name: str,
        generator_id: int,
        gpu_id: int,
        engine: EngineSchema,
        commands_queue: Queue[GeneratorCommand],
        event_queue: Queue[GeneratorEvent],
    ):
        self._name = generator_name
        self._generator_id = generator_id
        self._command_queue = commands_queue
        self._event_queue = event_queue
        self._engine = engine
        self._gpu_id = gpu_id

    def _create_pipe(self) -> DiffusionPipeline:
        vae = None
        if self._engine.vae_model is not None:
            vae = create_vae(self._engine.vae_model)

        controlnets = []
        if len(self._engine.control_net_models) > 0:
            controlnets = create_controlnets(self._engine.control_net_models)

        pipe = create_pipe(self._engine, vae, controlnets)
        if self._engine.clip_skip is not None:
            clip_skip = self._engine.clip_skip
            pipe.text_encoder.text_model.encoder.layers = (
                pipe.text_encoder.text_model.encoder.layers[: -(clip_skip - 1)]
            )

        if self._engine.scaling_factor_enabled is not None and vae is not None:
            if self._engine.scaling_factor_enabled:
                pipe.vae.config.scaling_factor = torch.tensor(
                    pipe.vae.config.scaling_factor,
                    dtype=pipe.vae.dtype,
                    device=pipe.device,
                )

        if len(self._engine.lora_models) > 0:
            load_loras(pipe, self._engine.lora_models)

        if len(self._engine.embedding_models) > 0:
            load_embeddings(pipe, self._engine.embedding_models)

        scheduler_config = {}
        if self._engine.scheduler_config is not None:
            scheduler_config = self._engine.scheduler_config

        print(self._engine.scheduler, scheduler_config)
        set_scheduler(pipe, self._engine.scheduler, scheduler_config)
        pipe = pipe.to("cuda:" + str(self._gpu_id))
        pipe.safety_checker = None

        return pipe

    def listening(self):
        pipe = self._create_pipe()
        self._event_queue.put(
            GeneratorEvent(
                generator_name=self._name,
                generator_id=self._generator_id,
                event=GeneratorEventType.READY,
                value=None,
            )
        )
        while True:
            cmd = self._command_queue.get()
            print(f"Generator {self._generator_id} received new command {cmd.command}")
            match cmd.command:
                case GeneratorCommandType.JOB:
                    logging.debug("received job")
                    if cmd.value is None:
                        self._event_queue.put(
                            GeneratorEvent(
                                generator_name=self._name,
                                generator_id=self._generator_id,
                                event=GeneratorEventType.ERROR,
                                value=TSTError(
                                    "command-value-was-none",
                                    "Command value was None on JOB command type",
                                ),
                            )
                        )
                        continue

                    job = cmd.value
                    assert job.id

                    prv_img_path = None
                    ip_adapter_image = None
                    ip_adapter_loaded = False
                    for img in job.images:
                        assert img.id

                        if (
                            prv_img_path is not None
                            and ip_adapter_loaded is False
                            and job.ip_adapter_config is not None
                        ):
                            print("ip adapter config", job.ip_adapter_config)
                            load_ip_adapter(pipe, job)
                            set_ip_adapter_scale(pipe, job)
                            ip_adapter_loaded = True
                            ip_adapter_image = Image.open(prv_img_path).convert("RGB")

                        run_pipe(pipe, self._engine, img, ip_adapter_image)
                        print(
                            "VRAM used size:",
                            torch.cuda.max_memory_allocated() / 1024**3,
                        )
                        prv_img_path = img.file_path

                        self._event_queue.put(
                            GeneratorEvent(
                                generator_name=self._name,
                                generator_id=self._generator_id,
                                event=GeneratorEventType.IMAGE_FINISHED,
                                value=ImageFinished(job_id=job.id, image_id=img.id),
                            )
                        )

                    if ip_adapter_loaded:
                        unload_ip_adapter(pipe, job)

                    self._event_queue.put(
                        GeneratorEvent(
                            generator_name=self._name,
                            generator_id=self._generator_id,
                            event=GeneratorEventType.JOB_FINISHED,
                            value=JobFinished(job_id=job.id),
                        )
                    )
                case GeneratorCommandType.CLOSE:
                    logging.debug("closing")
                    break

        self._event_queue.put(
            GeneratorEvent(
                generator_name=self._name,
                generator_id=self._generator_id,
                event=GeneratorEventType.CLOSED,
                value=None,
            )
        )


def start_generator(
    generator_name: str,
    generator_id: int,
    gpu_id: int,
    engine: EngineSchema,
    commands_queue: Queue[GeneratorCommand],
    result_queue: Queue[GeneratorEvent],
):
    generator = GeneratorProcess(
        generator_name, generator_id, gpu_id, engine, commands_queue, result_queue
    )
    logging.debug(f"start generator with engine named {engine.name} and id {engine.id}")
    generator.listening()
