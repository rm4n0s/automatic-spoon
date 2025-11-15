import logging
from multiprocessing.queues import Queue

from diffusers import DiffusionPipeline
from pytsterrors import TSTError

from src.api.v1.engines.schemas import EngineSchema
from src.core.enums import (
    GeneratorCommandType,
    GeneratorResultType,
)

from .pipe import (
    create_controlnets,
    create_pipe,
    create_vae,
    load_embeddings,
    load_loras,
    run_pipe,
    set_scheduler,
)
from .types import GeneratorCommand, GeneratorResult, ImageFinished, JobFinished


class GeneratorProcess:
    _name: str
    _generator_id: int
    _command_queue: Queue[GeneratorCommand]
    _result_queue: Queue[GeneratorResult]
    _engine: EngineSchema
    _gpu_id: int

    def __init__(
        self,
        generator_name: str,
        generator_id: int,
        gpu_id: int,
        engine: EngineSchema,
        commands_queue: Queue[GeneratorCommand],
        result_queue: Queue[GeneratorResult],
    ):
        self._name = generator_name
        self._generator_id = generator_id
        self._command_queue = commands_queue
        self._result_queue = result_queue
        self._engine = engine
        self._gpu_id = gpu_id

    def _create_pipe(self) -> DiffusionPipeline:
        vae = None
        if self._engine.vae_model is not None:
            vae = create_vae(self._engine.vae_model)

        controlnets = []
        if len(self._engine.control_net_models) > 0:
            controlnets = create_controlnets(self._engine.control_net_models)

        pipe = create_pipe(self._engine.checkpoint_model, vae, controlnets)
        if len(self._engine.lora_models) > 0:
            load_loras(pipe, self._engine.lora_models)

        if len(self._engine.embedding_models) > 0:
            load_embeddings(pipe, self._engine.embedding_models)

        set_scheduler(pipe, self._engine.scheduler)
        pipe = pipe.to("cuda:" + str(self._gpu_id))
        pipe.safety_checker = None
        return pipe

    def listening(self):
        pipe = self._create_pipe()
        self._result_queue.put(
            GeneratorResult(
                generator_name=self._name,
                generator_id=self._generator_id,
                result=GeneratorResultType.READY,
                value=None,
            )
        )
        while True:
            cmd = self._command_queue.get()
            match cmd.command:
                case GeneratorCommandType.JOB:
                    logging.debug("received job")
                    if cmd.value is None:
                        self._result_queue.put(
                            GeneratorResult(
                                generator_name=self._name,
                                generator_id=self._generator_id,
                                result=GeneratorResultType.ERROR,
                                value=TSTError(
                                    "command-value-was-none",
                                    "Command value was None on JOB command type",
                                ),
                            )
                        )
                        continue

                    job = cmd.value
                    assert job.id
                    for img in job.images:
                        assert img.id
                        run_pipe(pipe, self._engine, img)
                        self._result_queue.put(
                            GeneratorResult(
                                generator_name=self._name,
                                generator_id=self._generator_id,
                                result=GeneratorResultType.IMAGE_FINISHED,
                                value=ImageFinished(job_id=job.id, image_id=img.id),
                            )
                        )

                    self._result_queue.put(
                        GeneratorResult(
                            generator_name=self._name,
                            generator_id=self._generator_id,
                            result=GeneratorResultType.JOB_FINISHED,
                            value=JobFinished(job_id=job.id),
                        )
                    )
                case GeneratorCommandType.CLOSE:
                    logging.debug("closing")
                    break

        self._result_queue.put(
            GeneratorResult(
                generator_name=self._name,
                generator_id=self._generator_id,
                result=GeneratorResultType.CLOSED,
                value=None,
            )
        )


def start_generator(
    generator_name: str,
    generator_id: int,
    gpu_id: int,
    engine: EngineSchema,
    commands_queue: Queue[GeneratorCommand],
    result_queue: Queue[GeneratorResult],
):
    generator = GeneratorProcess(
        generator_name, generator_id, gpu_id, engine, commands_queue, result_queue
    )
    logging.debug(f"start generator with engine named {engine.name} and id {engine.id}")
    generator.listening()
