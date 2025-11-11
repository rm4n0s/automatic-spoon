import logging
from multiprocessing.queues import Queue

from diffusers import DiffusionPipeline

from src.api.v1.engines.schemas import (
    GeneratorCommand,
    GeneratorResult,
    EngineSchema,
)
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


class GeneratorProcess:
    _command_queue: Queue[GeneratorCommand]
    _result_queue: Queue[GeneratorResult]
    _engine: EngineSchema

    def __init__(
        self,
        engine: EngineSchema,
        commands_queue: Queue[GeneratorCommand],
        result_queue: Queue[GeneratorResult],
    ):
        self._command_queue = commands_queue
        self._result_queue = result_queue
        self._engine = engine

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
        pipe = pipe.to("cuda")
        pipe.safety_checker = None
        return pipe

    def listening(self):
        pipe = self._create_pipe()
        while True:
            cmd = self._command_queue.get()
            match cmd.command:
                case GeneratorCommandType.JOB:
                    logging.debug("received job")
                    if cmd.value is None:
                        self._result_queue.put(
                            GeneratorResult(
                                result=GeneratorResultType.ERROR, value="job was None"
                            )
                        )
                        continue

                    job = cmd.value
                    image = run_pipe(pipe, self._engine, job)
                    image.save(job.save_file_path)
                    self._result_queue.put(
                        GeneratorResult(result=GeneratorResultType.JOB, value=job)
                    )
                case GeneratorCommandType.CLOSE:
                    logging.debug("closing")
                    break

        self._result_queue.put(GeneratorResult(result=GeneratorResultType.CLOSED, value=None))


def start_generator(
    engine: EngineSchema,
    commands_queue: Queue[GeneratorCommand],
    result_queue: Queue[GeneratorResult],
):
    generator = GeneratorProcess(engine, commands_queue, result_queue)
    logging.debug(f"start generator with engine named {engine.name} and id {engine.id}")
    generator.listening()
