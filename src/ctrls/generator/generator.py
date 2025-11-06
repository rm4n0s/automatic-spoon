import multiprocessing

from diffusers import DiffusionPipeline
from pytsterrors import TSTError

from src.ctrls.ctrl_types.types import Engine, Job

from .commands import EngineCommands
from .pipe import (
    create_controlnets,
    create_pipe,
    create_vae,
    load_embeddings,
    load_loras,
    run_pipe,
    set_scheduler,
)


class Generator:
    _command_queue: multiprocessing.Queue[tuple[EngineCommands, Job]]
    _engine: Engine

    def __init__(
        self,
        engine: Engine,
        commands_queue: multiprocessing.Queue[tuple[EngineCommands, Job]],
    ):
        self._command_queue = commands_queue
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
            cmd, job = self._command_queue.get()
            match cmd:
                case EngineCommands.JOB:
                    run_pipe(pipe, self._engine, job)


def start_generator(
    engine: Engine,
    commands_queue: multiprocessing.Queue[tuple[EngineCommands, Job]],
):
    generator = Generator(engine, commands_queue)
