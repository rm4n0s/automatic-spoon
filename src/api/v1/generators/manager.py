import asyncio
import multiprocessing
from dataclasses import dataclass
from multiprocessing.queues import Queue
from threading import Lock, Thread

from pytsterrors import TSTError

from src.core.enums import GeneratorCommandType, GeneratorResultType, GeneratorStatus

from .process.generator import start_generator
from .repositories import GeneratorRepo
from .schemas import GeneratorCommand, GeneratorResult, GeneratorSchema


@dataclass
class GeneratorProcess:
    generator: GeneratorSchema
    commands_queue: Queue[GeneratorCommand]


async def _on_process_manager_init(generator_repo: GeneratorRepo):
    gens = await generator_repo.get_all()
    for gen in gens:
        if gen.status != GeneratorStatus.CLOSED:
            assert gen.id is not None
            _ = await generator_repo.update_status(
                id=gen.id, status=GeneratorStatus.CLOSED
            )


class ProcessManager:
    _procs: dict[int, GeneratorProcess]
    _lock: Lock
    _resultq: Queue[GeneratorResult]
    _thread: Thread
    _generator_repo: GeneratorRepo

    def __init__(self, generator_repo: GeneratorRepo):
        self._generator_repo = generator_repo
        self._procs = {}
        self._lock = Lock()
        self._resultq = multiprocessing.Queue()

        self._thread = Thread(target=self._infinite_loop, daemon=True)
        self._thread.start()

    def _infinite_loop(self):
        asyncio.run(_on_process_manager_init(self._generator_repo))
        while True:
            res = self._resultq.get()
            print(
                f"generator {res.generator_name} with ID {res.generator_id} received {res.result}"
            )
            match res.result:
                case GeneratorResultType.READY:
                    asyncio.run(self.on_ready(res.generator_id))
                case GeneratorResultType.CLOSED:
                    asyncio.run(self.on_closed(res.generator_id))

    async def on_ready(self, id: int):
        _ = await self._generator_repo.update_status(
            id=id, status=GeneratorStatus.READY
        )

    async def on_closed(self, id: int):
        _ = await self._generator_repo.update_status(
            id=id, status=GeneratorStatus.CLOSED
        )

    async def start_generator(self, gen: GeneratorSchema):
        if gen.id in self._procs.keys():
            return

        def _start_generator(gen: GeneratorSchema):
            if gen.id is None:
                raise TSTError(
                    "generator-id-is-none", "Tried to start a generator with no ID"
                )

            commandq: Queue[GeneratorCommand] = multiprocessing.Queue()

            p = multiprocessing.Process(
                target=start_generator,
                args=(
                    gen.name,
                    gen.id,
                    gen.engine,
                    commandq,
                    self._resultq,
                ),
            )
            p.start()
            with self._lock:
                self._procs[gen.id] = GeneratorProcess(
                    generator=gen,
                    commands_queue=commandq,
                )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _start_generator, gen)

    async def stop_generator(self, id: int):
        if id not in self._procs.keys():
            return

        self._procs[id].commands_queue.put(
            GeneratorCommand(command=GeneratorCommandType.CLOSE, value=None)
        )
