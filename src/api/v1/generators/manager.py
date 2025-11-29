import asyncio
import multiprocessing
from dataclasses import dataclass
from multiprocessing.queues import Queue
from threading import Lock, Thread

from src.api.v1.jobs.repositories import JobRepo
from src.core.enums import (
    GeneratorCommandType,
    GeneratorResultType,
    GeneratorStatus,
    JobStatus,
    ManagerSignalType,
)

from .process.generator import start_generator
from .process.types import GeneratorCommand, GeneratorResult, JobFinished
from .repositories import GeneratorRepo
from .schemas import GeneratorSchema


@dataclass
class GeneratorProcess:
    generator: GeneratorSchema
    commands_queue: Queue[GeneratorCommand]
    status: GeneratorStatus


@dataclass
class ManagerSignal:
    signal: ManagerSignalType
    value: int | None


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
    _result_queue: Queue[GeneratorResult]
    _signal_queue: Queue[ManagerSignal]
    _result_listener_thread: Thread
    _signal_listener_thread: Thread
    _generator_repo: GeneratorRepo
    _job_repo: JobRepo

    def __init__(self, generator_repo: GeneratorRepo, job_repo: JobRepo):
        self._generator_repo = generator_repo
        self._job_repo = job_repo
        self._procs = {}
        self._lock = Lock()
        self._result_queue = multiprocessing.Queue()
        self._signal_queue = multiprocessing.Queue()

        self._signal_listener_thread = Thread(
            target=self._listen_for_signals, daemon=True
        )
        self._signal_listener_thread.start()
        self._result_listener_thread = Thread(
            target=self._listen_for_results, daemon=True
        )
        self._result_listener_thread.start()

    def _listen_for_results(self):
        asyncio.run(_on_process_manager_init(self._generator_repo))
        while True:
            res = self._result_queue.get()
            print(
                f"generator {res.generator_name} with ID {res.generator_id} received {res.result}"
            )
            match res.result:
                case GeneratorResultType.JOB_STARTING:
                    asyncio.run(self.on_job_starting(res.generator_id))
                case GeneratorResultType.JOB_FINISHED:
                    assert isinstance(res.value, JobFinished)
                    asyncio.run(self.on_job_finished(res.generator_id, res.value))
                case GeneratorResultType.READY:
                    asyncio.run(self.on_ready(res.generator_id))
                case GeneratorResultType.CLOSED:
                    asyncio.run(self.on_closed(res.generator_id))

    def _listen_for_signals(self):
        while True:
            signal = self._signal_queue.get()
            print(f"received new signal {signal.signal}")
            match signal.signal:
                case ManagerSignalType.NEW_JOB:
                    job_id = signal.value
                    if isinstance(job_id, int):
                        asyncio.run(self.on_new_job(job_id))

    async def on_new_job(self, job_id: int):
        print(f"New Job with ID {job_id} created")
        job = await self._job_repo.get_or_none(id=job_id)
        if job is None:
            return

        print(job.__dict__)
        if job.generator_id in self._procs.keys():
            proc = self._procs[job.generator_id]
            print(proc.status)
            if proc.status == GeneratorStatus.READY:
                proc.commands_queue.put(
                    GeneratorCommand(command=GeneratorCommandType.JOB, value=job)
                )

    async def on_job_starting(self, generator_id: int):
        print("on job starting")
        with self._lock:
            self._procs[generator_id].status = GeneratorStatus.BUSY

        _ = await self._generator_repo.update_status(generator_id, GeneratorStatus.BUSY)

    async def on_job_finished(self, generator_id: int, job_finished: JobFinished):
        print("on job finished")
        with self._lock:
            self._procs[generator_id].status = GeneratorStatus.READY

        job = await self._job_repo.update_status(
            job_finished.job_id, JobStatus.FINISHED
        )
        print("finished job ", job)
        _ = await self._generator_repo.update_status(
            generator_id, GeneratorStatus.READY
        )

    async def on_ready(self, generator_id: int):
        with self._lock:
            self._procs[generator_id].status = GeneratorStatus.READY

        _ = await self._generator_repo.update_status(
            id=generator_id, status=GeneratorStatus.READY
        )

    async def on_closed(self, generator_id: int):
        print("on generator closed")
        with self._lock:
            del self._procs[generator_id]

        _ = await self._generator_repo.update_status(
            id=generator_id, status=GeneratorStatus.CLOSED
        )

    async def start_generator(self, gen: GeneratorSchema):
        if gen.id in self._procs.keys():
            return

        def _start_generator(gen: GeneratorSchema):
            assert gen.id is not None

            commandq: Queue[GeneratorCommand] = multiprocessing.Queue()

            p = multiprocessing.Process(
                target=start_generator,
                args=(
                    gen.name,
                    gen.id,
                    gen.gpu_id,
                    gen.engine,
                    commandq,
                    self._result_queue,
                ),
            )
            p.start()
            with self._lock:
                self._procs[gen.id] = GeneratorProcess(
                    generator=gen,
                    commands_queue=commandq,
                    status=GeneratorStatus.STARTING,
                )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _start_generator, gen)

    async def stop_generator(self, id: int):
        if id not in self._procs.keys():
            return

        self._procs[id].commands_queue.put(
            GeneratorCommand(command=GeneratorCommandType.CLOSE, value=None)
        )

    async def send_signal_new_job(self, job_id: int):
        self._signal_queue.put(
            ManagerSignal(signal=ManagerSignalType.NEW_JOB, value=job_id)
        )
