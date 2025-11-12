import multiprocessing
import time
from dataclasses import dataclass
from multiprocessing.queues import Queue
from threading import Lock, Thread

from pytsterrors import TSTError

from .process.generator import start_generator
from .schemas import GeneratorCommand, GeneratorResult, GeneratorSchema


@dataclass
class GeneratorProcess:
    generator: GeneratorSchema
    commands_queue: Queue[GeneratorCommand]
    results_queue: Queue[GeneratorResult]


class ProcessManager:
    def __init__(self):
        self._procs: dict[int, GeneratorProcess] = {}
        self._lock = Lock()  # protects concurrent writes
        self._thread = Thread(target=self._infinite_loop, daemon=True)
        self._thread.start()

    def _infinite_loop(self):
        while True:
            # Your loop logic here (e.g., periodic task, monitoring, etc.)
            print("Background thread running...")  # Replace with actual work
            time.sleep(5)  # Adjust interval as needed

    def start_generator(self, gen: GeneratorSchema):
        if gen.id is None:
            raise TSTError(
                "generator-id-is-none", "Tried to start a generator with no ID"
            )

        commandq: Queue[GeneratorCommand] = multiprocessing.Queue()
        resultq: Queue[GeneratorResult] = multiprocessing.Queue()

        p = multiprocessing.Process(
            target=start_generator,
            args=(
                gen.name,
                gen.id,
                gen.engine,
                commandq,
                resultq,
            ),
        )
        p.start()
        with self._lock:
            self._procs[gen.id] = GeneratorProcess(
                generator=gen,
                commands_queue=commandq,
                results_queue=resultq,
            )
