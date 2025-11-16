from typing import Iterable

from dishka import Provider, Scope, provide

from src.api.v1.engines.repositories import EngineRepo
from src.api.v1.jobs.repositories import JobRepo

from .manager import ProcessManager
from .repositories import GeneratorRepo
from .services import GeneratorService


class GeneratorRepoProvider(Provider):
    @provide(scope=Scope.APP)
    def provide_repository(self) -> GeneratorRepo:
        return GeneratorRepo()


class GeneratorServiceProvider(Provider):
    @provide(scope=Scope.REQUEST)
    def provide_service(
        self,
        generator_repo: GeneratorRepo,
        engine_repo: EngineRepo,
        manager: ProcessManager,
    ) -> GeneratorService:
        return GeneratorService(generator_repo, engine_repo, manager)


class ProcessManagerProvider(Provider):
    scope = Scope.APP  # <-- singleton for the whole app

    @provide
    def process_manager(
        self, generator_repo: GeneratorRepo, job_repo: JobRepo
    ) -> Iterable[ProcessManager]:
        manager = ProcessManager(generator_repo, job_repo)
        yield manager
        # No cleanup: thread has no stop, runs until process exit
