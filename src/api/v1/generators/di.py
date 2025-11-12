from typing import Iterable

from dishka import Provider, Scope, provide

from src.api.v1.engines.repositories import EngineRepo

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
        self, generator_repo: GeneratorRepo, engine_repo: EngineRepo
    ) -> GeneratorService:
        return GeneratorService(generator_repo, engine_repo)


class ProcessManagerProvider(Provider):
    scope = Scope.APP  # <-- singleton for the whole app
    manager = ProcessManager()

    @provide
    def process_manager(self) -> Iterable[ProcessManager]:
        yield self.manager
        # No cleanup: thread has no stop, runs until process exit
