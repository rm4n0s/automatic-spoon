# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from typing import Iterable

from dishka import Provider, Scope, provide

from src.api.v1.engines.repositories import EngineRepo
from src.api.v1.images.repositories import ImageRepo
from src.api.v1.jobs.repositories import JobRepo

from .manager import GeneratorManager
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
        job_repo: JobRepo,
        manager: GeneratorManager,
    ) -> GeneratorService:
        return GeneratorService(generator_repo, engine_repo, job_repo, manager)


class GeneratorManagerProvider(Provider):
    scope = Scope.APP  # <-- singleton for the whole app

    @provide
    def process_manager(
        self, generator_repo: GeneratorRepo, job_repo: JobRepo, image_repo: ImageRepo
    ) -> Iterable[GeneratorManager]:
        manager = GeneratorManager(generator_repo, job_repo, image_repo)
        yield manager
        # No cleanup: thread has no stop, runs until process exit
