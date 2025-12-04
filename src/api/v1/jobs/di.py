# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from dishka import Provider, Scope, provide

from src.api.v1.generators.manager import ProcessManager
from src.api.v1.generators.repositories import GeneratorRepo
from src.api.v1.images.repositories import ImageRepo

from .repositories import JobRepo
from .services import JobService


class JobRepoProvider(Provider):
    @provide(scope=Scope.APP)
    def provide_repository(self) -> JobRepo:
        return JobRepo()


class JobServiceProvider(Provider):
    @provide(scope=Scope.REQUEST)
    def provide_service(
        self,
        generator_repo: GeneratorRepo,
        job_repo: JobRepo,
        image_repo: ImageRepo,
        manager: ProcessManager,
    ) -> JobService:
        return JobService(generator_repo, job_repo, image_repo, manager)
