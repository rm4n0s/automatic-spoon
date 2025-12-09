# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: MIT

from dishka import Provider, Scope, provide

from src.api.v1.aimodels.repositories import AIModelRepo

from .repositories import EngineRepo
from .services import EngineService


class EngineRepoProvider(Provider):
    @provide(scope=Scope.APP)
    def provide_repository(self) -> EngineRepo:
        return EngineRepo()


class EngineServiceProvider(Provider):
    @provide(scope=Scope.REQUEST)
    def provide_service(
        self, engine_repo: EngineRepo, aimodel_repo: AIModelRepo
    ) -> EngineService:
        return EngineService(engine_repo, aimodel_repo)
