# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from dishka import Provider, Scope, provide

from .repositories import AIModelRepo
from .services import AIModelService


class AIModelRepoProvider(Provider):
    @provide(scope=Scope.APP)
    def provide_repository(self) -> AIModelRepo:
        return AIModelRepo()


class AIModelServiceProvider(Provider):
    @provide(scope=Scope.REQUEST)
    def provide_service(self, repository: AIModelRepo) -> AIModelService:
        return AIModelService(repository)
