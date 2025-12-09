# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: MIT

from dishka import Provider, Scope, provide

from .services import GPUService


class GPUServiceProvider(Provider):
    @provide(scope=Scope.REQUEST)
    def provide_service(
        self,
    ) -> GPUService:
        return GPUService()
