# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: MIT

from dishka import Provider, Scope, provide

from .repositories import ImageRepo


class ImageRepoProvider(Provider):
    @provide(scope=Scope.APP)
    def provide_repository(self) -> ImageRepo:
        return ImageRepo()
