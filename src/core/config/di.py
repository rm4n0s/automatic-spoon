# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from typing import Iterable

from dishka import Provider, Scope, provide

from .config import Config


class ConfigProvider(Provider):
    # <-- singleton for the whole app
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    @provide(scope=Scope.APP)
    def get_config(
        self,
    ) -> Iterable[Config]:
        yield self.config
