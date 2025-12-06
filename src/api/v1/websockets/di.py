# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from typing import Iterable

from dishka import Provider, Scope, provide

from src.api.v1.generators.manager import GeneratorManager

from .services import WSEventGeneratorStreamerService


class WSEventGeneratorStreamerServiceProvider(Provider):
    scope = Scope.APP  # <-- singleton for the whole app

    @provide
    def event_streamer(
        self, manager: GeneratorManager
    ) -> Iterable[WSEventGeneratorStreamerService]:
        streamer = WSEventGeneratorStreamerService(manager)
        yield streamer
        # No cleanup: thread has no stop, runs until process exit
