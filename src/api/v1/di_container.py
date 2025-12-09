# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: MIT

from dishka import Provider, make_async_container
from dishka.integrations.fastapi import FastapiProvider

from src.api.v1.aimodels.di import AIModelRepoProvider, AIModelServiceProvider
from src.api.v1.engines.di import EngineRepoProvider, EngineServiceProvider
from src.api.v1.generators.di import (
    GeneratorManagerProvider,
    GeneratorRepoProvider,
    GeneratorServiceProvider,
)
from src.api.v1.gpus.di import GPUServiceProvider
from src.api.v1.images.di import ImageRepoProvider
from src.api.v1.jobs.di import JobRepoProvider, JobServiceProvider
from src.api.v1.websockets.di import WSEventGeneratorStreamerServiceProvider
from src.core.config import Config
from src.core.config.di import ConfigProvider


def create_dishka_container(config: Config):
    providers: list[Provider] = [
        ConfigProvider(config),
        AIModelRepoProvider(),
        AIModelServiceProvider(),
        EngineRepoProvider(),
        EngineServiceProvider(),
        GeneratorRepoProvider(),
        GeneratorServiceProvider(),
        GeneratorManagerProvider(),
        FastapiProvider(),
        JobRepoProvider(),
        JobServiceProvider(),
        ImageRepoProvider(),
        GPUServiceProvider(),
        WSEventGeneratorStreamerServiceProvider(),
    ]
    container = make_async_container(*providers)

    return container
