from dishka import Provider, make_async_container
from dishka.integrations.fastapi import FastapiProvider

from src.api.v1.aimodels.di import AIModelRepoProvider, AIModelServiceProvider
from src.api.v1.engines.di import EngineRepoProvider, EngineServiceProvider
from src.api.v1.generators.di import (
    GeneratorRepoProvider,
    GeneratorServiceProvider,
    ProcessManagerProvider,
)

providers: list[Provider] = [
    AIModelRepoProvider(),
    AIModelServiceProvider(),
    EngineRepoProvider(),
    EngineServiceProvider(),
    GeneratorRepoProvider(),
    GeneratorServiceProvider(),
    ProcessManagerProvider(),
    FastapiProvider(),
]
container = make_async_container(*providers)
