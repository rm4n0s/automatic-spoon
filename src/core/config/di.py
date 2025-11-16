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
