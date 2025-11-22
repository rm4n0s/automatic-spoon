from dishka import Provider, Scope, provide

from .repositories import ImageRepo


class ImageRepoProvider(Provider):
    @provide(scope=Scope.APP)
    def provide_repository(self) -> ImageRepo:
        return ImageRepo()
