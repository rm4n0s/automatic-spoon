from dishka import Provider, Scope, provide

from .repositories import JobRepo


class JobRepoProvider(Provider):
    @provide(scope=Scope.APP)
    def provide_repository(self) -> JobRepo:
        return JobRepo()
