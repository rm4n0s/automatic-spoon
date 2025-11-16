from dishka import Provider, Scope, provide

from src.api.v1.generators.manager import ProcessManager

from .repositories import JobRepo
from .services import JobService


class JobRepoProvider(Provider):
    @provide(scope=Scope.APP)
    def provide_repository(self) -> JobRepo:
        return JobRepo()


class JobServiceProvider(Provider):
    @provide(scope=Scope.REQUEST)
    def provide_service(self, job_repo: JobRepo, manager: ProcessManager) -> JobService:
        return JobService(job_repo, manager)
