from dishka import Provider, Scope, provide

from .services import GPUService


class GPUServiceProvider(Provider):
    @provide(scope=Scope.REQUEST)
    def provide_service(
        self,
    ) -> GPUService:
        return GPUService()
