from pytsterrors import TSTError

from src.api.v1.engines.repositories import EngineRepo
from src.api.v1.jobs.repositories import JobRepo
from src.core.enums import GeneratorStatus

from .manager import ProcessManager
from .repositories import GeneratorRepo
from .schemas import GeneratorSchema
from .user_inputs import GeneratorUserInput


class GeneratorService:
    generator_repo: GeneratorRepo
    engine_repo: EngineRepo
    job_repo: JobRepo
    manager: ProcessManager

    def __init__(
        self,
        generator_repo: GeneratorRepo,
        engine_repo: EngineRepo,
        job_repo: JobRepo,
        manager: ProcessManager,
    ):
        self.engine_repo = engine_repo
        self.generator_repo = generator_repo
        self.manager = manager
        self.job_repo = job_repo

    async def _validate(self, input: GeneratorUserInput) -> list[dict[str, str]]:
        res = []
        ok = await self.engine_repo.exists(id=input.engine_id)

        if not ok:
            res.append(
                {
                    "field": "engine_id",
                    "error": f"the ID {input.engine_id} doesn't exit in aimodels",
                }
            )

        return res

    async def create(self, input: GeneratorUserInput) -> GeneratorSchema:
        errs = await self._validate(input)
        if len(errs) > 0:
            raise TSTError(
                "incorrect-input",
                "Incorrect input",
                metadata={"error_per_field": errs, "status_code": 400},
            )

        engine = await self.engine_repo.get_one(input.engine_id)
        gs = GeneratorSchema(
            name=input.name,
            engine=engine,
            status=GeneratorStatus.CLOSED,
            gpu_id=input.gpu_id,
        )
        gs = await self.generator_repo.create(gs)
        return gs

    async def start_generator(self, id: int) -> GeneratorSchema:
        if not await self.generator_repo.exists(id=id):
            raise TSTError(
                "generator-not-found",
                f"Generator with ID {id} not found",
                metadata={"status_code": 404},
            )

        gen = await self.generator_repo.update_status(id, GeneratorStatus.STARTING)
        await self.manager.start_generator(gen)
        return gen

    async def close_generator(self, id: int) -> GeneratorSchema:
        if not await self.generator_repo.exists(id=id):
            raise TSTError(
                "generator-not-found",
                f"Generator with ID {id} not found",
                metadata={"status_code": 404},
            )

        gen = await self.generator_repo.update_status(id, GeneratorStatus.CLOSING)
        assert gen.id is not None
        await self.manager.stop_generator(gen.id)
        return gen

    async def delete_generator(self, id: int):
        jobs = await self.job_repo.filter(generator_id=id)
        for job in jobs:
            assert job.id is not None
            await self.job_repo.delete(job.id)

        await self.generator_repo.delete(id)
