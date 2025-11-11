from pytsterrors import TSTError

from src.api.v1.engines.repositories import EngineRepo
from src.api.v1.generators.repositories import GeneratorRepo
from src.api.v1.generators.schemas import GeneratorSchema, GeneratorSchemaAsUserInput
from src.core.enums import GeneratorStatus
from src.core.tags.user_errors import GENERATOR_NOT_FOUND_ERROR, WRONG_INPUT


class GeneratorService:
    generator_repo: GeneratorRepo
    engine_repo: EngineRepo

    def __init__(self, generator_repo: GeneratorRepo, engine_repo: EngineRepo):
        self.engine_repo = engine_repo
        self.generator_repo = generator_repo

    async def _validate(
        self, input: GeneratorSchemaAsUserInput
    ) -> list[dict[str, str]]:
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

    async def create(self, input: GeneratorSchemaAsUserInput) -> GeneratorSchema:
        errs = await self._validate(input)
        if len(errs) > 0:
            raise TSTError(WRONG_INPUT, "", metadata={"error_per_field": errs})

        engine = await self.engine_repo.get_one(input.engine_id)
        gs = GeneratorSchema(
            name=input.name, engine=engine, status=GeneratorStatus.STOPPED
        )
        gs = await self.generator_repo.create(gs)
        return gs

    async def start(self, id: int) -> GeneratorSchema:
        if not await self.generator_repo.exists(id):
            raise TSTError(GENERATOR_NOT_FOUND_ERROR, "")

        return await self.generator_repo.update_status(id, GeneratorStatus.STARTING)

    async def stop(self, id: int) -> GeneratorSchema:
        if not await self.generator_repo.exists(id):
            raise TSTError(GENERATOR_NOT_FOUND_ERROR, "")

        return await self.generator_repo.update_status(id, GeneratorStatus.STOPPING)
