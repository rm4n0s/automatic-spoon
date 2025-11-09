from src.schemas.engine_schemas import EngineSchema


class EngineRepo:
    async def create(self, input: EngineSchema) -> EngineSchema:
        raise Exception

    async def get_all(self) -> list[EngineSchema]:
        return []

    async def get_one(self, id: int) -> EngineSchema:
        raise Exception
