from src.db.repositories.aimodel_repo import AIModelRepo
from src.db.repositories.engine_repo import EngineRepo
from src.schemas.engine_schemas import EngineSchema

from .schemas import EngineSchemaAsUserInput


class EngineService:
    engine_repo: EngineRepo
    aimodel_repo: AIModelRepo

    def __init__(self, engine_repo: EngineRepo, aimodel_repo: AIModelRepo):
        self.engine_repo = engine_repo
        self.aimodel_repo = aimodel_repo

    async def create(self, input: EngineSchemaAsUserInput) -> EngineSchema | None:
        engine = EngineSchemaAsUserInput.model_validate(
            input, context={"aimodel_repo": self.aimodel_repo}
        )
        return None
