from src.db.repositories.aimodel_repo import AIModelRepo
from src.schemas.aimodel_schemas import AIModelSchema
from src.schemas.enums import AIModelStatus

from .schemas import AIModelSchemaAsUserInput


class AIModelService:
    repo: AIModelRepo

    def __init__(self, repo: AIModelRepo):
        self.repo = repo

    async def create(self, input: AIModelSchemaAsUserInput) -> AIModelSchema:
        data = AIModelSchema(
            name=input.name,
            status=AIModelStatus.READY,
            path=input.path,
            path_type=input.path_type,
            variant=input.variant,
            model_type=input.model_type,
            model_base=input.model_base,
            tags=input.tags,
            trigger_pos_words=input.trigger_pos_words,
            trigger_neg_words=input.trigger_neg_words,
        )

        return await self.repo.create(data)
