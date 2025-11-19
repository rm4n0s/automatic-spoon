from pytsterrors import TSTError

from src.core.enums import AIModelStatus

from .repositories import AIModelRepo
from .schemas import AIModelSchema
from .user_inputs import AIModelUserInput


class AIModelService:
    aimodel_repo: AIModelRepo

    def __init__(self, aimodel_repo: AIModelRepo):
        self.aimodel_repo = aimodel_repo

    async def create(self, input: AIModelUserInput) -> AIModelSchema:
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

        return await self.aimodel_repo.create(data)

    async def delete(self, id: int) -> str | None:
        if await self.aimodel_repo.is_used_by_engine(id):
            raise TSTError(
                "engine-is-using-aimodel",
                f"AIModel with ID {id} can't be deleted while is being used by an engine",
                metadata={"status_code": 400},
            )
        await self.aimodel_repo.delete(id)
