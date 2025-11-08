from pytsterrors import TSTError

from src.core.tags.user_errors import AIMODEL_NOT_FOUND
from src.db.models import AIModel
from src.schemas.aimodel_schemas import AIModelSchema


class AIModelRepo:
    async def create(self, input: AIModelSchema) -> AIModelSchema | None:
        aimodel = await AIModel.create(**input.model_dump(exclude_unset=True))
        return AIModelSchema.model_validate(aimodel) if aimodel else None

    async def get_all(self) -> list[AIModelSchema]:
        aimodels = await AIModel.all()
        return [AIModelSchema.model_validate(m) for m in aimodels]

    async def get_one(self, id: int) -> AIModelSchema:
        obj = await AIModel.get_or_none(id=id)
        if not obj:
            raise TSTError(AIMODEL_NOT_FOUND, f"AIModel with ID {id} not found")

        return AIModelSchema.model_validate(obj)
