from pytsterrors import TSTError

from src.core.tags.user_errors import AIMODEL_NOT_FOUND_ERROR
from src.db.models import AIModel
from src.schemas.aimodel_schemas import AIModelSchema
from src.schemas.enums import AIModelType


class AIModelRepo:
    async def create(self, input: AIModelSchema) -> AIModelSchema:
        aimodel = await AIModel.create(**input.model_dump(exclude_unset=True))
        return AIModelSchema.model_validate(aimodel)

    async def get_all(self) -> list[AIModelSchema]:
        aimodels = await AIModel.all()
        return [AIModelSchema.model_validate(m) for m in aimodels]

    async def get_one(self, id: int) -> AIModelSchema:
        obj = await AIModel.get_or_none(id=id)
        if not obj:
            raise TSTError(AIMODEL_NOT_FOUND_ERROR, f"AIModel with ID {id} not found")

        return AIModelSchema.model_validate(obj)

    async def get_one_or_none(self, id: int) -> AIModelSchema | None:
        obj = await AIModel.get_or_none(id=id)
        return AIModelSchema.model_validate(obj) if obj else None

    async def exists(self, id: int, model_type: AIModelType) -> bool:
        return await AIModel.exists(id=id, model_type=model_type)
