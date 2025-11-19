from pytsterrors import TSTError

from src.core.enums import AIModelType
from src.db.models import AIModel, AIModelForEngine

from .schemas import AIModelSchema


class AIModelRepo:
    async def create(self, input: AIModelSchema) -> AIModelSchema:
        aimodel = await AIModel.create(**input.model_dump(exclude_unset=True))
        return AIModelSchema.model_validate(aimodel)

    async def delete(self, id: int):
        obj = await AIModel.get_or_none(id=id)
        if not obj:
            raise TSTError(
                "aimodel-not-found",
                f"AIModel with ID {id} not found",
                metadata={"status_code": 404},
            )

        await obj.delete()

    async def get_all(self) -> list[AIModelSchema]:
        aimodels = await AIModel.all()
        return [AIModelSchema.model_validate(m) for m in aimodels]

    async def get_one(self, id: int) -> AIModelSchema:
        obj = await AIModel.get_or_none(id=id)
        if not obj:
            raise TSTError(
                "aimodel-not-found",
                f"AIModel with ID {id} not found",
                metadata={"status_code": 404},
            )

        return AIModelSchema.model_validate(obj)

    async def get_one_or_none(self, id: int) -> AIModelSchema | None:
        obj = await AIModel.get_or_none(id=id)
        return AIModelSchema.model_validate(obj) if obj else None

    async def exists(self, id: int, model_type: AIModelType) -> bool:
        return await AIModel.exists(id=id, model_type=model_type)

    async def is_used_by_engine(self, id: int) -> bool:
        return await AIModelForEngine.exists(aimodel_id=id)
