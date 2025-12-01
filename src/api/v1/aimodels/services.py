import os

from pytsterrors import TSTError

from src.core.enums import AIModelStatus, AIModelType, PathType

from .repositories import AIModelRepo
from .schemas import AIModelSchema
from .user_inputs import AIModelUserInput


class AIModelService:
    aimodel_repo: AIModelRepo

    def __init__(self, aimodel_repo: AIModelRepo):
        self.aimodel_repo = aimodel_repo

    async def _validate(self, input: AIModelUserInput) -> list[dict[str, str]]:
        res = []
        if input.name == "":
            res.append(
                {
                    "field": "name",
                    "error": "name can't be empty",
                }
            )

        if input.path == "":
            res.append(
                {
                    "field": "path",
                    "error": "path can't be empty",
                }
            )
        else:
            ok = await self.aimodel_repo.exists(path=input.path)
            if ok:
                res.append(
                    {
                        "field": "path",
                        "error": f"the path {input.path} is already used by another AIModel",
                    }
                )
            else:
                if input.path_type == PathType.FILE:
                    ok = os.path.isfile(input.path)
                    if not ok:
                        res.append(
                            {
                                "field": "path",
                                "error": f"the path {input.path} doesn't exist",
                            }
                        )

        if (
            input.model_type == AIModelType.CONTROLNET
            and input.control_net_type is None
        ):
            res.append(
                {
                    "field": "control_net_type",
                    "error": "when model_type is 'controlnet' then control_net_type can't be empty",
                }
            )
        return res

    async def create(self, input: AIModelUserInput) -> AIModelSchema:
        errs = await self._validate(input)
        if len(errs) > 0:
            raise TSTError(
                "incorrect-input",
                "Incorrect input",
                metadata={"error_per_field": errs, "status_code": 400},
            )
        data = AIModelSchema(
            name=input.name,
            status=AIModelStatus.READY,
            path=input.path,
            path_type=input.path_type,
            variant=input.variant,
            model_type=input.model_type,
            model_base=input.model_base,
            control_net_type=input.control_net_type,
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
