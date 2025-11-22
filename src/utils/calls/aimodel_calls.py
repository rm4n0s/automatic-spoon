import httpx

from src.api.v1.aimodels.schemas import AIModelSchema
from src.api.v1.aimodels.user_inputs import AIModelUserInput


class AIModelCaller:
    _host: str

    def __init__(self, host: str):
        self._host = host

    def get_list_aimodels(self) -> list[AIModelSchema]:
        resp = httpx.get(self._host + "/api/v1/aimodels")
        _ = resp.raise_for_status()
        json_data = resp.json()
        return [AIModelSchema(**item) for item in json_data]

    def create_aimodel(self, input: AIModelUserInput) -> AIModelSchema:
        resp = httpx.post(self._host + "/api/v1/aimodels", json=input.model_dump())
        _ = resp.raise_for_status()
        json_data = resp.json()
        return AIModelSchema.model_validate(json_data)
