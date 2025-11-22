import httpx

from src.api.v1.info.schemas import InfoSchema


class InfoCaller:
    _host: str

    def __init__(self, host: str):
        self._host = host

    def get_info(self) -> InfoSchema:
        resp = httpx.get(self._host + "/api/v1/info")
        _ = resp.raise_for_status()
        json_data = resp.json()
        return InfoSchema.model_validate(json_data)
