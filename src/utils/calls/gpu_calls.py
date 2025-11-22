import httpx

from src.api.v1.gpus.schemas import GPUSchema


class GPUCaller:
    _host: str

    def __init__(self, host: str):
        self._host = host

    def get_list_gpus(self) -> list[GPUSchema]:
        resp = httpx.get(self._host + "/api/v1/gpus")
        _ = resp.raise_for_status()
        json_data = resp.json()
        return [GPUSchema(**item) for item in json_data]
