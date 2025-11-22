from pydantic import BaseModel


class GPUSchema(BaseModel):
    id: int
    name: str
    total_vram_gb: float
