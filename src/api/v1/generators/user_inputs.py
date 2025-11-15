from pydantic import BaseModel, Field


class GeneratorUserInput(BaseModel):
    name: str
    engine_id: int
    gpu_id: int = Field(default=0)
