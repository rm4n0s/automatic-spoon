from pydantic import BaseModel, ConfigDict, Field

from src.api.v1.engines.schemas import EngineSchema
from src.api.v1.jobs.schemas import JobSchema
from src.core.enums import GeneratorCommandType, GeneratorResultType, GeneratorStatus


class GeneratorCommand(BaseModel):
    command: GeneratorCommandType
    value: JobSchema | None


class GeneratorResult(BaseModel):
    result: GeneratorResultType
    value: JobSchema | str | None


class GeneratorSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: int | None = Field(default=None)
    name: str
    engine: EngineSchema
    status: GeneratorStatus


class GeneratorSchemaAsUserInput(BaseModel):
    name: str
    engine_id: int
