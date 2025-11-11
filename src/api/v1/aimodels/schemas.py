from pydantic import BaseModel, ConfigDict, Field

from src.core.enums import (
    AIModelBase,
    AIModelStatus,
    AIModelType,
    ControlNetType,
    PathType,
    Variant,
)


class AIModelSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: int | None = Field(default=None)
    name: str
    status: AIModelStatus
    path: str
    path_type: PathType
    variant: Variant
    model_type: AIModelType
    model_base: AIModelBase
    tags: str
    control_net_type: ControlNetType | None = Field(default=None)
    trigger_pos_words: str | None = Field(default=None)
    trigger_neg_words: str | None = Field(default=None)
    error: str | None = Field(default=None)


class AIModelSchemaAsUserInput(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    name: str
    path: str
    path_type: PathType
    variant: Variant
    model_type: AIModelType
    model_base: AIModelBase
    control_net_type: ControlNetType | None = Field(default=None)
    tags: str = Field(default="")
    trigger_pos_words: str = Field(default="")
    trigger_neg_words: str = Field(default="")
