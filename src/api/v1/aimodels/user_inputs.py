from pydantic import BaseModel, ConfigDict, Field

from src.core.enums import (
    AIModelBase,
    AIModelType,
    ControlNetType,
    PathType,
    Variant,
)


class AIModelUserInput(BaseModel):
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
