from tortoise import fields
from tortoise.models import Model

from src.ctrls.ctrl_types.enums import (
    AIModelBase,
    AIModelStatus,
    AIModelType,
    PathType,
    Variant,
)


class AIModel(Model):
    id = fields.IntField(primary_key=True)
    name = fields.TextField()
    status = fields.CharEnumField(enum_type=AIModelStatus)
    error = fields.TextField()
    path = fields.TextField()
    trigger_pos_words = fields.TextField()
    trigger_neg_words = fields.TextField()
    path_type = fields.CharEnumField(enum_type=PathType)
    variant = fields.CharEnumField(enum_type=Variant)
    model_type = fields.CharEnumField(enum_type=AIModelType)
    model_base = fields.CharEnumField(enum_type=AIModelBase)
    tags = fields.TextField()
