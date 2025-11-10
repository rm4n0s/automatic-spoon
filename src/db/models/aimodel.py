from tortoise import fields
from tortoise.models import Model

from src.db.models.common import TimestampMixin
from src.schemas.enums import (
    AIModelBase,
    AIModelStatus,
    AIModelType,
    ControlNetType,
    PathType,
    Variant,
)


class AIModel(TimestampMixin, Model):
    id = fields.IntField(primary_key=True)
    name = fields.TextField()
    status = fields.CharEnumField(enum_type=AIModelStatus)
    error = fields.TextField(null=True)
    path = fields.TextField()
    trigger_pos_words = fields.TextField()
    trigger_neg_words = fields.TextField()
    control_net_type = fields.CharEnumField(enum_type=ControlNetType, null=True)
    path_type = fields.CharEnumField(enum_type=PathType)
    variant = fields.CharEnumField(enum_type=Variant)
    model_type = fields.CharEnumField(enum_type=AIModelType)
    model_base = fields.CharEnumField(enum_type=AIModelBase)
    tags = fields.TextField()
