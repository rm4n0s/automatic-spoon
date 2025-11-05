from .enums import AIModelBase, AIModelStatus, AIModelType, Variant, PathType
from tortoise.models import Model
from tortoise import fields


class AIModel(Model):
    id = fields.IntField(primary_key=True)
    name = fields.TextField()
    status = fields.CharEnumField(enum_type=AIModelStatus)
    error = fields.TextField()
    path = fields.TextField()
    path_type = fields.CharEnumField(enum_type=PathType)
    variant = fields.CharEnumField(enum_type=Variant)
    model_type = fields.CharEnumField(enum_type=AIModelType)
    model_base = fields.CharEnumField(enum_type=AIModelBase)
    tags = fields.TextField()
