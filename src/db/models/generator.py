from tortoise import fields
from tortoise.models import Model

from src.core.enums import (
    GeneratorStatus,
)
from src.db.models.common import TimestampMixin


class Generator(TimestampMixin, Model):
    id = fields.IntField(primary_key=True)
    name = fields.TextField()
    engine_id = fields.IntField()
    status = fields.CharEnumField(enum_type=GeneratorStatus)
