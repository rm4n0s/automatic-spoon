from tortoise import fields
from tortoise.models import Model

from src.db.models.common import TimestampMixin
from src.schemas.enums import JobStatus


class Job(TimestampMixin, Model):
    id = fields.IntField(primary_key=True)
    image_id = fields.IntField()
    engine_id = fields.IntField()
    status = fields.CharEnumField(enum_type=JobStatus)
    finshed_at = fields.DatetimeField()
