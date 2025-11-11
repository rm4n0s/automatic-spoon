from tortoise import fields
from tortoise.models import Model

from src.core.enums import JobStatus
from src.db.models.common import TimestampMixin


class Job(TimestampMixin, Model):
    id = fields.IntField(primary_key=True)
    image_id = fields.IntField()
    engine_id = fields.IntField()
    status = fields.CharEnumField(enum_type=JobStatus)
    finshed_at = fields.DatetimeField()
