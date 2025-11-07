from tortoise import fields
from tortoise.models import Model

from src.ctrls.ctrl_types.enums import JobStatus


class Job(Model):
    id = fields.IntField(primary_key=True)
    image_id = fields.IntField()
    engine_id = fields.IntField()
    status = fields.CharEnumField(enum_type=JobStatus)
    created_at = fields.DatetimeField()
    updated_at = fields.DatetimeField()
    finshed_at = fields.DatetimeField()
