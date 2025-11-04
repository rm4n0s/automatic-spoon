from tortoise.models import Model
from tortoise import fields

from .enums import JobStatus

class Job(Model):
    id = fields.IntField(primary_key=True)
    image_id = fields.IntField()
    engine_id = fields.IntField()
    status = fields.CharEnumField(enum_type=JobStatus)
    created_at = fields.DateField()
    updated_at = fields.DateField()
    finshed_at = fields.DateField()
