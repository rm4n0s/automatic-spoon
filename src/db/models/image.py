from tortoise import fields
from tortoise.models import Model

from src.core.enums import FileImageType
from src.db.models.common import TimestampMixin


class Image(TimestampMixin, Model):
    id = fields.IntField(primary_key=True)
    generator_id = fields.IntField()
    job_id = fields.IntField()
    ready = fields.BooleanField(default=False)
    file_path = fields.TextField()
    prompt = fields.TextField()
    negative_prompt = fields.TextField()
    seed = fields.IntField(null=True, default=None)
    guidance_scale = fields.FloatField(null=True, default=None)
    width = fields.IntField(null=True, default=None)
    height = fields.IntField(null=True, default=None)
    steps = fields.IntField(null=True, default=None)
    file_type = fields.CharEnumField(enum_type=FileImageType)
    control_guidance_start = fields.FloatField(null=True, default=None)
    control_guidance_end = fields.FloatField(null=True, default=None)


class ControlNetImage(TimestampMixin, Model):
    id = fields.IntField(primary_key=True)
    image_id = fields.IntField()
    job_id = fields.IntField()
    aimodel_id = fields.IntField(null=True, default=None)
    file_path = fields.TextField()
    controlnet_conditioning_scale = fields.FloatField()
