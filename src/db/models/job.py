from tortoise import fields
from tortoise.models import Model

from src.core.enums import AIModelType, FileImageType, JobStatus
from src.db.models.common import TimestampMixin


class Job(TimestampMixin, Model):
    id = fields.IntField(primary_key=True)
    generator_id = fields.IntField()
    status = fields.CharEnumField(enum_type=JobStatus)
    finshed_at = fields.DatetimeField()


class Image(TimestampMixin, Model):
    id = fields.IntField(primary_key=True)
    generator_id = fields.IntField()
    job_id = fields.IntField()
    prompt = fields.TextField()
    negative_prompt = fields.TextField()
    reference_image_path = fields.TextField()
    pose_image_path = fields.TextField()
    image_file_type = fields.CharEnumField(enum_type=FileImageType)


class AIModelForImage(Model):
    id = fields.IntField(primary_key=True)
    weight = fields.IntField()
    model_id = fields.IntField()
    model_type = fields.CharEnumField(enum_type=AIModelType)
