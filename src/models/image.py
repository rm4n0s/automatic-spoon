from tortoise import fields
from tortoise.models import Model

from src.ctrls.ctrl_types.enums import AIModelType, FileImageType


class Image(Model):
    id = fields.IntField(primary_key=True)
    engine_id = fields.IntField()
    prompt = fields.TextField()
    negative_prompt = fields.TextField()
    reference_image_path = fields.TextField()
    pose_image_path = fields.TextField()
    image_file_type = fields.CharEnumField(enum_type=FileImageType)


class AIModelForImage(Model):
    id = fields.IntField(primary_key=True)
    engine_id = fields.IntField()
    model_id = fields.IntField()
    model_type = fields.CharEnumField(enum_type=AIModelType)
