from tortoise import fields
from tortoise.models import Model

from src.core.enums import (
    AIModelType,
    LongPromptTechnique,
    Scheduler,
)
from src.db.models.common import TimestampMixin


class Engine(TimestampMixin, Model):
    id = fields.IntField(primary_key=True)
    name = fields.TextField()
    long_prompt_technique = fields.CharEnumField(enum_type=LongPromptTechnique)
    scheduler = fields.CharEnumField(enum_type=Scheduler)
    guidance_scale = fields.FloatField()
    seed = fields.IntField()
    width = fields.IntField()
    height = fields.IntField()
    steps = fields.IntField()
    controlnet_conditioning_scale = fields.FloatField(null=True)  # Adjust 0.8-1.2
    control_guidance_start = fields.FloatField(null=True)
    control_guidance_end = fields.FloatField(null=True)


class AIModelForEngine(Model):
    id = fields.IntField(primary_key=True)
    engine_id = fields.IntField()
    weight = fields.FloatField(null=True)
    aimodel_id = fields.IntField()
    aimodel_type = fields.CharEnumField(enum_type=AIModelType)
