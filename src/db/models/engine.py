from tortoise import fields
from tortoise.models import Model

from src.schemas.enums import (
    AIModelType,
    ControlNetPose,
    EngineStatus,
    LongPromptTechnique,
    Scheduler,
)


class Engine(Model):
    id = fields.IntField(primary_key=True)
    name = fields.TextField()
    status = fields.CharEnumField(enum_type=EngineStatus)
    long_prompt_technique = fields.CharEnumField(enum_type=LongPromptTechnique)
    controlnet = fields.CharEnumField(enum_type=ControlNetPose)
    scheduler = fields.CharEnumField(enum_type=Scheduler)
    guidance_scale = fields.FloatField()
    seed = fields.IntField()
    width = fields.IntField()
    height = fields.IntField()
    steps = fields.IntField()
    controlnet_conditioning_scale = fields.FloatField()  # Adjust 0.8-1.2
    control_guidance_start = fields.FloatField()
    control_guidance_end = fields.FloatField()
    created_at = fields.DatetimeField()
    updated_at = fields.DatetimeField()
    closed_at = fields.DatetimeField()


class AIModelForEngine(Model):
    id = fields.IntField(primary_key=True)
    engine_id = fields.IntField()
    weight = fields.FloatField()
    model_id = fields.IntField()
    model_type = fields.CharEnumField(enum_type=AIModelType)
