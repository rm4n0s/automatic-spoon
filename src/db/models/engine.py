# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: MIT

from tortoise import fields
from tortoise.models import Model

from src.core.enums import (
    AIModelType,
    LongPromptTechnique,
    PipeType,
    Scheduler,
)
from src.db.models.common import TimestampMixin


class Engine(TimestampMixin, Model):
    id = fields.IntField(primary_key=True)
    name = fields.TextField()
    long_prompt_technique = fields.CharEnumField(
        enum_type=LongPromptTechnique, null=True
    )
    scheduler = fields.CharEnumField(enum_type=Scheduler)
    scaling_factor_enabled = fields.BooleanField(null=True)
    scheduler_config = fields.JSONField(null=True)
    guidance_scale = fields.FloatField()
    seed = fields.IntField()
    width = fields.IntField()
    height = fields.IntField()
    steps = fields.IntField()
    pipe_type = fields.CharEnumField(enum_type=PipeType)
    controlnet_conditioning_scale = fields.FloatField(null=True)
    control_guidance_start = fields.FloatField(null=True)
    control_guidance_end = fields.FloatField(null=True)
    clip_skip = fields.IntField(null=True)


class AIModelForEngine(Model):
    id = fields.IntField(primary_key=True)
    engine_id = fields.IntField()
    weight = fields.FloatField(null=True)
    aimodel_id = fields.IntField()
    model_type = fields.CharEnumField(enum_type=AIModelType)
