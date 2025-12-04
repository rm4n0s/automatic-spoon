# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from tortoise import fields
from tortoise.models import Model

from src.core.enums import (
    GeneratorStatus,
)
from src.db.models.common import TimestampMixin


class Generator(TimestampMixin, Model):
    id = fields.IntField(primary_key=True)
    name = fields.TextField()
    engine_id = fields.IntField()
    gpu_id = fields.IntField(default=0)
    status = fields.CharEnumField(enum_type=GeneratorStatus)
