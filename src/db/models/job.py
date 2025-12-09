# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: MIT

from tortoise import fields
from tortoise.models import Model

from src.core.enums import JobStatus
from src.db.models.common import TimestampMixin


class Job(TimestampMixin, Model):
    id = fields.IntField(primary_key=True)
    generator_id = fields.IntField()
    status = fields.CharEnumField(enum_type=JobStatus, default=JobStatus.WAITING)
    ip_adapter_config = fields.JSONField(null=True, default=None)
    finshed_at = fields.DatetimeField(null=True, default=None)
