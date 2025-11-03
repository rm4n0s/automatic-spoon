from tortoise.models import Model
from tortoise import fields

class Job(Model):
    id = fields.IntField(primary_key=True)
