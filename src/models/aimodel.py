from tortoise.models import Model
from tortoise import fields

class AIModel(Model):
    id = fields.IntField(primary_key=True)
    name = fields.TextField()