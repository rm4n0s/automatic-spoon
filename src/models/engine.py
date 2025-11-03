from tortoise.models import Model
from tortoise import fields

class Engine(Model):
    id = fields.IntField(primary_key=True)
    name = fields.TextField()