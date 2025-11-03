from tortoise.models import Model
from tortoise import fields

class Image(Model):
    id = fields.IntField(primary_key=True)
