from enum import Enum 
from src import models
from dataclasses import dataclass

class EngineCommands(str, Enum):
    JOB = "job"


@dataclass 
class StartCommand():
    engine: models.Engine
    aimodels: list[models.AIModel]


@dataclass 
class JobCommand():
    job: models.Job
    aimodels: list[models.AIModel]