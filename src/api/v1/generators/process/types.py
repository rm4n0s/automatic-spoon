from dataclasses import dataclass

from pytsterrors import TSTError

from src.api.v1.jobs.schemas import JobSchema
from src.core.enums import GeneratorCommandType, GeneratorResultType


@dataclass
class GeneratorCommand:
    command: GeneratorCommandType
    value: JobSchema | None


@dataclass
class JobFinished:
    job_id: int


@dataclass
class ImageFinished:
    job_id: int
    image_id: int


@dataclass
class GeneratorResult:
    generator_name: str
    generator_id: int
    result: GeneratorResultType
    value: JobFinished | ImageFinished | TSTError | None
