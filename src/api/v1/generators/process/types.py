# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

import enum
import json
from dataclasses import asdict, dataclass
from typing import Any

from pytsterrors import TSTError

from src.api.v1.jobs.schemas import JobSchema
from src.core.enums import GeneratorCommandType, GeneratorEventType


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
class GeneratorEvent:
    generator_name: str
    generator_id: int
    event: GeneratorEventType
    value: JobFinished | ImageFinished | TSTError | None


# Custom JSON encoder to handle Enums
class EnumEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, enum.Enum):
            return o.value
        return super().default(o)


# Function to serialize GeneratorEvent to JSON
def generator_event_to_json(event: GeneratorEvent) -> str:
    print(event)
    data = asdict(event)
    # Convert value to dict if not None
    if data["value"] is not None:
        if not isinstance(data["value"], dict):
            data["value"] = asdict(data["value"])
    # The result is already an Enum, which will be handled by the encoder
    return json.dumps(data, cls=EnumEncoder)


# Function to deserialize JSON back to GeneratorEvent
def json_to_generator_event(json_str: str) -> GeneratorEvent:
    data = json.loads(json_str)
    event_str = data["event"]
    event = next((r for r in GeneratorEventType if r.value == event_str), None)
    if event is None:
        raise ValueError(f"Invalid GeneratorEventType: {event_str}")

    # Reconstruct value based on result (using it as a discriminator)
    value_data = data["value"]
    value = None
    if event == GeneratorEventType.JOB_FINISHED:
        if value_data is not None:
            value = JobFinished(**value_data)
    elif event == GeneratorEventType.IMAGE_FINISHED:
        if value_data is not None:
            value = ImageFinished(**value_data)
    elif event == GeneratorEventType.ERROR:
        if value_data is not None:
            value = TSTError(**value_data)
    else:
        raise ValueError(f"Unhandled GeneratorEventType: {event}")

    # Create the GeneratorEvent
    return GeneratorEvent(
        generator_name=data["generator_name"],
        generator_id=data["generator_id"],
        event=event,
        value=value,
    )
