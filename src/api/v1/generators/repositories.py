# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

from typing import Any

from pytsterrors import TSTError
from tortoise.expressions import Q

from src.api.v1.engines.repositories import serialize_engine
from src.core.enums import GeneratorStatus
from src.db.models import Engine, Generator

from .schemas import GeneratorSchema


class GeneratorRepo:
    async def create(self, input: GeneratorSchema) -> GeneratorSchema:
        g = await Generator.create(
            engine_id=input.engine.id,
            status=input.status,
            name=input.name,
            gpu_id=input.gpu_id,
        )
        input.id = g.id
        return input

    async def get_one(self, id: int) -> GeneratorSchema:
        g = await Generator.get_or_none(id=id)
        if not g:
            raise TSTError(
                "generator-is-not-found",
                f"Generator with ID {id} not found",
                metadata={"status_code": 404},
            )

        return await serialize_generator(g)

    async def update_status(self, id: int, status: GeneratorStatus) -> GeneratorSchema:
        g = await Generator.get_or_none(id=id)
        if not g:
            raise TSTError(
                "generator-is-not-found",
                f"Generator with ID {id} not found",
                metadata={"status_code": 404},
            )

        g.status = status

        await g.save()
        return await serialize_generator(g)

    async def get_all(self) -> list[GeneratorSchema]:
        gls = await Generator.all()

        gsls = []
        for g in gls:
            gsls.append(await serialize_generator(g))

        return gsls

    async def delete(self, id: int):
        g = await Generator.get_or_none(id=id)
        if not g:
            raise TSTError(
                "generator-not-found",
                f"Generator with ID {id} not found",
                metadata={"status_code": 404},
            )
        await g.delete()

    async def exists(self, *args: Q, **kwargs: Any) -> bool:
        return await Generator.exists(*args, **kwargs)


async def serialize_generator(g: Generator) -> GeneratorSchema:
    e = await Engine.get_or_none(id=g.engine_id)
    if not e:
        raise TSTError(
            "engine-is-missing",
            f"Generator with ID {id} has an engine with {g.engine_id} that is missing",
        )
    es = await serialize_engine(e)
    return GeneratorSchema(id=g.id, name=g.name, status=g.status, engine=es)
