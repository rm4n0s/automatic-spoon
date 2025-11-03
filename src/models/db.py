from tortoise import Tortoise


async def init_db(filepath) -> None:
    await Tortoise.init(
        db_url=f"sqlite://{filepath}",
        modules={
            "models": ["src.models.aimodel", "src.models.engine", "src.models.image", "src.models.job"],
        },
    )
    await Tortoise.generate_schemas()


async def close_db() -> None:
    await Tortoise.close_connections()
