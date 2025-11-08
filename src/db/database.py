from tortoise import Tortoise


async def async_init_db(filepath: str) -> None:
    await Tortoise.init(
        db_url=f"sqlite://{filepath}",
        modules={
            "models": [
                "src.db.models",
            ],
        },
    )
    await Tortoise.generate_schemas()


async def async_close_db() -> None:
    await Tortoise.close_connections()
