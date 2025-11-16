import argparse
import logging
import multiprocessing
import os
import typing
from contextlib import asynccontextmanager

import uvicorn
from dishka.async_container import AsyncContainer
from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pytsterrors import TSTError

from src.api.v1.di_container import create_dishka_container
from src.api.v1.generators.manager import ProcessManager
from src.api.v1.router import api_router
from src.core.config import Config, enable_hugging_face_envs, read_config
from src.core.tags.user_errors import user_error_responses
from src.db.database import async_close_db, async_init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    container: AsyncContainer = app.state.dishka_container  # Set by setup_dishka
    config = await container.get(Config)
    os.makedirs(config.images_path, exist_ok=True)
    enable_hugging_face_envs(config)
    await async_init_db(config.db_path)
    _ = await container.get(ProcessManager)
    yield
    print("closing server")
    await async_close_db()


def add_exception_handlers(app: FastAPI):
    @app.exception_handler(TSTError)
    async def tst_error_handler(request: Request, exc: TSTError):
        error = user_error_responses.get(exc.tag())
        if error is not None:
            content: dict[str, typing.Any] = {"error": error.response}
            meta = exc.metadata()
            if meta is not None:
                content["metadata"] = meta
            return JSONResponse(
                status_code=error.status,
                content=content,
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"error": exc.message(), "for_admin": exc.to_json()},
            )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        tst = TSTError(
            "unexpected_error",
            "an unexpected error:" + exc.__str__(),
            other_exception=exc,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "for_admin": tst.to_json()},
        )


def main():
    # logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(
        prog="Automatic Spoon",
        description="It is a server for generating images",
        epilog="Text at the bottom of help",
    )

    _ = parser.add_argument("--port", default=8080, help="the port")
    _ = parser.add_argument("--host", default="localhost", help="the host")
    _ = parser.add_argument(
        "--config", default="config.yaml", help="the configuration file"
    )
    _ = parser.add_argument(
        "--reload", default=False, help="reload server after source code change"
    )
    _ = parser.add_argument(
        "--reload-dirs",
        default="./src",
        help="reload server after changes in specific folder",
    )
    args = parser.parse_args()
    config_path = args.config

    config = read_config(config_path)
    container = create_dishka_container(config)
    app = FastAPI(lifespan=lifespan, title="Automatic Spoon")
    setup_dishka(container, app)
    add_exception_handlers(app)
    app.include_router(api_router, prefix="/api/v1")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=args.reload_dirs,
    )
