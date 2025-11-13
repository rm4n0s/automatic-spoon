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

from src.api.v1.di_container import container
from src.api.v1.generators.manager import ProcessManager
from src.api.v1.router import api_router
from src.core.config import enable_hugging_face_envs, read_config
from src.core.tags.user_errors import user_error_responses
from src.db.database import async_close_db, async_init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.getenv("APP_CONFIG_PATH")
    if not config_path:
        raise RuntimeError("APP_CONFIG_PATH environment variable is required")
    config = read_config(config_path)
    app.state.config = config
    enable_hugging_face_envs(config)
    await async_init_db(config.db_path)

    container: AsyncContainer = app.state.dishka_container  # Set by setup_dishka
    await container.get(ProcessManager)
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


app = FastAPI(lifespan=lifespan, title="Automatic Spoon")
setup_dishka(container, app)
add_exception_handlers(app)
app.include_router(api_router, prefix="/api/v1")


def main():
    # logging.basicConfig(level=logging.DEBUG)
    multiprocessing.set_start_method("spawn")
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

    print(args.config)
    os.environ["APP_CONFIG_PATH"] = args.config

    uvicorn.run(
        "src.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=args.reload_dirs,
    )
