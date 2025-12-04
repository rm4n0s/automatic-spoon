# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0

import argparse
import os
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
        print(exc)
        meta = exc.metadata()
        content: dict[str, str] = {"error": exc.message()}
        if meta is not None:
            if "error_per_field" in meta.keys():
                content["error_per_field"] = meta["error_per_field"]

            if "status_code" in meta.keys():
                status_code = meta["status_code"]
                return JSONResponse(
                    status_code=status_code,
                    content=content,
                )

        return JSONResponse(
            status_code=500,
            content={"error": exc.message(), "for_admin": exc.to_json()},
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        print(exc)
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

    _ = parser.add_argument("--port", type=int, default=8080, help="the port")
    _ = parser.add_argument("--host", type=str, default="localhost", help="the host")
    _ = parser.add_argument(
        "--config", default="config.yaml", help="the configuration file"
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
        reload=False,
        workers=1,
    )
