# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0
from dishka.integrations.fastapi import FromDishka, inject
from fastapi import APIRouter, WebSocket

from src.api.v1.websockets.services import WSEventGeneratorStreamerService

router = APIRouter()


# WebSocket for events
@router.websocket("/events/generators")
@inject
async def events_websocket(
    websocket: WebSocket, streamer: FromDishka[WSEventGeneratorStreamerService]
):
    await streamer.create_connection(websocket)
