# Copyright © 2025-2026 Emmanouil Ragiadakos
# SPDX-License-Identifier: SSPL-1.0
import asyncio
from threading import Thread

from fastapi import WebSocket, WebSocketDisconnect

from src.api.v1.generators.manager import GeneratorManager


class WSEventGeneratorStreamerService:
    _active_connections: list[WebSocket]
    _manager: GeneratorManager
    _event_broadcaster_thread: Thread

    def __init__(self, manager: GeneratorManager):
        self._active_connections = []
        self._manager = manager
        self._event_broadcaster_thread = Thread(
            target=self._event_broadcaster, daemon=True
        )
        self._event_broadcaster_thread.start()

    def _event_broadcaster(self):
        asyncio.run(self._internal_event_broadcaster())

    async def _internal_event_broadcaster(self):
        while True:
            if not self._manager.websocket_event_queue.empty():
                event = self._manager.websocket_event_queue.get()
                print("broadcast event ", event)
                # Broadcast to all connected clients
                for connection in self._active_connections[
                    :
                ]:  # Copy to avoid mutation issues
                    try:
                        await connection.send_text(event)
                    except Exception:
                        self._active_connections.remove(connection)
            await asyncio.sleep(0.1)

    async def create_connection(self, websocket: WebSocket):
        await websocket.accept()
        self._active_connections.append(websocket)
        try:
            while True:
                data = await websocket.receive_text()
        except WebSocketDisconnect:
            self._active_connections.remove(websocket)
