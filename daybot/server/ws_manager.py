"""WebSocket connection manager for broadcasting song request events."""

from __future__ import annotations

import json
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    """Manages WebSocket connections per channel."""

    def __init__(self):
        self.connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, channel_id: str) -> None:
        await websocket.accept()
        if channel_id not in self.connections:
            self.connections[channel_id] = []
        self.connections[channel_id].append(websocket)

    def disconnect(self, websocket: WebSocket, channel_id: str) -> None:
        if channel_id in self.connections:
            self.connections[channel_id] = [
                ws for ws in self.connections[channel_id] if ws is not websocket
            ]
            if not self.connections[channel_id]:
                del self.connections[channel_id]

    async def broadcast(self, channel_id: str, event: str, data: Any = None) -> None:
        """Send an event to all connections for a channel."""
        message = json.dumps({"event": event, "data": data})
        if channel_id not in self.connections:
            return
        dead = []
        for ws in self.connections[channel_id]:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, channel_id)


manager = ConnectionManager()
