"""FastAPI application — Song Request server."""

from __future__ import annotations

import asyncio
import secrets
from pathlib import Path
from typing import Optional

from fastapi import (FastAPI, Header, HTTPException, Query, Request, WebSocket,
                     WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .media_resolver import resolve_song, resolve_youtube_playlist
from .models import (AddSongRequest, ImportPlaylistRequest, ReorderRequest,
                     Settings, SongUser, UpdateSettingsRequest)
from .store import ChannelStore, store
from .ws_manager import manager

app = FastAPI(title="Song Request Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files for the client ──────────────────────────────────────────

CLIENT_DIR = Path(__file__).resolve().parent.parent / "client"

# ── WebSocket tokens (simple map: token → channel_id) ────────────────────

_ws_tokens: dict[str, str] = {}


# ── Helpers ──────────────────────────────────────────────────────────────

def _get_channel(nightbot_channel: Optional[str]) -> ChannelStore:
    channel_id = nightbot_channel or "default"
    cs = store.get_or_create_channel(channel_id)
    return cs


def _serialize_song(item) -> dict:
    return item.model_dump(by_alias=True)


def _serialize_queue(cs: ChannelStore) -> dict:
    return {
        "queue": [_serialize_song(s) for s in cs.queue],
        "_currentSong": _serialize_song(cs.current_song) if cs.current_song else None,
    }


def _default_user() -> SongUser:
    return SongUser(displayName="Web User", name="webuser", provider="twitch", providerId="web")


# ── Root ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    if (CLIENT_DIR / "index.html").exists():
        return FileResponse(str(CLIENT_DIR / "index.html"))
    return {"message": "Song Request Server is running. Client not found at /client/"}


@app.get("/style.css")
async def style_css():
    return FileResponse(str(CLIENT_DIR / "style.css"), media_type="text/css")


@app.get("/app.js")
async def app_js():
    return FileResponse(str(CLIENT_DIR / "app.js"), media_type="application/javascript")


# ── Channel Resolution (public, no auth) ────────────────────────────────

@app.get("/1/channels/{provider}/{username}")
async def get_channel(provider: str, username: str):
    cs = store.find_channel_by_provider(provider, username)
    if cs is None:
        cs = store.get_or_create_channel(f"{provider}_{username}")
        cs.channel.provider = provider
        cs.channel.name = username
        cs.channel.displayName = username
    return {"channel": cs.channel.model_dump(by_alias=True)}


# ── Settings ─────────────────────────────────────────────────────────────

@app.get("/1/song_requests")
async def get_settings(nightbot_channel: Optional[str] = Header(None, alias="Nightbot-Channel")):
    cs = _get_channel(nightbot_channel)
    return {
        "settings": cs.settings.model_dump(),
        "providers": {"youtube": "YouTube", "soundcloud": "SoundCloud"},
        "playlists": [
            {"id": "channel", "name": "Channel"},
            {"id": "monstercat", "name": "Monstercat"},
        ],
    }


@app.put("/1/song_requests")
async def update_settings(
    body: UpdateSettingsRequest,
    nightbot_channel: Optional[str] = Header(None, alias="Nightbot-Channel"),
):
    cs = _get_channel(nightbot_channel)
    update = body.model_dump(exclude_unset=True)
    current = cs.settings.model_dump()
    current.update(update)
    cs.settings = Settings(**current)

    # Broadcast volume changes
    if "volume" in update:
        await manager.broadcast(cs.channel.id, "songRequestVolume", {"volume": cs.settings.volume})

    return {"settings": cs.settings.model_dump()}


# ── Queue ────────────────────────────────────────────────────────────────

@app.get("/1/song_requests/queue")
async def get_queue(nightbot_channel: Optional[str] = Header(None, alias="Nightbot-Channel")):
    cs = _get_channel(nightbot_channel)
    return _serialize_queue(cs)


@app.post("/1/song_requests/queue")
async def add_to_queue(
    body: AddSongRequest,
    nightbot_channel: Optional[str] = Header(None, alias="Nightbot-Channel"),
):
    cs = _get_channel(nightbot_channel)
    user = _default_user()

    if body.fromPlaylist:
        item = cs.queue_from_playlist(body.q, user)
        if item is None:
            raise HTTPException(status_code=404, detail="Playlist item not found")
    else:
        try:
            track_dict = resolve_song(body.q, cs.settings.searchProvider)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        item = cs.add_to_queue(track_dict, user)

    # If nothing is currently playing, auto-play the first item
    if cs.current_song is None and cs.queue:
        cs.current_song = cs.queue.pop(0)
        cs._reindex_queue()
        await manager.broadcast(cs.channel.id, "songRequestPlay", {"item": _serialize_song(cs.current_song)})

    await manager.broadcast(cs.channel.id, "songRequestQueueAdd", {"item": _serialize_song(item)})
    return {"item": _serialize_song(item)}


@app.delete("/1/song_requests/queue")
async def clear_queue(nightbot_channel: Optional[str] = Header(None, alias="Nightbot-Channel")):
    cs = _get_channel(nightbot_channel)
    cs.clear_queue()
    await manager.broadcast(cs.channel.id, "songRequestQueueClear", {})
    return {"message": "Queue cleared"}


@app.delete("/1/song_requests/queue/{item_id}")
async def remove_from_queue(
    item_id: str,
    nightbot_channel: Optional[str] = Header(None, alias="Nightbot-Channel"),
):
    cs = _get_channel(nightbot_channel)
    removed = cs.remove_from_queue(item_id)
    if removed is None:
        raise HTTPException(status_code=404, detail="Queue item not found")
    await manager.broadcast(cs.channel.id, "songRequestQueueRemove", {"item": _serialize_song(removed)})
    return {"message": "Removed"}


@app.post("/1/song_requests/queue/{item_id}/play")
async def play_queue_item(
    item_id: str,
    nightbot_channel: Optional[str] = Header(None, alias="Nightbot-Channel"),
):
    cs = _get_channel(nightbot_channel)
    item = cs.play_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Queue item not found")
    await manager.broadcast(cs.channel.id, "songRequestPlay", {"item": _serialize_song(item)})
    return {"item": _serialize_song(item)}


@app.post("/1/song_requests/queue/{item_id}/promote")
async def promote_queue_item(
    item_id: str,
    nightbot_channel: Optional[str] = Header(None, alias="Nightbot-Channel"),
):
    cs = _get_channel(nightbot_channel)
    item = cs.promote(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Queue item not found")
    await manager.broadcast(cs.channel.id, "songRequestQueuePromote", {"item": _serialize_song(item)})
    return {"item": _serialize_song(item)}


@app.patch("/1/song_requests/queue/order")
async def reorder_queue(
    body: ReorderRequest,
    nightbot_channel: Optional[str] = Header(None, alias="Nightbot-Channel"),
):
    cs = _get_channel(nightbot_channel)
    cs.reorder(body.order)
    return _serialize_queue(cs)


# ── Playlist ─────────────────────────────────────────────────────────────

@app.get("/1/song_requests/playlist")
async def get_playlist(
    offset: int = Query(0),
    limit: int = Query(20),
    nightbot_channel: Optional[str] = Header(None, alias="Nightbot-Channel"),
):
    cs = _get_channel(nightbot_channel)
    total = len(cs.playlist)
    items = cs.playlist[offset: offset + limit]
    return {
        "playlist": [_serialize_song(s) for s in items],
        "_total": total,
        "_offset": offset,
        "_limit": limit,
    }


@app.post("/1/song_requests/playlist")
async def add_to_playlist(
    body: AddSongRequest,
    nightbot_channel: Optional[str] = Header(None, alias="Nightbot-Channel"),
):
    cs = _get_channel(nightbot_channel)
    try:
        track_dict = resolve_song(body.q, cs.settings.searchProvider)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    item = cs.add_to_playlist(track_dict)
    return {"item": _serialize_song(item)}


@app.delete("/1/song_requests/playlist")
async def clear_playlist(nightbot_channel: Optional[str] = Header(None, alias="Nightbot-Channel")):
    cs = _get_channel(nightbot_channel)
    cs.clear_playlist()
    return {"message": "Playlist cleared"}


@app.delete("/1/song_requests/playlist/{item_id}")
async def remove_from_playlist(
    item_id: str,
    nightbot_channel: Optional[str] = Header(None, alias="Nightbot-Channel"),
):
    cs = _get_channel(nightbot_channel)
    removed = cs.remove_from_playlist(item_id)
    if removed is None:
        raise HTTPException(status_code=404, detail="Playlist item not found")
    return {"message": "Removed"}


@app.post("/1/song_requests/playlist/import")
async def import_playlist(
    body: ImportPlaylistRequest,
    nightbot_channel: Optional[str] = Header(None, alias="Nightbot-Channel"),
):
    cs = _get_channel(nightbot_channel)

    async def _do_import():
        try:
            tracks = resolve_youtube_playlist(body.url)
            for t in tracks:
                cs.add_to_playlist(t)
        except Exception:
            pass  # Fire-and-forget

    asyncio.create_task(_do_import())
    return {"message": "Import started"}


# ── WebSocket Token ──────────────────────────────────────────────────────

@app.get("/1/me/ws_token")
async def get_ws_token(nightbot_channel: Optional[str] = Header(None, alias="Nightbot-Channel")):
    channel_id = nightbot_channel or "default"
    token = secrets.token_urlsafe(32)
    _ws_tokens[token] = channel_id
    return {"token": token}


# ── WebSocket ────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = "", channel: str = ""):
    # Validate token
    channel_id = _ws_tokens.pop(token, None) or channel or "default"
    await manager.connect(websocket, channel_id)
    try:
        while True:
            # Keep connection alive; we don't expect client messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel_id)


# ── Skip (convenience — advance to next song) ───────────────────────────

@app.post("/1/song_requests/queue/skip")
async def skip_song(nightbot_channel: Optional[str] = Header(None, alias="Nightbot-Channel")):
    cs = _get_channel(nightbot_channel)
    next_song = cs.skip()
    if next_song:
        await manager.broadcast(cs.channel.id, "songRequestPlay", {"item": _serialize_song(next_song)})
    else:
        await manager.broadcast(cs.channel.id, "songRequestSkip", {})
    return {
        "_currentSong": _serialize_song(next_song) if next_song else None,
    }


# ── Pause / Resume (broadcast only) ─────────────────────────────────────

@app.post("/1/song_requests/queue/pause")
async def pause_song(nightbot_channel: Optional[str] = Header(None, alias="Nightbot-Channel")):
    cs = _get_channel(nightbot_channel)
    await manager.broadcast(cs.channel.id, "songRequestPause", {})
    return {"message": "Paused"}
