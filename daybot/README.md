# Song Request System

A recreation of Nightbot's song request & queue management feature, with a Python FastAPI backend and vanilla JS frontend.

## Quick Start

```bash
# Start the server (from this directory)
uv run uvicorn server.main:app --host 127.0.0.1 --port 8787

# Open in browser
# http://127.0.0.1:8787
```

## Features

### Queue Management
- Add songs by name (YouTube search) or URL (YouTube/SoundCloud)
- Drag-and-drop reorder
- Play, promote, delete, and clear queue
- Auto-play next song when current ends
- Search/filter within queue

### Player
- Embedded YouTube player with full controls
- Play/pause, skip, previous (history), seek bar
- Volume slider with mute toggle
- Shuffle queue

### Playlist
- Persistent playlist separate from the live queue
- Add songs, import YouTube playlists, queue from playlist
- Paginated display

### Settings
- Enable/disable song requests
- Configure providers, volume, user levels, limits
- YouTube-specific options (music category, liked videos)

### Real-time Updates
- WebSocket connection for live queue updates
- Events: add, remove, clear, promote, play, skip, pause, volume

## API

All endpoints follow the Nightbot API convention under `/1/`. Pass `Nightbot-Channel: default` header.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/1/song_requests` | Get settings |
| PUT | `/1/song_requests` | Update settings |
| GET | `/1/song_requests/queue` | Get queue + current song |
| POST | `/1/song_requests/queue` | Add song `{q: "..."}` |
| DELETE | `/1/song_requests/queue` | Clear queue |
| DELETE | `/1/song_requests/queue/:id` | Remove item |
| POST | `/1/song_requests/queue/:id/play` | Play item |
| POST | `/1/song_requests/queue/:id/promote` | Promote to top |
| PATCH | `/1/song_requests/queue/order` | Reorder `{order: [...]}` |
| POST | `/1/song_requests/queue/skip` | Skip to next |
| GET | `/1/song_requests/playlist` | Get playlist |
| POST | `/1/song_requests/playlist` | Add to playlist |
| DELETE | `/1/song_requests/playlist` | Clear playlist |
| POST | `/1/song_requests/playlist/import` | Import YT playlist |
| GET | `/1/channels/:provider/:username` | Resolve channel |
| GET | `/1/me/ws_token` | Get WebSocket token |
| WS | `/ws?token=...&channel=...` | WebSocket |

## Architecture

- **server/main.py** — FastAPI routes and WebSocket handler
- **server/models.py** — Pydantic data models
- **server/store.py** — In-memory data store
- **server/media_resolver.py** — yt-dlp based song resolver
- **server/ws_manager.py** — WebSocket connection manager
- **client/** — Vanilla HTML/CSS/JS frontend
