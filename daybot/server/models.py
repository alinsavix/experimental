"""Data models for the song request system."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


def new_id() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ── Track ────────────────────────────────────────────────────────────────

class Track(BaseModel):
    provider: str  # "youtube" or "soundcloud"
    providerId: str
    url: str
    title: str
    artist: str = ""
    duration: int = 0  # seconds
    artistAvatarUrl: str = ""
    thumbnailAccentColor: Optional[str] = None


# ── User ─────────────────────────────────────────────────────────────────

class SongUser(BaseModel):
    provider: str = "twitch"
    providerId: str = ""
    name: str = ""
    displayName: str = ""
    userType: str = "regular"


# ── Song Item (queue or playlist entry) ──────────────────────────────────

class SongItem(BaseModel):
    id: str = Field(alias="_id", default_factory=new_id)
    track: Track
    user: Optional[SongUser] = None
    createdAt: str = Field(default_factory=now_iso)
    updatedAt: str = Field(default_factory=now_iso)
    position: int = Field(alias="_position", default=0)

    model_config = {"populate_by_name": True}


# ── Settings ─────────────────────────────────────────────────────────────

class YouTubeSettings(BaseModel):
    limitToMusic: bool = False
    limitToLikedVideos: bool = False


class LimitsSettings(BaseModel):
    queue: int = 20
    user: int = 5
    playlistOnly: bool = False
    exemptUserLevel: str = "moderator"


class Settings(BaseModel):
    enabled: bool = False
    providers: list[str] = Field(default_factory=lambda: ["youtube", "soundcloud"])
    playlist: str = "channel"
    userLevel: str = "everyone"
    searchProvider: str = "youtube"
    youtube: YouTubeSettings = Field(default_factory=YouTubeSettings)
    limits: LimitsSettings = Field(default_factory=LimitsSettings)
    volume: int = 50


# ── Channel ──────────────────────────────────────────────────────────────

class Channel(BaseModel):
    id: str = Field(alias="_id", default_factory=new_id)
    name: str
    displayName: str = ""
    provider: str = "twitch"
    providerId: str = ""

    model_config = {"populate_by_name": True}


# ── Request / Response Bodies ────────────────────────────────────────────

class AddSongRequest(BaseModel):
    q: str
    fromPlaylist: bool = False
    captcha: Optional[str] = None


class ReorderRequest(BaseModel):
    order: list[str]


class ImportPlaylistRequest(BaseModel):
    url: str


class UpdateSettingsRequest(BaseModel):
    enabled: Optional[bool] = None
    providers: Optional[list[str]] = None
    playlist: Optional[str] = None
    userLevel: Optional[str] = None
    searchProvider: Optional[str] = None
    youtube: Optional[YouTubeSettings] = None
    limits: Optional[LimitsSettings] = None
    volume: Optional[int] = None
