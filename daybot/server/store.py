"""In-memory data store for channels, queues, playlists, and settings."""

from __future__ import annotations

import copy
from typing import Optional

from .models import Channel, Settings, SongItem, SongUser, Track, new_id, now_iso


class ChannelStore:
    """Per-channel state: queue, playlist, settings, current song."""

    def __init__(self, channel: Channel):
        self.channel = channel
        self.settings = Settings()
        self.queue: list[SongItem] = []
        self.playlist: list[SongItem] = []
        self.current_song: Optional[SongItem] = None

    # ── Queue helpers ────────────────────────────────────────────────

    def _reindex_queue(self) -> None:
        for i, item in enumerate(self.queue):
            item.position = i + 1

    def add_to_queue(self, track_dict: dict, user: Optional[SongUser] = None) -> SongItem:
        track = Track(**track_dict)
        item = SongItem(
            _id=new_id(),
            track=track,
            user=user,
            createdAt=now_iso(),
            updatedAt=now_iso(),
            _position=len(self.queue) + 1,
        )
        self.queue.append(item)
        self._reindex_queue()
        return item

    def remove_from_queue(self, item_id: str) -> Optional[SongItem]:
        for i, item in enumerate(self.queue):
            if item.id == item_id:
                removed = self.queue.pop(i)
                self._reindex_queue()
                return removed
        return None

    def clear_queue(self) -> None:
        self.queue.clear()

    def promote(self, item_id: str) -> Optional[SongItem]:
        for i, item in enumerate(self.queue):
            if item.id == item_id:
                self.queue.pop(i)
                self.queue.insert(0, item)
                self._reindex_queue()
                return item
        return None

    def play_item(self, item_id: str) -> Optional[SongItem]:
        for i, item in enumerate(self.queue):
            if item.id == item_id:
                self.queue.pop(i)
                self.current_song = item
                self._reindex_queue()
                return item
        return None

    def reorder(self, order: list[str]) -> None:
        item_map = {item.id: item for item in self.queue}
        new_queue = []
        for oid in order:
            if oid in item_map:
                new_queue.append(item_map.pop(oid))
        # Append any items not in the order list at the end
        for item in self.queue:
            if item.id in item_map:
                new_queue.append(item)
        self.queue = new_queue
        self._reindex_queue()

    def skip(self) -> Optional[SongItem]:
        """Skip current song → play next in queue, or stop."""
        if self.queue:
            self.current_song = self.queue.pop(0)
            self._reindex_queue()
            return self.current_song
        self.current_song = None
        return None

    # ── Playlist helpers ─────────────────────────────────────────────

    def add_to_playlist(self, track_dict: dict) -> SongItem:
        track = Track(**track_dict)
        item = SongItem(
            _id=new_id(),
            track=track,
            createdAt=now_iso(),
            updatedAt=now_iso(),
            _position=len(self.playlist) + 1,
        )
        self.playlist.append(item)
        return item

    def remove_from_playlist(self, item_id: str) -> Optional[SongItem]:
        for i, item in enumerate(self.playlist):
            if item.id == item_id:
                return self.playlist.pop(i)
        return None

    def clear_playlist(self) -> None:
        self.playlist.clear()

    def queue_from_playlist(self, playlist_item_id: str, user: Optional[SongUser] = None) -> Optional[SongItem]:
        """Copy a playlist item into the queue."""
        for pitem in self.playlist:
            if pitem.id == playlist_item_id:
                return self.add_to_queue(pitem.track.model_dump(), user)
        return None


class DataStore:
    """Global store holding all channels."""

    def __init__(self):
        self.channels: dict[str, ChannelStore] = {}
        self._ensure_default_channel()

    def _ensure_default_channel(self) -> None:
        """Create a default channel for easy testing."""
        ch = Channel(_id="default", name="testchannel", displayName="Test Channel",
                     provider="twitch", providerId="12345")
        self.channels["default"] = ChannelStore(ch)

    def get_channel(self, channel_id: str) -> Optional[ChannelStore]:
        return self.channels.get(channel_id)

    def get_or_create_channel(self, channel_id: str) -> ChannelStore:
        if channel_id not in self.channels:
            ch = Channel(_id=channel_id, name=channel_id, displayName=channel_id)
            self.channels[channel_id] = ChannelStore(ch)
        return self.channels[channel_id]

    def find_channel_by_provider(self, provider: str, username: str) -> Optional[ChannelStore]:
        for cs in self.channels.values():
            if cs.channel.provider == provider and cs.channel.name.lower() == username.lower():
                return cs
        return None


# Singleton
store = DataStore()
