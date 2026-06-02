"""Resolve song URLs and search queries using yt-dlp."""

from __future__ import annotations

import re

import yt_dlp


_YDL_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
    "extract_flat": False,
    "noplaylist": True,
}

# Patterns that look like URLs
_URL_RE = re.compile(r"https?://", re.IGNORECASE)

# YouTube URL patterns
_YT_VIDEO_RE = re.compile(
    r"(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([\w-]+)"
)
_YT_PLAYLIST_RE = re.compile(r"youtube\.com/playlist\?.*list=([\w-]+)")

# SoundCloud pattern
_SC_RE = re.compile(r"soundcloud\.com/", re.IGNORECASE)


def _is_url(q: str) -> bool:
    return bool(_URL_RE.match(q.strip()))


def _detect_provider(url: str) -> str:
    if _YT_VIDEO_RE.search(url):
        return "youtube"
    if _SC_RE.search(url):
        return "soundcloud"
    return "youtube"


def resolve_song(query: str, search_provider: str = "youtube") -> dict:
    """Resolve a query (URL or search string) into track metadata.

    Returns a dict with keys: provider, providerId, url, title, artist, duration,
    artistAvatarUrl, thumbnailAccentColor.
    """
    q = query.strip()

    if _is_url(q):
        provider = _detect_provider(q)
        search_query = q
    else:
        provider = search_provider
        if provider == "soundcloud":
            search_query = f"scsearch1:{q}"
        else:
            search_query = f"ytsearch1:{q}"

    opts = {**_YDL_OPTS}

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(search_query, download=False)

    # If search returns a playlist of results, take the first
    if info and "entries" in info:
        entries = list(info["entries"])
        if not entries:
            raise ValueError(f"No results found for: {query}")
        info = entries[0]

    if not info:
        raise ValueError(f"No results found for: {query}")

    # Determine provider and ID
    extractor = (info.get("extractor_key") or info.get("extractor") or "").lower()
    if "soundcloud" in extractor:
        provider = "soundcloud"
        provider_id = str(info.get("id", ""))
        url = info.get("url") or info.get("webpage_url") or q
    else:
        provider = "youtube"
        provider_id = info.get("id", "")
        url = f"https://www.youtube.com/watch?v={provider_id}" if provider_id else q

    return {
        "provider": provider,
        "providerId": provider_id,
        "url": info.get("webpage_url") or url,
        "title": info.get("title") or "Unknown",
        "artist": info.get("uploader") or info.get("artist") or info.get("channel") or "",
        "duration": int(info.get("duration") or 0),
        "artistAvatarUrl": info.get("channel_url") or "",
        "thumbnailAccentColor": None,
    }


def resolve_youtube_playlist(playlist_url: str) -> list[dict]:
    """Resolve all videos in a YouTube playlist. Returns list of track dicts."""
    opts = {
        **_YDL_OPTS,
        "extract_flat": True,
        "noplaylist": False,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

    if not info or "entries" not in info:
        return []

    results = []
    for entry in info["entries"]:
        if not entry:
            continue
        vid_id = entry.get("id", "")
        results.append({
            "provider": "youtube",
            "providerId": vid_id,
            "url": entry.get("url") or f"https://www.youtube.com/watch?v={vid_id}",
            "title": entry.get("title") or "Unknown",
            "artist": entry.get("uploader") or entry.get("channel") or "",
            "duration": int(entry.get("duration") or 0),
            "artistAvatarUrl": "",
            "thumbnailAccentColor": None,
        })

    return results
