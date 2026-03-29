"""YouTube Music MCP server.

Uses ytmusicapi for search/browse and yt-dlp + mpv for playback.
MPV runs as a background process with IPC socket for control.

Tools:
  - play_music: search and play a song/artist/album
  - pause_music: pause or resume playback
  - skip_track: skip to next track in queue
  - stop_music: stop playback entirely
  - search_music: search without playing (browse results)
  - get_now_playing: what's currently playing
"""

import json
import subprocess
import socket
import time
from pathlib import Path

from fastmcp import FastMCP
from ytmusicapi import YTMusic

mcp = FastMCP("media", port=8006)

_MPV_SOCKET = "/tmp/argus-mpv-socket"
_ytmusic = YTMusic()


def _mpv_command(*args) -> dict | None:
    """Send a JSON IPC command to the running mpv instance."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(_MPV_SOCKET)
        cmd = json.dumps({"command": list(args)}) + "\n"
        sock.sendall(cmd.encode())
        resp = sock.recv(4096).decode()
        sock.close()
        return json.loads(resp)
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        return None


def _mpv_get_property(prop: str):
    """Get a property from the running mpv instance."""
    result = _mpv_command("get_property", prop)
    if result and "data" in result:
        return result["data"]
    return None


def _is_mpv_running() -> bool:
    return Path(_MPV_SOCKET).exists() and _mpv_command("get_property", "pid") is not None


def _play_url(url: str, title: str = "") -> str:
    """Start mpv with a YouTube URL via yt-dlp."""
    # Kill existing mpv if running
    if _is_mpv_running():
        _mpv_command("quit")
        time.sleep(0.5)

    subprocess.Popen(
        [
            "mpv",
            "--no-video",
            f"--input-ipc-server={_MPV_SOCKET}",
            "--really-quiet",
            url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(1)
    return f"Now playing: **{title}**" if title else "Playback started."


@mcp.tool()
def play_music(query: str) -> str:
    """Search YouTube Music and play the top result.

    Use this when the user asks to play music, a song, an artist, or an album.
    Examples: "play some jazz", "play Bohemian Rhapsody", "play Taylor Swift"

    Args:
        query: What to play — song name, artist, genre, or mood.
               Example: "lofi beats" or "Beethoven Moonlight Sonata"
    """
    results = _ytmusic.search(query, filter="songs", limit=1)

    if not results:
        results = _ytmusic.search(query, limit=1)

    if not results:
        return f"No results found for '{query}'."

    item = results[0]
    video_id = item.get("videoId")
    if not video_id:
        return f"No playable result for '{query}'."

    title = item.get("title", query)
    artists = ", ".join(a["name"] for a in item.get("artists", []))
    display = f"{title} — {artists}" if artists else title

    url = f"https://music.youtube.com/watch?v={video_id}"
    return _play_url(url, display)


@mcp.tool()
def pause_music() -> str:
    """Pause or resume music playback.

    Use this when the user says "pause", "resume", "unpause", or
    "pause the music".
    """
    if not _is_mpv_running():
        return "Nothing is playing."

    _mpv_command("cycle", "pause")
    paused = _mpv_get_property("pause")
    return "Paused." if paused else "Resumed."


@mcp.tool()
def skip_track() -> str:
    """Skip to the next track.

    Use this when the user says "skip", "next", or "next song".
    """
    if not _is_mpv_running():
        return "Nothing is playing."

    _mpv_command("playlist-next")
    return "Skipped to next track."


@mcp.tool()
def stop_music() -> str:
    """Stop music playback entirely.

    Use this when the user says "stop the music", "turn off music",
    or "stop playing".
    """
    if not _is_mpv_running():
        return "Nothing is playing."

    _mpv_command("quit")
    return "Playback stopped."


@mcp.tool()
def search_music(query: str, max_results: int = 5) -> str:
    """Search YouTube Music without playing — just show results.

    Use this when the user wants to browse or asks "what songs does X have"
    without explicitly asking to play anything.

    Args:
        query: Search query — song, artist, album, or genre.
        max_results: Number of results to show (1-10, default 5).
    """
    max_results = max(1, min(10, max_results))
    results = _ytmusic.search(query, filter="songs", limit=max_results)

    if not results:
        return f"No results for '{query}'."

    lines = [f"**Results for '{query}':**\n"]
    for i, item in enumerate(results, 1):
        title = item.get("title", "Unknown")
        artists = ", ".join(a["name"] for a in item.get("artists", []))
        album = item.get("album", {}).get("name", "")
        duration = item.get("duration", "")

        line = f"{i}. **{title}** — {artists}"
        if album:
            line += f" ({album})"
        if duration:
            line += f" [{duration}]"
        lines.append(line)

    return "\n".join(lines)


@mcp.tool()
def get_now_playing() -> str:
    """Get information about what's currently playing.

    Use this when the user asks "what's playing", "what song is this",
    or "what are we listening to".
    """
    if not _is_mpv_running():
        return "Nothing is playing."

    title = _mpv_get_property("media-title") or "Unknown"
    position = _mpv_get_property("time-pos") or 0
    duration = _mpv_get_property("duration") or 0
    paused = _mpv_get_property("pause")

    pos_min, pos_sec = divmod(int(position), 60)
    dur_min, dur_sec = divmod(int(duration), 60)
    status = "Paused" if paused else "Playing"

    return (
        f"**{status}:** {title}\n"
        f"Position: {pos_min}:{pos_sec:02d} / {dur_min}:{dur_sec:02d}"
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
