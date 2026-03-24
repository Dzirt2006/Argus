from pathlib import Path

from fastmcp import FastMCP

BASE_DIR = Path("/data")

mcp = FastMCP("filesystem")


def _safe_path(path: str) -> Path:
    """Resolve a user-provided path and ensure it stays under /data/."""
    resolved = (BASE_DIR / path).resolve()
    if not resolved.is_relative_to(BASE_DIR):
        raise ValueError(f"Path must be under {BASE_DIR}, got {resolved}")
    return resolved


@mcp.tool()
async def read_file(path: str) -> str:
    """Read the full contents of a text file.

    Args:
        path: File path relative to /data/, e.g. 'notes/todo.txt'.
    """
    try:
        return _safe_path(path).read_text(encoding="utf-8")
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def write_file(path: str, content: str) -> str:
    """Write text content to a file. Creates parent directories if they don't exist.

    Args:
        path: File path relative to /data/, e.g. 'notes/todo.txt'.
        content: The text to write.
    """
    try:
        target = _safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Success: wrote {len(content)} chars to {target}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def list_directory(path: str = ".") -> str:
    """List files and subdirectories. Directories end with '/'.

    Args:
        path: Directory path relative to /data/, e.g. 'notes'. Defaults to /data/ root.
    """
    try:
        target = _safe_path(path)
        if not target.is_dir():
            return f"Error: {target} is not a directory"
        entries = sorted(target.iterdir())
        lines = [f"{e.name}/" if e.is_dir() else e.name for e in entries]
        return "\n".join(lines) if lines else "(empty directory)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)
