import shutil
import subprocess
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("system")


@mcp.tool()
async def get_system_uptime() -> str:
    """Get how long the system has been running and the current load average."""
    try:
        seconds = float(Path("/proc/uptime").read_text().split()[0])
        days, remainder = divmod(int(seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, _ = divmod(remainder, 60)
        load = Path("/proc/loadavg").read_text().strip()
        return f"Uptime: {days}d {hours}h {minutes}m\nLoad average: {load}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_disk_usage(path: str = "/") -> str:
    """Check available disk space for a given mount point.

    Args:
        path: Mount point to check, e.g. '/' or '/data'. Defaults to '/'.
    """
    try:
        usage = shutil.disk_usage(path)
        total_gb = usage.total / (1024**3)
        used_gb = usage.used / (1024**3)
        free_gb = usage.free / (1024**3)
        pct = (usage.used / usage.total) * 100
        return f"Disk {path}: {used_gb:.1f}GB / {total_gb:.1f}GB ({pct:.0f}% used, {free_gb:.1f}GB free)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_memory_usage() -> str:
    """Get current RAM usage: total, used, and available memory."""
    try:
        meminfo = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            parts = line.split(":")
            if len(parts) == 2:
                meminfo[parts[0].strip()] = int(parts[1].strip().split()[0])
        total = meminfo["MemTotal"] / (1024 * 1024)
        available = meminfo["MemAvailable"] / (1024 * 1024)
        used = total - available
        pct = (used / total) * 100
        return f"RAM: {used:.1f}GB / {total:.1f}GB ({pct:.0f}% used, {available:.1f}GB available)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
async def get_gpu_status() -> str:
    """Get NVIDIA GPU status: model name, temperature, VRAM usage, and utilization."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return f"Error: nvidia-smi failed: {result.stderr.strip()}"
        output = []
        for line in result.stdout.strip().splitlines():
            name, temp, mem_used, mem_total, util = [x.strip() for x in line.split(",")]
            output.append(f"{name}: {temp}°C, {mem_used}MB/{mem_total}MB VRAM ({util}% util)")
        return "\n".join(output)
    except FileNotFoundError:
        return "Error: nvidia-smi not found — no NVIDIA GPU available in this container"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8002)
