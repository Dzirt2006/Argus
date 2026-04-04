"""Weather MCP server — via Open-Meteo API.

Free, no API key. Geocodes city names automatically.

Tools:
  - get_current_weather: current conditions for a city
  - get_forecast: multi-day forecast for a city
"""

from fastmcp import FastMCP
import httpx

mcp = FastMCP("weather")

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def _geocode(city: str) -> tuple[float, float, str]:
    """Resolve a city name to (latitude, longitude, display_name).

    Tries the full query first, then just the city name before any comma.
    Raises ValueError if the city is not found.
    """
    # Normalize: strip, remove hyphens, try full query then city-only
    normalized = city.strip().replace("-", " ")
    city_only = normalized.split(",")[0].strip()
    for query in dict.fromkeys([normalized, city_only]):
        resp = httpx.get(_GEOCODE_URL, params={"name": query, "count": 5})
        resp.raise_for_status()
        results = resp.json().get("results")
        if results:
            r = results[0]
            name = f"{r['name']}, {r.get('admin1', '')}, {r.get('country', '')}".strip(", ")
            return r["latitude"], r["longitude"], name
    raise ValueError(f"Could not find city: {city}")


@mcp.tool()
def get_current_weather(city: str) -> str:
    """Get current weather conditions for a city.

    Use this when the user asks about the weather right now, current temperature,
    or conditions outside.

    Args:
        city: Full city name, e.g. "Madison, WI" or "London, UK".
              Be specific to avoid ambiguity (there are many Springfields).
    """
    lat, lon, name = _geocode(city)
    resp = httpx.get(
        _WEATHER_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                       "wind_speed_10m,weather_code,precipitation",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
        },
    )
    resp.raise_for_status()
    current = resp.json()["current"]

    return (
        f"**Current weather in {name}:**\n"
        f"Temperature: {current['temperature_2m']}°F "
        f"(feels like {current['apparent_temperature']}°F)\n"
        f"Humidity: {current['relative_humidity_2m']}%\n"
        f"Wind: {current['wind_speed_10m']} mph\n"
        f"Precipitation: {current['precipitation']} mm\n"
        f"Conditions: {_weather_code_to_text(current['weather_code'])}"
    )


@mcp.tool()
def get_forecast(city: str, days: int = 3) -> str:
    """Get a multi-day weather forecast for a city.

    Use this when the user asks about future weather, weekend forecast,
    or wants to plan around weather conditions.

    Args:
        city: Full city name, e.g. "Madison, WI" or "Tokyo, Japan".
        days: Number of days to forecast (1-7, default 3).
    """
    days = max(1, min(7, days))
    lat, lon, name = _geocode(city)
    resp = httpx.get(
        _WEATHER_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                     "weather_code,wind_speed_10m_max",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "forecast_days": days,
        },
    )
    resp.raise_for_status()
    daily = resp.json()["daily"]

    lines = [f"**{days}-day forecast for {name}:**\n"]
    for i in range(len(daily["time"])):
        lines.append(
            f"**{daily['time'][i]}**: "
            f"{daily['temperature_2m_min'][i]}–{daily['temperature_2m_max'][i]}°F, "
            f"{_weather_code_to_text(daily['weather_code'][i])}, "
            f"wind up to {daily['wind_speed_10m_max'][i]} mph, "
            f"precip {daily['precipitation_sum'][i]} mm"
        )

    return "\n".join(lines)


def _weather_code_to_text(code: int) -> str:
    """Convert WMO weather code to human-readable text."""
    codes = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
        82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm", 96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return codes.get(code, f"Unknown ({code})")


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8004)
