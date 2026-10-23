"""Weather tools: current_weather, forecast.

Real implementation uses the OpenWeatherMap API; mock returns deterministic
plausible weather data.
"""

from __future__ import annotations

import logging
from typing import Any

from .._http import http_get
from ._mock import mock_delay, stable_hash
from ...config import settings

logger = logging.getLogger(__name__)

CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


def _weather_enabled() -> bool:
    return bool(settings.weather_api_key) and not settings.use_mock_fallback


def _mock_state(seed: int) -> dict[str, Any]:
    states = ["clear", "cloudy", "rain", "partly cloudy", "storm"]
    return {
        "condition": states[seed % len(states)],
        "temp_c": round(15 + (seed % 15), 1),
        "temp_feels_c": round(13 + (seed % 17), 1),
        "humidity": 40 + (seed % 50),
        "wind_kmh": round(5 + (seed % 25), 1),
    }


async def current_weather(place: str) -> dict[str, Any]:
    """Current weather for a place."""
    if not _weather_enabled():
        await mock_delay()
        return {"place": place, **_mock_state(stable_hash(place) % 5), "mock": True}

    params = {"q": place, "appid": settings.weather_api_key, "units": "metric"}
    data = await http_get(CURRENT_URL, params=params)
    return {
        "place": data.get("name", place),
        "condition": data["weather"][0]["main"].lower(),
        "temp_c": data["main"]["temp"],
        "temp_feels_c": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "wind_kmh": data["wind"].get("speed", 0),
        "mock": False,
    }


async def forecast(place: str, *, days: int = 3) -> list[dict[str, Any]]:
    """Multi-day forecast (clamped to 5)."""
    days = max(1, min(days, 5))
    if not _weather_enabled():
        await mock_delay()
        base = stable_hash(place) % 5
        return [
            {"day": i + 1, "place": place, **_mock_state((base + i) % 5), "mock": True}
            for i in range(days)
        ]

    params = {"q": place, "appid": settings.weather_api_key, "units": "metric", "cnt": days * 8}
    data = await http_get(FORECAST_URL, params=params)
    # OpenWeather returns 3-hour steps; collapse by day.
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for entry in data.get("list", []):
        day = int(entry["dt"] // 86400)
        if day in seen:
            continue
        seen.add(day)
        out.append(
            {
                "day": len(out) + 1,
                "place": place,
                "condition": entry["weather"][0]["main"].lower(),
                "temp_c": entry["main"]["temp"],
                "humidity": entry["main"]["humidity"],
                "mock": False,
            }
        )
        if len(out) >= days:
            break
    return out