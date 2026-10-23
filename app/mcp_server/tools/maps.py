"""Google Maps tools: geocode, directions, nearby_places.

Real implementation uses the Google Maps REST API via httpx; mock fallback
returns deterministic coordinates and points of interest.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .._http import http_get
from ._mock import mock_delay, stable_hash
from ...config import settings

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"


def _maps_enabled() -> bool:
    return bool(settings.google_maps_api_key) and not settings.use_mock_fallback


async def geocode(place: str) -> dict[str, Any]:
    """Resolve a place name to lat/lng and formatted address."""
    if not _maps_enabled():
        await mock_delay()
        # Deterministic mock based on the place name hash.
        h = stable_hash(place) % 90
        return {
            "place": place,
            "formatted": f"{place}",
            "lat": round(20 + (h % 40) + 0.01, 4),
            "lng": round(-10 + (h % 120) - 0.01, 4),
            "mock": True,
        }

    params = {"address": place, "key": settings.google_maps_api_key}
    data = await http_get(GEOCODE_URL, params=params)
    if not data.get("results"):
        return {"place": place, "error": "no results", "mock": False}
    loc = data["results"][0]
    geom = loc["geometry"]["location"]
    return {
        "place": place,
        "formatted": loc["formatted_address"],
        "lat": geom["lat"],
        "lng": geom["lng"],
        "mock": False,
    }


async def directions(origin: str, destination: str) -> dict[str, Any]:
    """Driving directions between two places."""
    if not _maps_enabled():
        await mock_delay()
        return {
            "origin": origin,
            "destination": destination,
            "distance_km": round(stable_hash(origin + destination) % 200 + 1, 1),
            "duration_min": round(stable_hash(destination) % 180 + 10, 1),
            "mock": True,
        }

    params = {
        "origin": origin,
        "destination": destination,
        "key": settings.google_maps_api_key,
    }
    data = await http_get(DIRECTIONS_URL, params=params)
    if not data.get("routes"):
        return {"origin": origin, "destination": destination, "error": "no route", "mock": False}
    leg = data["routes"][0]["legs"][0]
    return {
        "origin": leg["start_address"],
        "destination": leg["end_address"],
        "distance_km": round(leg["distance"]["value"] / 1000, 2),
        "duration_min": round(leg["duration"]["value"] / 60, 1),
        "mock": False,
    }


async def nearby_places(place: str, *, radius_m: int = 1500) -> list[dict[str, Any]]:
    """Find notable nearby places (attractions/restaurants/hotels)."""
    if not _maps_enabled():
        await mock_delay()
        seed = stable_hash(place) % 5
        types = ["attraction", "restaurant", "museum", "park", "hotel"]
        return [
            {
                "name": f"{place} {types[(seed + i) % len(types)]} #{i + 1}",
                "type": types[(seed + i) % len(types)],
                "rating": round(3.5 + (i % 1.5), 1),
                "place": place,
                "mock": True,
            }
            for i in range(5)
        ]

    geo = await geocode(place)
    if "lat" not in geo:
        return []
    params = {
        "location": f"{geo['lat']},{geo['lng']}",
        "radius": radius_m,
        "key": settings.google_maps_api_key,
    }
    data = await http_get(NEARBY_URL, params=params)
    out: list[dict[str, Any]] = []
    for r in data.get("results", [])[:10]:
        out.append(
            {
                "name": r.get("name"),
                "type": (r.get("types") or ["unknown"])[0],
                "rating": r.get("rating"),
                "place": place,
                "mock": False,
            }
        )
    return out


# Avoid unused import lint when httpx is only used inside the helper module.
_ = httpx