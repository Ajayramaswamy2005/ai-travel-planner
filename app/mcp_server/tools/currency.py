"""Currency tools: convert_currency, list_currencies.

Uses the free exchangerate.host API when key is present; mock otherwise.
"""

from __future__ import annotations

import logging
from typing import Any

from .._http import http_get
from ._mock import mock_delay
from ...config import settings

logger = logging.getLogger(__name__)

CONVERT_URL = "https://api.exchangerate.host/convert"
SYMBOLS_URL = "https://api.exchangerate.host/symbols"


def _currency_enabled() -> bool:
    return not settings.use_mock_fallback


# Realistic-ish static rates for mock mode (relative to USD).
_MOCK_RATES = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "INR": 83.0,
    "JPY": 150.0,
    "AUD": 1.52,
    "CAD": 1.36,
    "SGD": 1.35,
    "AED": 3.67,
    "THB": 36.0,
}


async def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict[str, Any]:
    """Convert an amount between two currencies."""
    f, t = from_currency.upper(), to_currency.upper()
    if not _currency_enabled() or not settings.currency_api_key:
        await mock_delay()
        rate = _MOCK_RATES.get(t, 1.0) / _MOCK_RATES.get(f, 1.0)
        return {
            "from": f,
            "to": t,
            "amount": amount,
            "rate": round(rate, 4),
            "converted": round(amount * rate, 2),
            "mock": True,
        }

    params = {"from": f, "to": t, "amount": amount, "access_key": settings.currency_api_key}
    data = await http_get(CONVERT_URL, params=params)
    return {
        "from": f,
        "to": t,
        "amount": amount,
        "rate": data.get("info", {}).get("rate"),
        "converted": data.get("result"),
        "mock": False,
    }


async def list_currencies() -> list[str]:
    """Return supported currency codes."""
    if not _currency_enabled() or not settings.currency_api_key:
        await mock_delay()
        return list(_MOCK_RATES.keys())
    data = await http_get(SYMBOLS_URL, params={"access_key": settings.currency_api_key})
    return list((data.get("symbols") or {}).keys())