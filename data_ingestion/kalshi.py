from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_TIMEOUT = 20


def _get_json(
    endpoint: str,
    params: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    response = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=timeout)
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dict response from Kalshi, got {type(payload)!r}")
    return payload


def fetch_open_events(limit: int = 200, max_pages: int = 5) -> list[dict[str, Any]]:
    """Fetch open Kalshi events with nested markets included."""
    events: list[dict[str, Any]] = []
    cursor: str | None = None

    for _ in range(max_pages):
        params: dict[str, Any] = {
            "limit": limit,
            "status": "open",
            "with_nested_markets": "true",
        }
        if cursor:
            params["cursor"] = cursor

        payload = _get_json("/events", params=params)
        page_events = payload.get("events", [])
        if not isinstance(page_events, list):
            raise ValueError("Kalshi response did not include an 'events' list")

        events.extend(page_events)
        cursor = payload.get("cursor")
        if not cursor:
            break

    return events


def fetch_open_markets(limit: int = 200, max_pages: int = 5) -> list[dict[str, Any]]:
    """Fetch open Kalshi markets directly from the markets endpoint."""
    markets: list[dict[str, Any]] = []
    cursor: str | None = None

    for _ in range(max_pages):
        params: dict[str, Any] = {
            "limit": limit,
            "status": "open",
        }
        if cursor:
            params["cursor"] = cursor

        payload = _get_json("/markets", params=params)
        page_markets = payload.get("markets", [])
        if not isinstance(page_markets, list):
            raise ValueError("Kalshi response did not include a 'markets' list")

        markets.extend(page_markets)
        cursor = payload.get("cursor")
        if not cursor:
            break

    return markets


def fetch_historical_cutoff_timestamps() -> dict[str, Any]:
    """Fetch the documented cutoff metadata for Kalshi historical APIs."""
    return _get_json("/historical")


def fetch_market_candlesticks(
    series_ticker: str,
    ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int = 1440,
) -> dict[str, Any]:
    """Fetch live Kalshi candlesticks for a market."""
    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": period_interval,
    }
    return _get_json(f"/series/{series_ticker}/markets/{ticker}/candlesticks", params=params)


def fetch_historical_market_candlesticks(
    ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int = 1440,
) -> dict[str, Any]:
    """Fetch historical Kalshi candlesticks for a market."""
    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": period_interval,
    }
    return _get_json(f"/historical/markets/{ticker}/candlesticks", params=params)


def save_raw_kalshi_json(data: Any, output_path: str | Path) -> Path:
    """Persist raw Kalshi data for later inspection."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)

    return path
