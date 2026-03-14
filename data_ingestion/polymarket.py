from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
DEFAULT_TIMEOUT = 20


def _parse_stringified_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return [value]

    value = value.strip()
    if not value:
        return []

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [item.strip() for item in value.split(",") if item.strip()]

    if isinstance(parsed, list):
        return parsed
    return [parsed]


def _normalize_market_payload(market: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(market)
    normalized["outcomes"] = _parse_stringified_list(market.get("outcomes"))
    normalized["outcomePrices"] = _parse_stringified_list(market.get("outcomePrices"))
    normalized["clobTokenIds"] = _parse_stringified_list(market.get("clobTokenIds"))
    return normalized


def _get_json(
    endpoint: str,
    params: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[dict[str, Any]]:
    response = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=timeout)
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list response from Polymarket, got {type(payload)!r}")
    return payload


def _get_clob_json(
    endpoint: str,
    params: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    response = requests.get(f"{CLOB_BASE_URL}{endpoint}", params=params, timeout=timeout)
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dict response from Polymarket CLOB, got {type(payload)!r}")
    return payload


def fetch_active_events(limit: int = 100, max_pages: int = 5) -> list[dict[str, Any]]:
    """Fetch active Polymarket events. Events contain their associated markets."""
    events: list[dict[str, Any]] = []
    offset = 0

    for _ in range(max_pages):
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
        }
        page_events = _get_json("/events", params=params)

        for event in page_events:
            normalized_event = dict(event)
            markets = normalized_event.get("markets", [])
            if isinstance(markets, list):
                normalized_event["markets"] = [
                    _normalize_market_payload(market)
                    for market in markets
                    if isinstance(market, dict)
                ]
            events.append(normalized_event)

        if len(page_events) < limit:
            break
        offset += limit

    return events


def fetch_active_markets(limit: int = 100, max_pages: int = 5) -> list[dict[str, Any]]:
    """Fetch active Polymarket markets directly from the markets endpoint."""
    markets: list[dict[str, Any]] = []
    offset = 0

    for _ in range(max_pages):
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
        }
        page_markets = _get_json("/markets", params=params)
        markets.extend(
            _normalize_market_payload(market)
            for market in page_markets
            if isinstance(market, dict)
        )

        if len(page_markets) < limit:
            break
        offset += limit

    return markets


def fetch_prices_history(
    market_id: str,
    start_ts: int,
    end_ts: int,
    interval: str = "1m",
) -> dict[str, Any]:
    """Fetch Polymarket CLOB price history for a token/market id."""
    params = {
        "market": market_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "interval": interval,
    }
    return _get_clob_json("/prices-history", params=params)


def save_raw_polymarket_json(data: Any, output_path: str | Path) -> Path:
    """Persist raw Polymarket data for later inspection."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)

    return path
