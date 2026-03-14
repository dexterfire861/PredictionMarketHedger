from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TypedDict


UNIFIED_CONTRACT_FIELDS = [
    "source",
    "contract_id",
    "event_id",
    "question",
    "category_raw",
    "risk_category",
    "risk_tags",
    "implied_probability",
    "yes_price",
    "no_price",
    "volume",
    "liquidity",
    "expiration_ts",
    "active",
    "tradable",
    "geo_scope",
    "time_horizon",
    "basis_risk_notes",
    "url",
]


class NormalizedContract(TypedDict):
    source: str
    contract_id: str | None
    event_id: str | None
    question: str | None
    category_raw: str | None
    risk_category: str
    risk_tags: str
    implied_probability: float | None
    yes_price: float | None
    no_price: float | None
    volume: float | None
    liquidity: float | None
    expiration_ts: str | None
    active: bool
    tradable: bool
    geo_scope: str
    time_horizon: str
    basis_risk_notes: str
    url: str | None


RISK_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "inflation": ("inflation", "cpi", "pce", "ppi"),
    "interest_rate": ("interest rate", "rate cut", "rate hike", "fed", "fomc", "treasury yield"),
    "tariff": ("tariff", "duties", "import tax", "trade war"),
    "recession": ("recession", "gdp contraction", "economic downturn"),
    "labor_market": ("jobs report", "nonfarm payroll", "payrolls", "jobless claims", "unemployment", "labor market"),
    "hurricane": ("hurricane", "tropical storm", "storm surge"),
    "weather": ("weather", "temperature", "rainfall", "snow", "heat wave", "blizzard"),
    "pandemic_health": ("pandemic", "covid", "flu", "outbreak", "public health", "cdc", "who"),
    "equity_market": ("s&p", "nasdaq", "dow", "stock market", "equity", "spy", "qqq", "russell"),
    "geopolitical": ("election", "president", "congress", "war", "ceasefire", "nato", "china", "russia", "ukraine", "israel", "gaza", "iran"),
}


def _unwrap_collection(raw_data: Any, preferred_key: str) -> list[dict[str, Any]]:
    if raw_data is None:
        return []
    if isinstance(raw_data, list):
        return [item for item in raw_data if isinstance(item, dict)]
    if isinstance(raw_data, dict):
        value = raw_data.get(preferred_key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_probability(value: Any) -> float | None:
    number = _to_float(value)
    if number is None:
        return None
    if number > 1:
        number = number / 100.0
    return round(max(0.0, min(1.0, number)), 4)


def _midpoint(first: Any, second: Any) -> float | None:
    left = _normalize_probability(first)
    right = _normalize_probability(second)
    if left is not None and right is not None:
        return round((left + right) / 2, 4)
    return left if left is not None else right


def _coerce_timestamp(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return datetime.fromtimestamp(int(stripped), tz=timezone.utc).isoformat()
        return stripped
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    normalized = _coerce_timestamp(value)
    if not normalized:
        return None

    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None

    # Some API fields arrive as date-only strings like "2025-12-31".
    # Treat those naive timestamps as UTC to avoid mixing timezone-aware and naive values.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed


def _infer_time_horizon(expiration_ts: Any) -> str:
    expiration_dt = _parse_timestamp(expiration_ts)
    if not expiration_dt:
        return "unknown"

    days_until_expiry = (expiration_dt - datetime.now(timezone.utc)).total_seconds() / 86400
    if days_until_expiry <= 30:
        return "short_term"
    if days_until_expiry <= 180:
        return "medium_term"
    return "long_term"


def _infer_geo_scope(text: str) -> str:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ("u.s.", "united states", "american", "fed", "fomc", "congress", "white house")):
        return "us"
    if any(keyword in lowered for keyword in ("global", "worldwide", "opec", "nato")):
        return "global"
    if any(keyword in lowered for keyword in ("china", "russia", "ukraine", "europe", "eu", "uk", "israel", "gaza", "iran", "japan")):
        return "international"
    return "unknown"


def map_risk_category(question: str | None, category_raw: str | None = None) -> tuple[str, str]:
    text = " ".join(part for part in [question or "", category_raw or ""] if part).lower()

    for category, keywords in RISK_CATEGORY_KEYWORDS.items():
        matched = [keyword for keyword in keywords if keyword in text]
        if matched:
            return category, "|".join(matched)

    return "geopolitical", ""


def _build_basis_risk_note(source: str) -> str:
    # TODO: Enrich this with user-specific exposure metadata and better basis-risk scoring.
    return (
        f"{source} contract mapped with lightweight keyword heuristics; "
        "confirm settlement rules against the actual exposure being hedged."
    )


def _build_kalshi_url(contract_id: str | None) -> str | None:
    if not contract_id:
        return None
    return f"https://api.elections.kalshi.com/trade-api/v2/markets/{contract_id}"


def _build_polymarket_url(contract_id: str | None, slug: str | None) -> str | None:
    if slug:
        return f"https://polymarket.com/event/{slug}"
    if contract_id:
        return f"https://gamma-api.polymarket.com/markets/{contract_id}"
    return None


def _extract_polymarket_prices(market: dict[str, Any]) -> tuple[float | None, float | None]:
    outcomes = market.get("outcomes") or []
    prices = market.get("outcomePrices") or []

    if not isinstance(outcomes, list):
        outcomes = []
    if not isinstance(prices, list):
        prices = []

    normalized_prices = [_normalize_probability(price) for price in prices]
    pairs = list(zip(outcomes, normalized_prices))
    outcome_lookup = {
        str(outcome).strip().lower(): price
        for outcome, price in pairs
        if outcome is not None and price is not None
    }

    yes_price = outcome_lookup.get("yes")
    no_price = outcome_lookup.get("no")

    if yes_price is None and normalized_prices:
        yes_price = normalized_prices[0]
    if no_price is None and len(normalized_prices) > 1:
        no_price = normalized_prices[1]

    if yes_price is None and no_price is not None:
        yes_price = round(1 - no_price, 4)
    if no_price is None and yes_price is not None:
        no_price = round(1 - yes_price, 4)

    return yes_price, no_price


def _kalshi_market_to_contract(
    market: dict[str, Any],
    event: dict[str, Any] | None = None,
) -> NormalizedContract:
    question = market.get("title") or (event or {}).get("title")
    category_raw = (event or {}).get("category")
    risk_category, risk_tags = map_risk_category(question, category_raw)

    yes_price = _midpoint(market.get("yes_bid_dollars"), market.get("yes_ask_dollars"))
    no_price = _midpoint(market.get("no_bid_dollars"), market.get("no_ask_dollars"))
    last_price = _normalize_probability(market.get("last_price_dollars"))

    if yes_price is None:
        yes_price = last_price
    if no_price is None and yes_price is not None:
        no_price = round(1 - yes_price, 4)
    if yes_price is None and no_price is not None:
        yes_price = round(1 - no_price, 4)

    status = str(market.get("status") or "").lower()
    active = status not in {"closed", "determined", "disputed", "finalized", "settled"}

    combined_text = " ".join(part for part in [question or "", category_raw or ""] if part)

    return {
        "source": "kalshi",
        "contract_id": market.get("ticker"),
        "event_id": market.get("event_ticker") or (event or {}).get("event_ticker"),
        "question": question,
        "category_raw": category_raw,
        "risk_category": risk_category,
        "risk_tags": risk_tags,
        "implied_probability": yes_price,
        "yes_price": yes_price,
        "no_price": no_price,
        "volume": _to_float(market.get("volume_fp")),
        "liquidity": _to_float(market.get("liquidity_dollars")),
        "expiration_ts": _coerce_timestamp(
            market.get("expiration_time")
            or market.get("settlement_ts")
            or market.get("close_time")
        ),
        "active": active,
        "tradable": active,
        "geo_scope": _infer_geo_scope(combined_text),
        "time_horizon": _infer_time_horizon(
            market.get("expiration_time")
            or market.get("settlement_ts")
            or market.get("close_time")
        ),
        "basis_risk_notes": _build_basis_risk_note("Kalshi"),
        "url": _build_kalshi_url(market.get("ticker")),
    }


def _polymarket_market_to_contract(
    market: dict[str, Any],
    event: dict[str, Any] | None = None,
) -> NormalizedContract:
    question = market.get("question") or market.get("title") or (event or {}).get("title")

    linked_events = market.get("events")
    linked_event = linked_events[0] if isinstance(linked_events, list) and linked_events else {}
    if not isinstance(linked_event, dict):
        linked_event = {}

    category_raw = (
        market.get("category")
        or (event or {}).get("category")
        or linked_event.get("category")
        or (event or {}).get("subcategory")
    )
    risk_category, risk_tags = map_risk_category(question, category_raw)

    yes_price, no_price = _extract_polymarket_prices(market)
    active = bool(market.get("active")) and not bool(market.get("closed")) and not bool(market.get("archived"))

    tradable = active and not bool(market.get("restricted"))
    if market.get("acceptingOrders") is not None:
        tradable = tradable and bool(market.get("acceptingOrders"))
    elif market.get("enableOrderBook") is not None:
        tradable = tradable and bool(market.get("enableOrderBook"))

    combined_text = " ".join(part for part in [question or "", category_raw or ""] if part)

    return {
        "source": "polymarket",
        "contract_id": market.get("id") or market.get("conditionId"),
        "event_id": (event or {}).get("id") or linked_event.get("id"),
        "question": question,
        "category_raw": category_raw,
        "risk_category": risk_category,
        "risk_tags": risk_tags,
        "implied_probability": yes_price,
        "yes_price": yes_price,
        "no_price": no_price,
        "volume": _to_float(market.get("volumeNum") or market.get("volume")),
        "liquidity": _to_float(market.get("liquidityNum") or market.get("liquidity")),
        "expiration_ts": _coerce_timestamp(market.get("endDateIso") or market.get("endDate")),
        "active": active,
        "tradable": tradable,
        "geo_scope": _infer_geo_scope(combined_text),
        "time_horizon": _infer_time_horizon(market.get("endDateIso") or market.get("endDate")),
        "basis_risk_notes": _build_basis_risk_note("Polymarket"),
        "url": _build_polymarket_url(
            market.get("id") or market.get("conditionId"),
            market.get("slug") or (event or {}).get("slug"),
        ),
    }


def normalize_kalshi_markets(raw_data: Any) -> list[dict[str, Any]]:
    events = _unwrap_collection(raw_data, "events")
    normalized: list[dict[str, Any]] = []

    if events and any("markets" in event for event in events):
        for event in events:
            markets = event.get("markets", [])
            if not isinstance(markets, list):
                continue
            normalized.extend(
                _kalshi_market_to_contract(market, event)
                for market in markets
                if isinstance(market, dict)
            )
        return normalized

    markets = _unwrap_collection(raw_data, "markets")
    return [
        _kalshi_market_to_contract(market)
        for market in markets
        if isinstance(market, dict)
    ]


def normalize_polymarket_markets(raw_data: Any) -> list[dict[str, Any]]:
    events = _unwrap_collection(raw_data, "events")
    normalized: list[dict[str, Any]] = []

    if events and any("markets" in event for event in events):
        for event in events:
            markets = event.get("markets", [])
            if not isinstance(markets, list):
                continue
            normalized.extend(
                _polymarket_market_to_contract(market, event)
                for market in markets
                if isinstance(market, dict)
            )
        return normalized

    markets = _unwrap_collection(raw_data, "markets")
    return [
        _polymarket_market_to_contract(market)
        for market in markets
        if isinstance(market, dict)
    ]
