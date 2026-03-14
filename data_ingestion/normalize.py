from __future__ import annotations

import json
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
    "snapshot_ts",
    "as_of_ts",
]


CONTRACT_ENTITY_FIELDS = [
    "source",
    "contract_id",
    "event_id",
    "question",
    "description",
    "resolution_source",
    "category_raw",
    "risk_category",
    "risk_tags",
    "market_type",
    "subject_key",
    "metric_key",
    "comparator",
    "threshold_value",
    "threshold_unit",
    "region_key",
    "window_start_ts",
    "window_end_ts",
    "expiration_ts",
    "geo_scope",
    "time_horizon",
    "basis_risk_notes",
    "url",
    "first_seen_ts",
    "last_seen_ts",
]


CONTRACT_SNAPSHOT_FIELDS = [
    "source",
    "contract_id",
    "as_of_date",
    "as_of_ts",
    "snapshot_ts",
    "observation_method",
    "implied_probability",
    "yes_price",
    "no_price",
    "volume",
    "liquidity",
    "open_interest",
    "active",
    "tradable",
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
    snapshot_ts: str | None
    as_of_ts: str | None


class ContractEntity(TypedDict):
    source: str
    contract_id: str | None
    event_id: str | None
    question: str | None
    description: str | None
    resolution_source: str | None
    category_raw: str | None
    risk_category: str
    risk_tags: str
    market_type: str
    subject_key: str | None
    metric_key: str | None
    comparator: str | None
    threshold_value: str | None
    threshold_unit: str | None
    region_key: str
    window_start_ts: str | None
    window_end_ts: str | None
    expiration_ts: str | None
    geo_scope: str
    time_horizon: str
    basis_risk_notes: str
    url: str | None
    first_seen_ts: str
    last_seen_ts: str


class ContractSnapshot(TypedDict):
    source: str
    contract_id: str | None
    as_of_date: str | None
    as_of_ts: str | None
    snapshot_ts: str
    observation_method: str
    implied_probability: float | None
    yes_price: float | None
    no_price: float | None
    volume: float | None
    liquidity: float | None
    open_interest: float | None
    active: bool
    tradable: bool


RISK_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "inflation": ("inflation", "cpi", "pce", "ppi"),
    "interest_rate": ("interest rate", "rate cut", "rate hike", "fed", "fomc", "treasury yield"),
    "tariff": ("tariff", "duties", "import tax", "trade war"),
    "recession": ("recession", "gdp contraction", "economic downturn"),
    "labor_market": ("jobs report", "nonfarm payroll", "payrolls", "jobless claims", "unemployment", "labor market"),
    "hurricane": ("hurricane", "tropical storm", "storm surge"),
    "weather": ("weather", "temperature", "rainfall", "snow", "heat wave", "blizzard"),
    "pandemic_health": ("pandemic", "covid", "flu", "outbreak", "public health", "cdc"),
    "equity_market": ("s&p", "nasdaq", "dow", "stock market", "equity", "spy", "qqq", "russell"),
    "geopolitical": ("election", "president", "congress", "war", "ceasefire", "nato", "china", "russia", "ukraine", "israel", "gaza", "iran"),
}

ZERO_TIMESTAMP_VALUES = {
    "0001-01-01T00:00:00Z",
    "0001-01-01 00:00:00+00",
    "0001-01-01T00:00:00+00:00",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coalesce_text(*values: Any) -> str | None:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return None


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
        if not stripped or stripped in ZERO_TIMESTAMP_VALUES:
            return None
        if stripped.isdigit():
            return datetime.fromtimestamp(int(stripped), tz=timezone.utc).isoformat()
        return stripped
    return None


def parse_timestamp(value: Any) -> datetime | None:
    normalized = _coerce_timestamp(value)
    if not normalized:
        return None

    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed


def _first_timestamp(*values: Any) -> str | None:
    for value in values:
        normalized = _coerce_timestamp(value)
        if normalized:
            return normalized
    return None


def _timestamp_to_date(value: Any) -> str | None:
    parsed = parse_timestamp(value)
    if not parsed:
        return None
    return parsed.astimezone(timezone.utc).date().isoformat()


def _infer_time_horizon(expiration_ts: Any) -> str:
    expiration_dt = parse_timestamp(expiration_ts)
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


def _extract_polymarket_token_ids(market: dict[str, Any]) -> list[str]:
    raw_value = market.get("clobTokenIds")
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(token) for token in raw_value if str(token).strip()]
    if not isinstance(raw_value, str):
        return [str(raw_value)]

    raw_value = raw_value.strip()
    if not raw_value:
        return []

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return [item.strip() for item in raw_value.split(",") if item.strip()]

    if isinstance(parsed, list):
        return [str(token) for token in parsed if str(token).strip()]
    return [str(parsed)]


def _infer_market_type(raw_market_type: Any, question: str | None, outcomes: list[Any] | None = None) -> str:
    lowered_type = str(raw_market_type or "").strip().lower()
    lowered_question = (question or "").strip().lower()
    outcome_count = len(outcomes or [])

    if lowered_type in {"multi", "multi_outcome"}:
        return "multi_outcome"
    if outcome_count not in {0, 2}:
        return "multi_outcome"
    if lowered_question.startswith(("who will", "which ", "what ")) and "?" in lowered_question:
        return "multi_outcome"
    return "binary_yes_no"


def _combine_rules(primary: Any, secondary: Any) -> str | None:
    rule_parts = [part for part in [_clean_text(primary), _clean_text(secondary)] if part]
    if not rule_parts:
        return None
    return "\n\n".join(rule_parts)


def _kalshi_market_to_record(
    market: dict[str, Any],
    event: dict[str, Any] | None = None,
    snapshot_ts: str | None = None,
) -> dict[str, Any]:
    snapshot_ts = snapshot_ts or utc_now_iso()
    question = _coalesce_text(market.get("title"), (event or {}).get("title"))
    description = _combine_rules(market.get("rules_primary"), market.get("rules_secondary"))
    resolution_source = None
    category_raw = _clean_text((event or {}).get("category"))
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

    expiration_ts = _first_timestamp(
        market.get("expiration_time"),
        market.get("settlement_ts"),
        market.get("close_time"),
    )
    as_of_ts = _first_timestamp(
        market.get("updated_time"),
        (event or {}).get("last_updated_ts"),
        snapshot_ts,
    )
    combined_text = " ".join(
        part for part in [question or "", category_raw or "", description or ""] if part
    )

    return {
        "source": "kalshi",
        "contract_id": _clean_text(market.get("ticker")),
        "event_id": _coalesce_text(market.get("event_ticker"), (event or {}).get("event_ticker")),
        "question": question,
        "description": description,
        "resolution_source": resolution_source,
        "category_raw": category_raw,
        "risk_category": risk_category,
        "risk_tags": risk_tags,
        "implied_probability": yes_price,
        "yes_price": yes_price,
        "no_price": no_price,
        "volume": _to_float(market.get("volume_fp")),
        "liquidity": _to_float(market.get("liquidity_dollars")),
        "open_interest": _to_float(market.get("open_interest_fp")),
        "expiration_ts": expiration_ts,
        "active": active,
        "tradable": active,
        "geo_scope": _infer_geo_scope(combined_text),
        "time_horizon": _infer_time_horizon(expiration_ts),
        "basis_risk_notes": _build_basis_risk_note("Kalshi"),
        "url": _build_kalshi_url(_clean_text(market.get("ticker"))),
        "snapshot_ts": snapshot_ts,
        "as_of_ts": as_of_ts,
        "as_of_date": _timestamp_to_date(as_of_ts or snapshot_ts),
        "observation_method": "daily_close",
        "market_type": _infer_market_type(market.get("market_type"), question),
    }


def _polymarket_market_to_record(
    market: dict[str, Any],
    event: dict[str, Any] | None = None,
    snapshot_ts: str | None = None,
) -> dict[str, Any]:
    snapshot_ts = snapshot_ts or utc_now_iso()
    question = _coalesce_text(market.get("question"), market.get("title"), (event or {}).get("title"))

    linked_events = market.get("events")
    linked_event = linked_events[0] if isinstance(linked_events, list) and linked_events else {}
    if not isinstance(linked_event, dict):
        linked_event = {}

    description = _coalesce_text(market.get("description"), (event or {}).get("description"))
    resolution_source = _coalesce_text(
        market.get("resolutionSource"),
        (event or {}).get("resolutionSource"),
        linked_event.get("resolutionSource"),
    )
    category_raw = _coalesce_text(
        market.get("category"),
        (event or {}).get("category"),
        linked_event.get("category"),
        (event or {}).get("subcategory"),
    )
    risk_category, risk_tags = map_risk_category(question, category_raw)

    yes_price, no_price = _extract_polymarket_prices(market)
    active = bool(market.get("active")) and not bool(market.get("closed")) and not bool(market.get("archived"))

    tradable = active and not bool(market.get("restricted"))
    if market.get("acceptingOrders") is not None:
        tradable = tradable and bool(market.get("acceptingOrders"))
    elif market.get("enableOrderBook") is not None:
        tradable = tradable and bool(market.get("enableOrderBook"))

    expiration_ts = _first_timestamp(market.get("endDate"), market.get("endDateIso"))
    as_of_ts = _first_timestamp(
        market.get("updatedAt"),
        (event or {}).get("updatedAt"),
        linked_event.get("updatedAt"),
        snapshot_ts,
    )
    combined_text = " ".join(
        part for part in [question or "", category_raw or "", description or "", resolution_source or ""] if part
    )
    token_ids = _extract_polymarket_token_ids(market)

    return {
        "source": "polymarket",
        "contract_id": _coalesce_text(market.get("id"), market.get("conditionId")),
        "event_id": _coalesce_text((event or {}).get("id"), linked_event.get("id")),
        "question": question,
        "description": description,
        "resolution_source": resolution_source,
        "category_raw": category_raw,
        "risk_category": risk_category,
        "risk_tags": risk_tags,
        "implied_probability": yes_price,
        "yes_price": yes_price,
        "no_price": no_price,
        "volume": _to_float(market.get("volumeNum") or market.get("volume")),
        "liquidity": _to_float(market.get("liquidityNum") or market.get("liquidity")),
        "open_interest": _to_float(market.get("openInterest") or (event or {}).get("openInterest")),
        "expiration_ts": expiration_ts,
        "active": active,
        "tradable": tradable,
        "geo_scope": _infer_geo_scope(combined_text),
        "time_horizon": _infer_time_horizon(expiration_ts),
        "basis_risk_notes": _build_basis_risk_note("Polymarket"),
        "url": _build_polymarket_url(
            _coalesce_text(market.get("id"), market.get("conditionId")),
            _coalesce_text(market.get("slug"), (event or {}).get("slug")),
        ),
        "snapshot_ts": snapshot_ts,
        "as_of_ts": as_of_ts,
        "as_of_date": _timestamp_to_date(as_of_ts or snapshot_ts),
        "observation_method": "daily_close",
        "market_type": _infer_market_type(market.get("market_type"), question, market.get("outcomes")),
        "yes_token_id": token_ids[0] if token_ids else None,
    }


def normalize_kalshi_markets(
    raw_data: Any,
    snapshot_ts: str | None = None,
) -> list[dict[str, Any]]:
    events = _unwrap_collection(raw_data, "events")
    normalized: list[dict[str, Any]] = []

    if events and any("markets" in event for event in events):
        for event in events:
            markets = event.get("markets", [])
            if not isinstance(markets, list):
                continue
            normalized.extend(
                _kalshi_market_to_record(market, event, snapshot_ts=snapshot_ts)
                for market in markets
                if isinstance(market, dict)
            )
        return normalized

    markets = _unwrap_collection(raw_data, "markets")
    return [
        _kalshi_market_to_record(market, snapshot_ts=snapshot_ts)
        for market in markets
        if isinstance(market, dict)
    ]


def normalize_polymarket_markets(
    raw_data: Any,
    snapshot_ts: str | None = None,
) -> list[dict[str, Any]]:
    events = _unwrap_collection(raw_data, "events")
    normalized: list[dict[str, Any]] = []

    if events and any("markets" in event for event in events):
        for event in events:
            markets = event.get("markets", [])
            if not isinstance(markets, list):
                continue
            normalized.extend(
                _polymarket_market_to_record(market, event, snapshot_ts=snapshot_ts)
                for market in markets
                if isinstance(market, dict)
            )
        return normalized

    markets = _unwrap_collection(raw_data, "markets")
    return [
        _polymarket_market_to_record(market, snapshot_ts=snapshot_ts)
        for market in markets
        if isinstance(market, dict)
    ]
