from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import pandas as pd

try:
    from .kalshi import fetch_open_events, save_raw_kalshi_json
    from .normalize import UNIFIED_CONTRACT_FIELDS, normalize_kalshi_markets, normalize_polymarket_markets
    from .polymarket import fetch_active_events, save_raw_polymarket_json
except ImportError:
    from kalshi import fetch_open_events, save_raw_kalshi_json
    from normalize import UNIFIED_CONTRACT_FIELDS, normalize_kalshi_markets, normalize_polymarket_markets
    from polymarket import fetch_active_events, save_raw_polymarket_json


SourceName = Literal["kalshi", "polymarket"]
GeoScope = Literal["us", "international", "global", "unknown"]
TimeHorizon = Literal["short_term", "medium_term", "long_term", "unknown"]

DEFAULT_SOURCES: tuple[SourceName, ...] = ("kalshi", "polymarket")
VALID_SOURCES = set(DEFAULT_SOURCES)
VALID_GEO_SCOPES = {"us", "international", "global", "unknown"}
VALID_TIME_HORIZONS = {"short_term", "medium_term", "long_term", "unknown"}

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


class RiskRequest(TypedDict):
    query: str
    risk_categories: list[str]
    keywords: list[str]
    geo_scope: GeoScope | None
    time_horizon: TimeHorizon | None
    sources: list[SourceName]
    only_active: bool
    max_results: int | None


def _error_payload(source: str, error: Exception) -> dict[str, Any]:
    return {
        "source": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "error": str(error),
        "events": [],
    }


def _print_summary(df: pd.DataFrame) -> None:
    print("\nCounts by source:")
    if df.empty:
        print("  <empty>")
    else:
        print(df.groupby("source").size().to_string())

    print("\nCounts by risk_category:")
    if df.empty:
        print("  <empty>")
    else:
        print(df.groupby("risk_category").size().sort_values(ascending=False).to_string())


def _coerce_source_list(sources: list[str] | tuple[str, ...] | None) -> list[SourceName]:
    if not sources:
        return list(DEFAULT_SOURCES)

    normalized_sources = [str(source).strip().lower() for source in sources if str(source).strip()]
    invalid_sources = sorted({source for source in normalized_sources if source not in VALID_SOURCES})
    if invalid_sources:
        raise ValueError(f"Unsupported sources requested: {', '.join(invalid_sources)}")

    deduped_sources: list[SourceName] = []
    for source in normalized_sources:
        typed_source = cast(SourceName, source)
        if typed_source not in deduped_sources:
            deduped_sources.append(typed_source)

    return deduped_sources or list(DEFAULT_SOURCES)


def _validate_choice(value: str | None, valid_choices: set[str], field_name: str) -> str | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized not in valid_choices:
        raise ValueError(f"Unsupported {field_name}: {value}")
    return normalized


def _normalize_risk_request(risk_request: RiskRequest) -> RiskRequest:
    normalized_request: RiskRequest = {
        "query": str(risk_request.get("query", "") or "").strip(),
        "risk_categories": [
            str(category).strip().lower()
            for category in risk_request.get("risk_categories", [])
            if str(category).strip()
        ],
        "keywords": [
            str(keyword).strip().lower()
            for keyword in risk_request.get("keywords", [])
            if str(keyword).strip()
        ],
        "geo_scope": cast(
            GeoScope | None,
            _validate_choice(risk_request.get("geo_scope"), VALID_GEO_SCOPES, "geo_scope"),
        ),
        "time_horizon": cast(
            TimeHorizon | None,
            _validate_choice(
                risk_request.get("time_horizon"),
                VALID_TIME_HORIZONS,
                "time_horizon",
            ),
        ),
        "sources": _coerce_source_list(risk_request.get("sources")),
        "only_active": bool(risk_request.get("only_active", True)),
        "max_results": risk_request.get("max_results"),
    }

    if normalized_request["max_results"] is not None:
        max_results = int(normalized_request["max_results"])
        if max_results <= 0:
            raise ValueError("max_results must be a positive integer when provided")
        normalized_request["max_results"] = max_results

    return normalized_request


def _extract_query_tokens(query: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) >= 4]


def fetch_raw_data(sources: list[SourceName] | None = None) -> dict[str, Any]:
    requested_sources = _coerce_source_list(sources)
    raw_by_source: dict[str, Any] = {}

    if "kalshi" in requested_sources:
        try:
            raw_by_source["kalshi"] = fetch_open_events()
        except Exception as error:
            print(f"Kalshi fetch failed: {error}")
            raw_by_source["kalshi"] = _error_payload("kalshi", error)

    if "polymarket" in requested_sources:
        try:
            raw_by_source["polymarket"] = fetch_active_events()
        except Exception as error:
            print(f"Polymarket fetch failed: {error}")
            raw_by_source["polymarket"] = _error_payload("polymarket", error)

    return raw_by_source


def normalize_contracts(raw_by_source: dict[str, Any]) -> pd.DataFrame:
    all_contracts: list[dict[str, Any]] = []

    if "kalshi" in raw_by_source:
        all_contracts.extend(normalize_kalshi_markets(raw_by_source["kalshi"]))
    if "polymarket" in raw_by_source:
        all_contracts.extend(normalize_polymarket_markets(raw_by_source["polymarket"]))

    contracts_df = pd.DataFrame(all_contracts, columns=UNIFIED_CONTRACT_FIELDS)
    if contracts_df.empty:
        return pd.DataFrame(columns=UNIFIED_CONTRACT_FIELDS)
    return contracts_df


def filter_contracts_for_risk_request(
    df: pd.DataFrame,
    risk_request: RiskRequest,
) -> pd.DataFrame:
    normalized_request = _normalize_risk_request(risk_request)
    filtered_df = df.copy()

    requested_sources = normalized_request["sources"]
    filtered_df = filtered_df[filtered_df["source"].isin(requested_sources)]

    if normalized_request["only_active"]:
        filtered_df = filtered_df[filtered_df["active"] == True]  # noqa: E712

    if normalized_request["geo_scope"]:
        allowed_geo_scopes = {normalized_request["geo_scope"], "unknown"}
        filtered_df = filtered_df[filtered_df["geo_scope"].fillna("unknown").isin(allowed_geo_scopes)]

    if normalized_request["time_horizon"]:
        allowed_time_horizons = {normalized_request["time_horizon"], "unknown"}
        filtered_df = filtered_df[
            filtered_df["time_horizon"].fillna("unknown").isin(allowed_time_horizons)
        ]

    searchable_text = (
        filtered_df[["question", "category_raw", "risk_tags"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )

    requested_categories = set(normalized_request["risk_categories"])
    keyword_tokens = set(normalized_request["keywords"]) | set(_extract_query_tokens(normalized_request["query"]))

    if requested_categories or keyword_tokens:
        category_mask = filtered_df["risk_category"].fillna("").str.lower().isin(requested_categories)
        keyword_mask = searchable_text.apply(
            lambda text: any(token in text for token in keyword_tokens)
        )
        filtered_df = filtered_df[category_mask | keyword_mask]

    max_results = normalized_request["max_results"]
    if max_results is not None:
        filtered_df = filtered_df.head(max_results)

    return filtered_df.copy()


def build_contracts_dataset(
    risk_request: RiskRequest | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    normalized_request = _normalize_risk_request(risk_request) if risk_request is not None else None
    requested_sources = normalized_request["sources"] if normalized_request is not None else list(DEFAULT_SOURCES)

    raw_by_source = fetch_raw_data(requested_sources)

    if "kalshi" in raw_by_source:
        save_raw_kalshi_json(raw_by_source["kalshi"], DATA_DIR / "kalshi_raw.json")
    if "polymarket" in raw_by_source:
        save_raw_polymarket_json(raw_by_source["polymarket"], DATA_DIR / "polymarket_raw.json")

    contracts_df = normalize_contracts(raw_by_source)

    # TODO: Add consensus scoring across Kalshi and Polymarket once contract matching is in place.
    # TODO: Add basis risk enrichment from external macro metadata and user exposure inputs.
    # TODO: Load the normalized contract set into SQLite once the schema stabilizes.

    output_path = DATA_DIR / "contracts.csv"
    contracts_df.to_csv(output_path, index=False)

    filtered_df = None
    if normalized_request is not None:
        filtered_df = filter_contracts_for_risk_request(contracts_df, normalized_request)

    return contracts_df, filtered_df


def main() -> None:
    contracts_df, _ = build_contracts_dataset()
    output_path = DATA_DIR / "contracts.csv"

    print(f"Saved {len(contracts_df)} normalized contracts to {output_path}")
    _print_summary(contracts_df)


if __name__ == "__main__":
    main()
