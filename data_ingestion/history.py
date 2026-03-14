from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .kalshi import fetch_historical_market_candlesticks
    from .normalize import CONTRACT_SNAPSHOT_FIELDS, parse_timestamp
    from .polymarket import fetch_prices_history
except ImportError:
    from kalshi import fetch_historical_market_candlesticks
    from normalize import CONTRACT_SNAPSHOT_FIELDS, parse_timestamp
    from polymarket import fetch_prices_history


RAW_HISTORY_DIR = Path(__file__).resolve().parent / "data" / "raw_history"


def _empty_snapshot_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONTRACT_SNAPSHOT_FIELDS)


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


def _timestamp_to_iso(value: Any) -> str | None:
    parsed = parse_timestamp(value)
    if not parsed:
        return None
    return parsed.isoformat()


def _timestamp_to_date(value: Any) -> str | None:
    parsed = parse_timestamp(value)
    if not parsed:
        return None
    return parsed.date().isoformat()


def read_csv_table(path: str | Path, columns: list[str]) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(table_path)
    for column in columns:
        if column not in df.columns:
            df[column] = None
    return df[columns]


def write_csv_table(df: pd.DataFrame, path: str | Path, columns: list[str]) -> Path:
    table_path = Path(path)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    output_df = df.copy()
    for column in columns:
        if column not in output_df.columns:
            output_df[column] = None
    output_df = output_df[columns]
    output_df.to_csv(table_path, index=False)
    return table_path


def build_latest_snapshot_rows(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return _empty_snapshot_df()

    snapshots_df = pd.DataFrame(records)
    for column in CONTRACT_SNAPSHOT_FIELDS:
        if column not in snapshots_df.columns:
            snapshots_df[column] = None
    snapshots_df = snapshots_df[CONTRACT_SNAPSHOT_FIELDS]
    return dedupe_contract_snapshots(snapshots_df)


def dedupe_contract_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_snapshot_df()

    deduped_df = df.copy()
    for column in CONTRACT_SNAPSHOT_FIELDS:
        if column not in deduped_df.columns:
            deduped_df[column] = None

    deduped_df = deduped_df.sort_values(
        by=["source", "contract_id", "as_of_date", "snapshot_ts"],
        kind="stable",
    )
    deduped_df = deduped_df.drop_duplicates(
        subset=["source", "contract_id", "as_of_date"],
        keep="last",
    )
    deduped_df = deduped_df.reset_index(drop=True)
    return deduped_df[CONTRACT_SNAPSHOT_FIELDS]


def merge_contract_snapshots(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df.empty and new_df.empty:
        return _empty_snapshot_df()
    if existing_df.empty:
        return dedupe_contract_snapshots(new_df)
    if new_df.empty:
        return dedupe_contract_snapshots(existing_df)

    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    return dedupe_contract_snapshots(merged_df)


def save_raw_history_payload(
    source: str,
    contract_id: str,
    as_of_date: str,
    payload: Any,
    raw_history_dir: str | Path,
) -> Path:
    output_path = Path(raw_history_dir) / source / contract_id / f"{as_of_date}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    return output_path


def _extract_candlestick_points(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("candlesticks", "history", "data"):
            points = payload.get(key)
            if isinstance(points, list):
                return [point for point in points if isinstance(point, dict)]
    if isinstance(payload, list):
        return [point for point in payload if isinstance(point, dict)]
    return []


def _extract_price_history_points(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("history", "data", "points", "prices"):
            points = payload.get(key)
            if isinstance(points, list):
                return [point for point in points if isinstance(point, dict)]
    if isinstance(payload, list):
        return [point for point in payload if isinstance(point, dict)]
    return []


def _group_last_observation_by_day(
    points: list[dict[str, Any]],
    timestamp_keys: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    for point in points:
        timestamp_value = None
        for key in timestamp_keys:
            timestamp_value = _timestamp_to_iso(point.get(key))
            if timestamp_value:
                break
        if not timestamp_value:
            continue

        as_of_date = _timestamp_to_date(timestamp_value)
        if not as_of_date:
            continue

        current = grouped.get(as_of_date)
        if current is None or timestamp_value > current["_as_of_ts"]:
            grouped[as_of_date] = {"_as_of_ts": timestamp_value, "point": point}

    return grouped


def normalize_kalshi_candlesticks_to_snapshots(
    contract_id: str,
    payload: Any,
    snapshot_ts: str,
    active: bool = False,
    tradable: bool = False,
    liquidity: float | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for point in _extract_candlestick_points(payload):
        as_of_ts = _timestamp_to_iso(
            point.get("end_period_ts")
            or point.get("period_end")
            or point.get("close_time")
            or point.get("timestamp")
        )
        if not as_of_ts:
            continue

        price_value = (
            point.get("close_dollars")
            or point.get("close_price")
            or point.get("close")
            or (point.get("price") or {}).get("close")
            or point.get("last_price")
        )
        yes_price = _normalize_probability(price_value)
        no_price = round(1 - yes_price, 4) if yes_price is not None else None

        rows.append(
            {
                "source": "kalshi",
                "contract_id": contract_id,
                "as_of_date": _timestamp_to_date(as_of_ts),
                "as_of_ts": as_of_ts,
                "snapshot_ts": snapshot_ts,
                "observation_method": "daily_close",
                "implied_probability": yes_price,
                "yes_price": yes_price,
                "no_price": no_price,
                "volume": _to_float(point.get("volume_fp") or point.get("volume")),
                "liquidity": liquidity,
                "open_interest": _to_float(point.get("open_interest_fp") or point.get("open_interest")),
                "active": bool(active),
                "tradable": bool(tradable),
            }
        )

    return dedupe_contract_snapshots(pd.DataFrame(rows, columns=CONTRACT_SNAPSHOT_FIELDS))


def normalize_polymarket_price_history_to_snapshots(
    contract_id: str,
    payload: Any,
    snapshot_ts: str,
    active: bool = False,
    tradable: bool = False,
    volume: float | None = None,
    liquidity: float | None = None,
    open_interest: float | None = None,
) -> pd.DataFrame:
    grouped_points = _group_last_observation_by_day(
        _extract_price_history_points(payload),
        timestamp_keys=("t", "timestamp", "time", "updatedAt"),
    )
    rows: list[dict[str, Any]] = []

    for as_of_date in sorted(grouped_points):
        point = grouped_points[as_of_date]["point"]
        as_of_ts = grouped_points[as_of_date]["_as_of_ts"]
        price_value = point.get("p") or point.get("price") or point.get("value")
        yes_price = _normalize_probability(price_value)
        no_price = round(1 - yes_price, 4) if yes_price is not None else None

        rows.append(
            {
                "source": "polymarket",
                "contract_id": contract_id,
                "as_of_date": as_of_date,
                "as_of_ts": as_of_ts,
                "snapshot_ts": snapshot_ts,
                "observation_method": "derived_from_intraday",
                "implied_probability": yes_price,
                "yes_price": yes_price,
                "no_price": no_price,
                "volume": volume,
                "liquidity": liquidity,
                "open_interest": open_interest,
                "active": bool(active),
                "tradable": bool(tradable),
            }
        )

    return dedupe_contract_snapshots(pd.DataFrame(rows, columns=CONTRACT_SNAPSHOT_FIELDS))


def backfill_kalshi_daily_snapshots(
    markets: list[dict[str, Any]],
    start_ts: int,
    end_ts: int,
    snapshot_ts: str,
    raw_history_dir: str | Path,
) -> pd.DataFrame:
    snapshot_frames: list[pd.DataFrame] = []

    for market in markets:
        contract_id = str(market.get("contract_id") or "").strip()
        if not contract_id:
            continue

        payload = fetch_historical_market_candlesticks(
            ticker=contract_id,
            start_ts=start_ts,
            end_ts=end_ts,
            period_interval=1440,
        )
        daily_df = normalize_kalshi_candlesticks_to_snapshots(
            contract_id=contract_id,
            payload=payload,
            snapshot_ts=snapshot_ts,
            active=bool(market.get("active", False)),
            tradable=bool(market.get("tradable", False)),
            liquidity=_to_float(market.get("liquidity")),
        )

        for row in daily_df.to_dict(orient="records"):
            as_of_date = row.get("as_of_date")
            if not as_of_date:
                continue
            save_raw_history_payload(
                "kalshi",
                contract_id,
                as_of_date,
                {"candlesticks": _extract_candlestick_points(payload), "selected_row": row},
                raw_history_dir,
            )

        snapshot_frames.append(daily_df)

    if not snapshot_frames:
        return _empty_snapshot_df()
    return dedupe_contract_snapshots(pd.concat(snapshot_frames, ignore_index=True))


def backfill_polymarket_daily_snapshots(
    markets: list[dict[str, Any]],
    start_ts: int,
    end_ts: int,
    snapshot_ts: str,
    raw_history_dir: str | Path,
) -> pd.DataFrame:
    snapshot_frames: list[pd.DataFrame] = []

    for market in markets:
        contract_id = str(market.get("contract_id") or "").strip()
        yes_token_id = str(market.get("yes_token_id") or "").strip()
        if not contract_id or not yes_token_id:
            continue

        payload = fetch_prices_history(
            market_id=yes_token_id,
            start_ts=start_ts,
            end_ts=end_ts,
            interval="1m",
        )
        daily_df = normalize_polymarket_price_history_to_snapshots(
            contract_id=contract_id,
            payload=payload,
            snapshot_ts=snapshot_ts,
            active=bool(market.get("active", False)),
            tradable=bool(market.get("tradable", False)),
            volume=_to_float(market.get("volume")),
            liquidity=_to_float(market.get("liquidity")),
            open_interest=_to_float(market.get("open_interest")),
        )

        for row in daily_df.to_dict(orient="records"):
            as_of_date = row.get("as_of_date")
            if not as_of_date:
                continue
            save_raw_history_payload(
                "polymarket",
                contract_id,
                as_of_date,
                {"history": _extract_price_history_points(payload), "selected_row": row},
                raw_history_dir,
            )

        snapshot_frames.append(daily_df)

    if not snapshot_frames:
        return _empty_snapshot_df()
    return dedupe_contract_snapshots(pd.concat(snapshot_frames, ignore_index=True))
