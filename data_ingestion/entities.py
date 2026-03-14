from __future__ import annotations

import re
from typing import Any

import pandas as pd

try:
    from .normalize import CONTRACT_ENTITY_FIELDS, utc_now_iso
except ImportError:
    from normalize import CONTRACT_ENTITY_FIELDS, utc_now_iso


STOPWORDS = {
    "will",
    "the",
    "a",
    "an",
    "be",
    "by",
    "before",
    "after",
    "in",
    "on",
    "of",
    "for",
    "to",
    "this",
    "that",
    "any",
    "its",
    "their",
    "if",
}

TIMEWORDS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "q1",
    "q2",
    "q3",
    "q4",
    "year",
    "month",
    "week",
    "season",
}


def _empty_entities_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CONTRACT_ENTITY_FIELDS)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _build_subject_fallback(question: str) -> str | None:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if token not in STOPWORDS and token not in TIMEWORDS and not token.isdigit()
    ]
    if not tokens:
        return None
    return "-".join(tokens[:8])


def _infer_market_type(question: str, record_market_type: str | None) -> str:
    lowered_question = question.lower().strip()
    if record_market_type == "multi_outcome":
        return "multi_outcome"
    if lowered_question.startswith(("who will", "which ", "what ")) and "?" in lowered_question:
        return "multi_outcome"
    return "binary_yes_no"


def _infer_subject_and_metric(question: str, description: str, risk_category: str | None) -> tuple[str | None, str | None]:
    text = f"{question} {description}".lower()

    patterns = [
        (("cpi",), ("cpi", "cpi")),
        (("pce",), ("inflation", "pce")),
        (("ppi",), ("inflation", "ppi")),
        (("fed", "rate"), ("fed", "interest_rate")),
        (("fomc",), ("fed", "interest_rate")),
        (("tariff",), ("tariff", "tariff_event")),
        (("recession",), ("recession", "recession_event")),
        (("unemployment",), ("labor", "unemployment")),
        (("payroll",), ("labor", "payrolls")),
        (("jobless", "claims"), ("labor", "jobless_claims")),
        (("hurricane",), ("hurricane", "storm_event")),
        (("microstrategy", "bitcoin", "sell"), ("microstrategy", "bitcoin_sale")),
        (("kraken", "ipo"), ("kraken", "ipo")),
        (("election",), ("election", "election_outcome")),
    ]

    for keywords, result in patterns:
        if all(keyword in text for keyword in keywords):
            return result

    if risk_category and risk_category != "geopolitical":
        return risk_category, f"{risk_category}_event"

    fallback = _build_subject_fallback(question)
    if fallback:
        return fallback, "binary_event"
    return None, None


def _infer_region_key(text: str, geo_scope: str) -> str:
    if geo_scope and geo_scope != "unknown":
        return geo_scope

    lowered = text.lower()
    if any(keyword in lowered for keyword in ("u.s.", "united states", "american", "federal reserve")):
        return "us"
    if any(keyword in lowered for keyword in ("global", "worldwide", "opec", "nato")):
        return "global"
    if any(keyword in lowered for keyword in ("china", "russia", "ukraine", "europe", "israel", "iran", "japan")):
        return "international"
    return "unknown"


def _parse_threshold(text: str) -> tuple[str | None, str | None, str | None]:
    lowered = text.lower()

    between_match = re.search(
        r"\bbetween\s+(-?\d+(?:\.\d+)?)\s*(%|percent|bps|basis points|points)?\s+and\s+(-?\d+(?:\.\d+)?)",
        lowered,
    )
    if between_match:
        unit = between_match.group(2) or ""
        threshold_value = f"{between_match.group(1)}|{between_match.group(3)}"
        return "between", threshold_value, unit.strip() or None

    comparator_patterns = [
        ("gte", r"\b(at least|greater than or equal to|no less than)\s+(-?\d+(?:\.\d+)?)\s*(%|percent|bps|basis points|points)?"),
        ("lte", r"\b(at most|less than or equal to|no more than)\s+(-?\d+(?:\.\d+)?)\s*(%|percent|bps|basis points|points)?"),
        ("gt", r"\b(above|over|greater than|more than)\s+(-?\d+(?:\.\d+)?)\s*(%|percent|bps|basis points|points)?"),
        ("lt", r"\b(below|under|less than)\s+(-?\d+(?:\.\d+)?)\s*(%|percent|bps|basis points|points)?"),
        ("eq", r"\b(exactly|equal to)\s+(-?\d+(?:\.\d+)?)\s*(%|percent|bps|basis points|points)?"),
    ]

    for comparator, pattern in comparator_patterns:
        match = re.search(pattern, lowered)
        if match:
            return comparator, match.group(2), (match.group(3) or "").strip() or None

    symbolic_patterns = [
        ("gte", r">=\s*(-?\d+(?:\.\d+)?)\s*(%|percent|bps|basis points|points)?"),
        ("lte", r"<=\s*(-?\d+(?:\.\d+)?)\s*(%|percent|bps|basis points|points)?"),
        ("gt", r">\s*(-?\d+(?:\.\d+)?)\s*(%|percent|bps|basis points|points)?"),
        ("lt", r"<\s*(-?\d+(?:\.\d+)?)\s*(%|percent|bps|basis points|points)?"),
    ]

    for comparator, pattern in symbolic_patterns:
        match = re.search(pattern, lowered)
        if match:
            return comparator, match.group(1), (match.group(2) or "").strip() or None

    return None, None, None


def _merge_entity_rows(existing_row: dict[str, Any] | None, new_row: dict[str, Any]) -> dict[str, Any]:
    if existing_row is None:
        return new_row

    merged = dict(existing_row)
    for column, value in new_row.items():
        if column in {"first_seen_ts", "last_seen_ts"}:
            continue
        if value not in (None, ""):
            merged[column] = value

    first_seen_candidates = [
        candidate
        for candidate in [
            _clean_text(existing_row.get("first_seen_ts")),
            _clean_text(new_row.get("first_seen_ts")),
        ]
        if candidate
    ]
    last_seen_candidates = [
        candidate
        for candidate in [
            _clean_text(existing_row.get("last_seen_ts")),
            _clean_text(new_row.get("last_seen_ts")),
        ]
        if candidate
    ]
    merged["first_seen_ts"] = min(first_seen_candidates) if first_seen_candidates else ""
    merged["last_seen_ts"] = max(last_seen_candidates) if last_seen_candidates else ""
    return merged


def build_contract_entities(
    records: list[dict[str, Any]],
    existing_entities_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    entity_rows: list[dict[str, Any]] = []

    for record in records:
        question = _clean_text(record.get("question")) or ""
        description = _clean_text(record.get("description")) or ""
        risk_category = _clean_text(record.get("risk_category"))
        combined_text = " ".join(
            part
            for part in [
                question,
                description,
                _clean_text(record.get("resolution_source")),
                _clean_text(record.get("category_raw")),
            ]
            if part
        )
        market_type = _infer_market_type(question, _clean_text(record.get("market_type")))
        subject_key, metric_key = _infer_subject_and_metric(question, description, risk_category)
        comparator, threshold_value, threshold_unit = _parse_threshold(combined_text)
        snapshot_ts = _clean_text(record.get("snapshot_ts")) or utc_now_iso()

        entity_rows.append(
            {
                "source": _clean_text(record.get("source")),
                "contract_id": record.get("contract_id"),
                "event_id": record.get("event_id"),
                "question": record.get("question"),
                "description": record.get("description"),
                "resolution_source": record.get("resolution_source"),
                "category_raw": record.get("category_raw"),
                "risk_category": record.get("risk_category"),
                "risk_tags": record.get("risk_tags"),
                "market_type": market_type,
                "subject_key": subject_key,
                "metric_key": metric_key,
                "comparator": comparator,
                "threshold_value": threshold_value,
                "threshold_unit": threshold_unit,
                "region_key": _infer_region_key(combined_text, _clean_text(record.get("geo_scope")) or "unknown"),
                "window_start_ts": None,
                "window_end_ts": record.get("expiration_ts"),
                "expiration_ts": record.get("expiration_ts"),
                "geo_scope": record.get("geo_scope"),
                "time_horizon": record.get("time_horizon"),
                "basis_risk_notes": record.get("basis_risk_notes"),
                "url": record.get("url"),
                "first_seen_ts": snapshot_ts,
                "last_seen_ts": snapshot_ts,
            }
        )

    existing_rows: dict[tuple[str, str], dict[str, Any]] = {}
    if existing_entities_df is not None and not existing_entities_df.empty:
        for row in existing_entities_df.to_dict(orient="records"):
            key = (_clean_text(row.get("source")), _clean_text(row.get("contract_id")))
            if key[0] and key[1]:
                existing_rows[(key[0], key[1])] = row

    merged_rows: dict[tuple[str, str], dict[str, Any]] = dict(existing_rows)
    for row in entity_rows:
        key = (_clean_text(row.get("source")), _clean_text(row.get("contract_id")))
        if not key[0] or not key[1]:
            continue
        merged_rows[(key[0], key[1])] = _merge_entity_rows(merged_rows.get((key[0], key[1])), row)

    if not merged_rows:
        return _empty_entities_df()

    entities_df = pd.DataFrame(merged_rows.values())
    for column in CONTRACT_ENTITY_FIELDS:
        if column not in entities_df.columns:
            entities_df[column] = None
    entities_df = entities_df[CONTRACT_ENTITY_FIELDS]
    entities_df = entities_df.sort_values(by=["source", "contract_id"], kind="stable").reset_index(drop=True)
    return entities_df
