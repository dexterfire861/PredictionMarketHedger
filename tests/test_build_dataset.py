from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data_ingestion import build_dataset as dataset_module
from data_ingestion.normalize import UNIFIED_CONTRACT_FIELDS, normalize_polymarket_markets


KALSHI_RAW = [
    {
        "event_ticker": "KXINFLATION-25",
        "title": "Will CPI stay above 3% this month?",
        "category": "Macro",
        "markets": [
            {
                "ticker": "KXINFLATION-25.YES",
                "event_ticker": "KXINFLATION-25",
                "title": "Will CPI stay above 3% this month?",
                "yes_bid_dollars": 0.56,
                "yes_ask_dollars": 0.58,
                "no_bid_dollars": 0.42,
                "no_ask_dollars": 0.44,
                "volume_fp": 1450,
                "liquidity_dollars": 6000,
                "expiration_time": "2099-03-31T12:00:00Z",
                "status": "open",
            }
        ],
    },
    {
        "event_ticker": "KXTARIFF-25",
        "title": "Will the U.S. announce new tariffs?",
        "category": "Trade",
        "markets": [
            {
                "ticker": "KXTARIFF-25.YES",
                "event_ticker": "KXTARIFF-25",
                "title": "Will the U.S. announce new tariffs?",
                "yes_bid_dollars": 0.31,
                "yes_ask_dollars": 0.33,
                "no_bid_dollars": 0.67,
                "no_ask_dollars": 0.69,
                "volume_fp": 830,
                "liquidity_dollars": 3200,
                "expiration_time": "2099-09-15T12:00:00Z",
                "status": "open",
            }
        ],
    },
]

POLYMARKET_RAW = [
    {
        "id": "pm-event-1",
        "title": "Tariff and CPI event",
        "slug": "tariff-cpi-event",
        "category": "Macro",
        "markets": [
            {
                "id": "pm-1",
                "conditionId": "cond-1",
                "question": "Will CPI rise because of tariffs?",
                "slug": "cpi-rise-because-of-tariffs",
                "outcomes": ["Yes", "No"],
                "outcomePrices": ["0.62", "0.38"],
                "volume": "1200",
                "liquidity": "800",
                "active": True,
                "closed": False,
                "archived": False,
                "restricted": False,
                "enableOrderBook": True,
                "endDateIso": "2099-12-31",
            }
        ],
    },
    {
        "id": "pm-event-2",
        "title": "Global weather event",
        "slug": "global-weather-event",
        "category": "Climate",
        "markets": [
            {
                "id": "pm-2",
                "conditionId": "cond-2",
                "question": "Will a hurricane hit Florida this season?",
                "slug": "florida-hurricane-season",
                "outcomes": ["Yes", "No"],
                "outcomePrices": ["0.21", "0.79"],
                "volume": "950",
                "liquidity": "450",
                "active": True,
                "closed": False,
                "archived": False,
                "restricted": False,
                "enableOrderBook": True,
                "endDateIso": "2099-08-31",
            }
        ],
    },
]


def make_contract_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=UNIFIED_CONTRACT_FIELDS)


class BuildDatasetTests(unittest.TestCase):
    def with_temp_data_dir(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return Path(temp_dir.name)

    def test_build_contracts_dataset_without_request_writes_full_dataset(self) -> None:
        data_dir = self.with_temp_data_dir()

        with (
            patch.object(dataset_module, "DATA_DIR", data_dir),
            patch.object(dataset_module, "fetch_open_events", return_value=KALSHI_RAW),
            patch.object(dataset_module, "fetch_active_events", return_value=POLYMARKET_RAW),
        ):
            full_df, filtered_df = dataset_module.build_contracts_dataset()

        self.assertIsNone(filtered_df)
        self.assertEqual(set(full_df["source"]), {"kalshi", "polymarket"})
        self.assertTrue((data_dir / "contracts.csv").exists())
        self.assertTrue((data_dir / "kalshi_raw.json").exists())
        self.assertTrue((data_dir / "polymarket_raw.json").exists())

    def test_build_contracts_dataset_with_kalshi_only_skips_polymarket_fetch(self) -> None:
        data_dir = self.with_temp_data_dir()
        risk_request: dataset_module.RiskRequest = {
            "query": "inflation hedge",
            "risk_categories": ["inflation"],
            "keywords": ["cpi"],
            "geo_scope": "us",
            "time_horizon": "long_term",
            "sources": ["kalshi"],
            "only_active": True,
            "max_results": 10,
        }

        with (
            patch.object(dataset_module, "DATA_DIR", data_dir),
            patch.object(dataset_module, "fetch_open_events", return_value=KALSHI_RAW) as kalshi_mock,
            patch.object(dataset_module, "fetch_active_events", return_value=POLYMARKET_RAW) as polymarket_mock,
        ):
            full_df, filtered_df = dataset_module.build_contracts_dataset(risk_request)

        kalshi_mock.assert_called_once()
        polymarket_mock.assert_not_called()
        self.assertEqual(set(full_df["source"]), {"kalshi"})
        self.assertIsNotNone(filtered_df)
        self.assertEqual(set(filtered_df["source"]), {"kalshi"})

    def test_filter_contracts_matches_risk_category_or_keyword(self) -> None:
        df = make_contract_df(
            [
                {
                    "source": "kalshi",
                    "contract_id": "k1",
                    "event_id": "e1",
                    "question": "Will tariffs increase next month?",
                    "category_raw": "Trade",
                    "risk_category": "tariff",
                    "risk_tags": "tariff",
                    "implied_probability": 0.55,
                    "yes_price": 0.55,
                    "no_price": 0.45,
                    "volume": 100,
                    "liquidity": 200,
                    "expiration_ts": "2099-06-01T00:00:00Z",
                    "active": True,
                    "tradable": True,
                    "geo_scope": "us",
                    "time_horizon": "short_term",
                    "basis_risk_notes": "note",
                    "url": "https://example.com/k1",
                },
                {
                    "source": "polymarket",
                    "contract_id": "p1",
                    "event_id": "e2",
                    "question": "Will CPI print above expectations?",
                    "category_raw": "Macro",
                    "risk_category": "inflation",
                    "risk_tags": "cpi",
                    "implied_probability": 0.61,
                    "yes_price": 0.61,
                    "no_price": 0.39,
                    "volume": 150,
                    "liquidity": 250,
                    "expiration_ts": "2099-07-01T00:00:00Z",
                    "active": True,
                    "tradable": True,
                    "geo_scope": "us",
                    "time_horizon": "short_term",
                    "basis_risk_notes": "note",
                    "url": "https://example.com/p1",
                },
                {
                    "source": "kalshi",
                    "contract_id": "k2",
                    "event_id": "e3",
                    "question": "Will unemployment rise?",
                    "category_raw": "Jobs",
                    "risk_category": "labor_market",
                    "risk_tags": "unemployment",
                    "implied_probability": 0.32,
                    "yes_price": 0.32,
                    "no_price": 0.68,
                    "volume": 75,
                    "liquidity": 150,
                    "expiration_ts": "2099-08-01T00:00:00Z",
                    "active": True,
                    "tradable": True,
                    "geo_scope": "us",
                    "time_horizon": "short_term",
                    "basis_risk_notes": "note",
                    "url": "https://example.com/k2",
                },
            ]
        )
        risk_request: dataset_module.RiskRequest = {
            "query": "",
            "risk_categories": ["tariff"],
            "keywords": ["cpi"],
            "geo_scope": None,
            "time_horizon": None,
            "sources": ["kalshi", "polymarket"],
            "only_active": True,
            "max_results": None,
        }

        filtered_df = dataset_module.filter_contracts_for_risk_request(df, risk_request)

        self.assertEqual(filtered_df["contract_id"].tolist(), ["k1", "p1"])

    def test_filter_contracts_keeps_unknown_geo_and_time_horizon(self) -> None:
        df = make_contract_df(
            [
                {
                    "source": "kalshi",
                    "contract_id": "k1",
                    "event_id": "e1",
                    "question": "US short term",
                    "category_raw": "Macro",
                    "risk_category": "inflation",
                    "risk_tags": "",
                    "implied_probability": 0.5,
                    "yes_price": 0.5,
                    "no_price": 0.5,
                    "volume": 1,
                    "liquidity": 1,
                    "expiration_ts": None,
                    "active": True,
                    "tradable": True,
                    "geo_scope": "us",
                    "time_horizon": "short_term",
                    "basis_risk_notes": "note",
                    "url": None,
                },
                {
                    "source": "kalshi",
                    "contract_id": "k2",
                    "event_id": "e2",
                    "question": "Unknown scope",
                    "category_raw": "Macro",
                    "risk_category": "inflation",
                    "risk_tags": "",
                    "implied_probability": 0.5,
                    "yes_price": 0.5,
                    "no_price": 0.5,
                    "volume": 1,
                    "liquidity": 1,
                    "expiration_ts": None,
                    "active": True,
                    "tradable": True,
                    "geo_scope": "unknown",
                    "time_horizon": "unknown",
                    "basis_risk_notes": "note",
                    "url": None,
                },
                {
                    "source": "kalshi",
                    "contract_id": "k3",
                    "event_id": "e3",
                    "question": "International medium term",
                    "category_raw": "Macro",
                    "risk_category": "inflation",
                    "risk_tags": "",
                    "implied_probability": 0.5,
                    "yes_price": 0.5,
                    "no_price": 0.5,
                    "volume": 1,
                    "liquidity": 1,
                    "expiration_ts": None,
                    "active": True,
                    "tradable": True,
                    "geo_scope": "international",
                    "time_horizon": "medium_term",
                    "basis_risk_notes": "note",
                    "url": None,
                },
            ]
        )
        risk_request: dataset_module.RiskRequest = {
            "query": "",
            "risk_categories": [],
            "keywords": [],
            "geo_scope": "us",
            "time_horizon": "short_term",
            "sources": ["kalshi"],
            "only_active": True,
            "max_results": None,
        }

        filtered_df = dataset_module.filter_contracts_for_risk_request(df, risk_request)

        self.assertEqual(filtered_df["contract_id"].tolist(), ["k1", "k2"])

    def test_filter_contracts_applies_max_results_without_reordering(self) -> None:
        df = make_contract_df(
            [
                {
                    "source": "kalshi",
                    "contract_id": "k1",
                    "event_id": "e1",
                    "question": "Question one",
                    "category_raw": "Macro",
                    "risk_category": "inflation",
                    "risk_tags": "",
                    "implied_probability": 0.4,
                    "yes_price": 0.4,
                    "no_price": 0.6,
                    "volume": 1,
                    "liquidity": 1,
                    "expiration_ts": None,
                    "active": True,
                    "tradable": True,
                    "geo_scope": "us",
                    "time_horizon": "short_term",
                    "basis_risk_notes": "note",
                    "url": None,
                },
                {
                    "source": "kalshi",
                    "contract_id": "k2",
                    "event_id": "e2",
                    "question": "Question two",
                    "category_raw": "Macro",
                    "risk_category": "inflation",
                    "risk_tags": "",
                    "implied_probability": 0.4,
                    "yes_price": 0.4,
                    "no_price": 0.6,
                    "volume": 1,
                    "liquidity": 1,
                    "expiration_ts": None,
                    "active": True,
                    "tradable": True,
                    "geo_scope": "us",
                    "time_horizon": "short_term",
                    "basis_risk_notes": "note",
                    "url": None,
                },
                {
                    "source": "kalshi",
                    "contract_id": "k3",
                    "event_id": "e3",
                    "question": "Question three",
                    "category_raw": "Macro",
                    "risk_category": "inflation",
                    "risk_tags": "",
                    "implied_probability": 0.4,
                    "yes_price": 0.4,
                    "no_price": 0.6,
                    "volume": 1,
                    "liquidity": 1,
                    "expiration_ts": None,
                    "active": True,
                    "tradable": True,
                    "geo_scope": "us",
                    "time_horizon": "short_term",
                    "basis_risk_notes": "note",
                    "url": None,
                },
            ]
        )
        risk_request: dataset_module.RiskRequest = {
            "query": "",
            "risk_categories": [],
            "keywords": [],
            "geo_scope": None,
            "time_horizon": None,
            "sources": ["kalshi"],
            "only_active": True,
            "max_results": 2,
        }

        filtered_df = dataset_module.filter_contracts_for_risk_request(df, risk_request)

        self.assertEqual(filtered_df["contract_id"].tolist(), ["k1", "k2"])

    def test_normalize_polymarket_markets_handles_date_only_end_date_iso(self) -> None:
        rows = normalize_polymarket_markets(POLYMARKET_RAW)

        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(rows[0]["expiration_ts"], "2099-12-31")
        self.assertIn(rows[0]["time_horizon"], {"short_term", "medium_term", "long_term"})

    def test_kalshi_only_execution_filter_excludes_polymarket_even_if_tradable(self) -> None:
        data_dir = self.with_temp_data_dir()

        with (
            patch.object(dataset_module, "DATA_DIR", data_dir),
            patch.object(dataset_module, "fetch_open_events", return_value=KALSHI_RAW),
            patch.object(dataset_module, "fetch_active_events", return_value=POLYMARKET_RAW),
        ):
            full_df, _ = dataset_module.build_contracts_dataset()

        polymarket_tradable = full_df[
            (full_df["source"] == "polymarket") & (full_df["tradable"] == True)  # noqa: E712
        ]
        executable_df = full_df[
            (full_df["source"] == "kalshi") & (full_df["tradable"] == True)  # noqa: E712
        ]

        self.assertFalse(polymarket_tradable.empty)
        self.assertEqual(set(executable_df["source"]), {"kalshi"})


if __name__ == "__main__":
    unittest.main()
