from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data_ingestion import build_dataset as dataset_module
from data_ingestion import history as history_module
from data_ingestion.normalize import UNIFIED_CONTRACT_FIELDS, normalize_polymarket_markets


KALSHI_RAW = [
    {
        "event_ticker": "KX-CPI3-2099-06",
        "title": "US CPI above 3% in June 2099?",
        "category": "Macro",
        "last_updated_ts": "2099-06-30T23:59:00Z",
        "markets": [
            {
                "ticker": "KX-CPI3-2099-06",
                "event_ticker": "KX-CPI3-2099-06",
                "title": "Will US CPI be above 3% in June 2099?",
                "yes_bid_dollars": 0.56,
                "yes_ask_dollars": 0.58,
                "no_bid_dollars": 0.42,
                "no_ask_dollars": 0.44,
                "volume_fp": 1450,
                "liquidity_dollars": 6000,
                "open_interest_fp": 120,
                "expiration_time": "2099-07-01T00:00:00Z",
                "status": "active",
                "market_type": "binary",
                "updated_time": "2099-06-30T23:58:00Z",
                "rules_primary": "This market resolves to Yes if US CPI is above 3% for June 2099.",
            }
        ],
    },
    {
        "event_ticker": "KX-CPI4-2099-06",
        "title": "US CPI above 4% in June 2099?",
        "category": "Macro",
        "last_updated_ts": "2099-06-30T23:57:30Z",
        "markets": [
            {
                "ticker": "KX-CPI4-2099-06",
                "event_ticker": "KX-CPI4-2099-06",
                "title": "Will US CPI be above 4% in June 2099?",
                "yes_bid_dollars": 0.31,
                "yes_ask_dollars": 0.33,
                "no_bid_dollars": 0.67,
                "no_ask_dollars": 0.69,
                "volume_fp": 830,
                "liquidity_dollars": 3200,
                "open_interest_fp": 85,
                "expiration_time": "2099-07-01T00:00:00Z",
                "status": "active",
                "market_type": "binary",
                "updated_time": "2099-06-30T23:57:00Z",
                "rules_primary": "This market resolves to Yes if US CPI is above 4% for June 2099.",
            }
        ],
    },
    {
        "event_ticker": "KX-WINNER-2099",
        "title": "Who will win the 2099 election?",
        "category": "Politics",
        "last_updated_ts": "2099-11-04T23:58:00Z",
        "markets": [
            {
                "ticker": "KX-WINNER-2099-ALICE",
                "event_ticker": "KX-WINNER-2099",
                "title": "Who will win the 2099 election?",
                "yes_bid_dollars": 0.49,
                "yes_ask_dollars": 0.51,
                "no_bid_dollars": 0.49,
                "no_ask_dollars": 0.51,
                "volume_fp": 500,
                "liquidity_dollars": 1500,
                "open_interest_fp": 140,
                "expiration_time": "2099-11-05T00:00:00Z",
                "status": "active",
                "market_type": "binary",
                "updated_time": "2099-11-04T23:57:30Z",
                "rules_primary": "This market resolves to Yes if Alice wins the 2099 election.",
            }
        ],
    },
]

POLYMARKET_RAW = [
    {
        "id": "pm-cpi3-event",
        "title": "US CPI above 3% in June 2099?",
        "slug": "us-cpi-above-3-in-june-2099",
        "category": "Macro",
        "description": "This market resolves to Yes if US CPI is above 3% for June 2099.",
        "resolutionSource": "BLS",
        "updatedAt": "2099-06-30T23:59:30Z",
        "markets": [
            {
                "id": "pm-cpi3",
                "conditionId": "cond-cpi3",
                "question": "Will US CPI be above 3% in June 2099?",
                "slug": "us-cpi-above-3-in-june-2099",
                "description": "This market resolves to Yes if US CPI is above 3% for June 2099.",
                "resolutionSource": "BLS",
                "outcomes": ["Yes", "No"],
                "outcomePrices": ["0.57", "0.43"],
                "volume": "2000",
                "liquidity": "800",
                "openInterest": 90,
                "active": True,
                "closed": False,
                "archived": False,
                "restricted": False,
                "enableOrderBook": True,
                "endDate": "2099-07-01T00:00:00Z",
                "endDateIso": "2099-06-30",
                "updatedAt": "2099-06-30T23:59:00Z",
                "clobTokenIds": "[\"yes-cpi3\", \"no-cpi3\"]",
            }
        ],
    },
    {
        "id": "pm-cpi-target-event",
        "title": "US CPI above target in June 2099?",
        "slug": "us-cpi-above-target-in-june-2099",
        "category": "Macro",
        "description": "This market resolves to Yes if US CPI is above target for June 2099.",
        "resolutionSource": "BLS",
        "updatedAt": "2099-06-30T23:58:30Z",
        "markets": [
            {
                "id": "pm-cpi-target",
                "conditionId": "cond-cpi-target",
                "question": "Will US CPI be above target in June 2099?",
                "slug": "us-cpi-above-target-in-june-2099",
                "description": "This market resolves to Yes if US CPI is above target for June 2099.",
                "resolutionSource": "BLS",
                "outcomes": ["Yes", "No"],
                "outcomePrices": ["0.32", "0.68"],
                "volume": "1700",
                "liquidity": "550",
                "openInterest": 60,
                "active": True,
                "closed": False,
                "archived": False,
                "restricted": False,
                "enableOrderBook": True,
                "endDate": "2099-07-01T00:00:00Z",
                "endDateIso": "2099-06-30",
                "updatedAt": "2099-06-30T23:58:00Z",
                "clobTokenIds": "[\"yes-cpi-target\", \"no-cpi-target\"]",
            }
        ],
    },
    {
        "id": "pm-election-event",
        "title": "Who will win the 2099 election?",
        "slug": "who-will-win-the-2099-election",
        "category": "Politics",
        "description": "Winner market.",
        "updatedAt": "2099-11-04T23:59:30Z",
        "markets": [
            {
                "id": "pm-winner",
                "conditionId": "cond-winner",
                "question": "Who will win the 2099 election?",
                "slug": "who-will-win-the-2099-election",
                "description": "Winner market.",
                "outcomes": ["Alice", "Bob"],
                "outcomePrices": ["0.51", "0.49"],
                "volume": "900",
                "liquidity": "650",
                "openInterest": 110,
                "active": True,
                "closed": False,
                "archived": False,
                "restricted": False,
                "enableOrderBook": True,
                "endDate": "2099-11-05T00:00:00Z",
                "endDateIso": "2099-11-04",
                "updatedAt": "2099-11-04T23:59:00Z",
                "clobTokenIds": "[\"alice-token\", \"bob-token\"]",
            }
        ],
    },
]


def make_contract_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=UNIFIED_CONTRACT_FIELDS)


def make_contract_row(**overrides: object) -> dict[str, object]:
    row = {
        "source": "kalshi",
        "contract_id": "k-default",
        "event_id": "e-default",
        "question": "Default question",
        "category_raw": "Macro",
        "risk_category": "inflation",
        "risk_tags": "cpi",
        "implied_probability": 0.5,
        "yes_price": 0.5,
        "no_price": 0.5,
        "volume": 10.0,
        "liquidity": 20.0,
        "expiration_ts": "2099-07-01T00:00:00Z",
        "active": True,
        "tradable": True,
        "geo_scope": "us",
        "time_horizon": "short_term",
        "basis_risk_notes": "note",
        "url": "https://example.com/contract",
        "snapshot_ts": "2099-06-30T23:59:59Z",
        "as_of_ts": "2099-06-30T23:58:59Z",
    }
    row.update(overrides)
    return row


class BuildDatasetTests(unittest.TestCase):
    def with_temp_data_dir(self) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return Path(temp_dir.name)

    def patch_dataset_paths(self, data_dir: Path):
        return [
            patch.object(dataset_module, "DATA_DIR", data_dir),
            patch.object(dataset_module, "CONTRACTS_PATH", data_dir / "contracts.csv"),
            patch.object(dataset_module, "CONTRACT_ENTITIES_PATH", data_dir / "contract_entities.csv"),
            patch.object(dataset_module, "CONTRACT_SNAPSHOTS_PATH", data_dir / "contract_snapshots.csv"),
            patch.object(dataset_module, "RAW_HISTORY_DIR", data_dir / "raw_history"),
        ]

    def test_build_contracts_dataset_writes_latest_view_and_side_tables(self) -> None:
        data_dir = self.with_temp_data_dir()

        with ExitStack() as stack:
            for manager in self.patch_dataset_paths(data_dir):
                stack.enter_context(manager)
            stack.enter_context(patch.object(dataset_module, "fetch_open_events", return_value=KALSHI_RAW))
            stack.enter_context(patch.object(dataset_module, "fetch_active_events", return_value=POLYMARKET_RAW))
            stack.enter_context(
                patch.object(dataset_module, "utc_now_iso", return_value="2099-06-30T23:59:59+00:00")
            )
            full_df, filtered_df = dataset_module.build_contracts_dataset()

        self.assertIsNone(filtered_df)
        self.assertTrue((data_dir / "contracts.csv").exists())
        self.assertTrue((data_dir / "contract_entities.csv").exists())
        self.assertTrue((data_dir / "contract_snapshots.csv").exists())
        self.assertTrue((data_dir / "kalshi_raw.json").exists())
        self.assertTrue((data_dir / "polymarket_raw.json").exists())
        self.assertTrue(set(["snapshot_ts", "as_of_ts"]).issubset(full_df.columns))
        self.assertFalse(
            set(["linked_source", "linked_contract_id", "link_status"]).intersection(full_df.columns)
        )

    def test_contract_entities_remain_one_row_per_contract_and_preserve_first_seen(self) -> None:
        data_dir = self.with_temp_data_dir()

        with ExitStack() as stack:
            for manager in self.patch_dataset_paths(data_dir):
                stack.enter_context(manager)
            stack.enter_context(patch.object(dataset_module, "fetch_open_events", return_value=KALSHI_RAW))
            stack.enter_context(patch.object(dataset_module, "fetch_active_events", return_value=POLYMARKET_RAW))
            stack.enter_context(
                patch.object(
                    dataset_module,
                    "utc_now_iso",
                    side_effect=["2099-06-30T23:59:59+00:00", "2099-07-01T00:10:00+00:00"],
                )
            )
            dataset_module.build_contracts_dataset()
            dataset_module.build_contracts_dataset()

        entity_df = pd.read_csv(data_dir / "contract_entities.csv")
        self.assertEqual(len(entity_df), entity_df[["source", "contract_id"]].drop_duplicates().shape[0])

        cpi_entity = entity_df[entity_df["contract_id"] == "KX-CPI3-2099-06"].iloc[0]
        self.assertEqual(cpi_entity["first_seen_ts"], "2099-06-30T23:59:59+00:00")
        self.assertEqual(cpi_entity["last_seen_ts"], "2099-07-01T00:10:00+00:00")

    def test_contract_snapshots_deduplicate_by_day_and_keep_latest_snapshot_ts(self) -> None:
        data_dir = self.with_temp_data_dir()
        snapshot_path = data_dir / "contract_snapshots.csv"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "source": "kalshi",
                    "contract_id": "KX-CPI3-2099-06",
                    "as_of_date": "2099-06-30",
                    "as_of_ts": "2099-06-30T23:58:00+00:00",
                    "snapshot_ts": "2099-06-30T23:00:00+00:00",
                    "observation_method": "daily_close",
                    "implied_probability": 0.55,
                    "yes_price": 0.55,
                    "no_price": 0.45,
                    "volume": 100.0,
                    "liquidity": 200.0,
                    "open_interest": 50.0,
                    "active": True,
                    "tradable": True,
                }
            ]
        ).to_csv(snapshot_path, index=False)

        with ExitStack() as stack:
            for manager in self.patch_dataset_paths(data_dir):
                stack.enter_context(manager)
            stack.enter_context(patch.object(dataset_module, "fetch_open_events", return_value=KALSHI_RAW))
            stack.enter_context(patch.object(dataset_module, "fetch_active_events", return_value=[]))
            stack.enter_context(
                patch.object(dataset_module, "utc_now_iso", return_value="2099-06-30T23:59:59+00:00")
            )
            dataset_module.build_contracts_dataset(
                {
                    "query": "",
                    "risk_categories": [],
                    "keywords": [],
                    "geo_scope": None,
                    "time_horizon": None,
                    "sources": ["kalshi"],
                    "only_active": True,
                    "max_results": None,
                }
            )

        snapshot_df = pd.read_csv(snapshot_path)
        deduped = snapshot_df[snapshot_df["contract_id"] == "KX-CPI3-2099-06"]
        self.assertEqual(len(deduped[deduped["as_of_date"] == "2099-06-30"]), 1)
        self.assertEqual(
            deduped[deduped["as_of_date"] == "2099-06-30"]["snapshot_ts"].iloc[0],
            "2099-06-30T23:59:59+00:00",
        )

    def test_daily_snapshot_derivation_chooses_last_intraday_observation(self) -> None:
        payload = {
            "history": [
                {"t": "2099-05-31T10:00:00Z", "p": 0.25},
                {"t": "2099-05-31T23:50:00Z", "p": 0.41},
                {"t": "2099-06-01T00:10:00Z", "p": 0.39},
            ]
        }

        daily_df = history_module.normalize_polymarket_price_history_to_snapshots(
            contract_id="pm-cpi3",
            payload=payload,
            snapshot_ts="2099-06-30T23:59:59+00:00",
            active=True,
            tradable=True,
        )

        self.assertEqual(len(daily_df), 2)
        first_day = daily_df[daily_df["as_of_date"] == "2099-05-31"].iloc[0]
        self.assertEqual(first_day["yes_price"], 0.41)

    def test_build_contracts_dataset_removes_legacy_link_artifacts(self) -> None:
        data_dir = self.with_temp_data_dir()
        (data_dir / "exact_market_links.csv").write_text("stale\n", encoding="utf-8")
        (data_dir / "link_review_queue.csv").write_text("stale\n", encoding="utf-8")

        with ExitStack() as stack:
            for manager in self.patch_dataset_paths(data_dir):
                stack.enter_context(manager)
            stack.enter_context(patch.object(dataset_module, "fetch_open_events", return_value=KALSHI_RAW))
            stack.enter_context(patch.object(dataset_module, "fetch_active_events", return_value=POLYMARKET_RAW))
            stack.enter_context(
                patch.object(dataset_module, "utc_now_iso", return_value="2099-06-30T23:59:59+00:00")
            )
            dataset_module.build_contracts_dataset()

        self.assertFalse((data_dir / "exact_market_links.csv").exists())
        self.assertFalse((data_dir / "link_review_queue.csv").exists())

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

        with ExitStack() as stack:
            for manager in self.patch_dataset_paths(data_dir):
                stack.enter_context(manager)
            kalshi_mock = stack.enter_context(
                patch.object(dataset_module, "fetch_open_events", return_value=KALSHI_RAW)
            )
            polymarket_mock = stack.enter_context(
                patch.object(dataset_module, "fetch_active_events", return_value=POLYMARKET_RAW)
            )
            stack.enter_context(
                patch.object(dataset_module, "utc_now_iso", return_value="2099-06-30T23:59:59+00:00")
            )
            full_df, filtered_df = dataset_module.build_contracts_dataset(risk_request)

        kalshi_mock.assert_called_once()
        polymarket_mock.assert_not_called()
        self.assertEqual(set(full_df["source"]), {"kalshi"})
        self.assertIsNotNone(filtered_df)
        self.assertEqual(set(filtered_df["source"]), {"kalshi"})

    def test_filter_contracts_matches_risk_category_or_keyword(self) -> None:
        df = make_contract_df(
            [
                make_contract_row(
                    contract_id="k1",
                    question="Will tariffs increase next month?",
                    risk_category="tariff",
                    risk_tags="tariff",
                ),
                make_contract_row(
                    source="polymarket",
                    contract_id="p1",
                    question="Will CPI print above expectations?",
                    risk_category="inflation",
                    risk_tags="cpi",
                ),
                make_contract_row(
                    contract_id="k2",
                    question="Will unemployment rise?",
                    risk_category="labor_market",
                    risk_tags="unemployment",
                ),
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
                make_contract_row(contract_id="k1", geo_scope="us", time_horizon="short_term"),
                make_contract_row(contract_id="k2", geo_scope="unknown", time_horizon="unknown"),
                make_contract_row(contract_id="k3", geo_scope="international", time_horizon="medium_term"),
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
                make_contract_row(contract_id="k1"),
                make_contract_row(contract_id="k2"),
                make_contract_row(contract_id="k3"),
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
        rows = normalize_polymarket_markets(
            [
                {
                    "id": "pm-date-only-event",
                    "title": "Date only test",
                    "markets": [
                        {
                            "id": "pm-date-only",
                            "question": "Will CPI be above 3% in June 2099?",
                            "description": "Date only endDateIso test.",
                            "outcomes": ["Yes", "No"],
                            "outcomePrices": ["0.55", "0.45"],
                            "volume": "100",
                            "liquidity": "200",
                            "active": True,
                            "closed": False,
                            "archived": False,
                            "restricted": False,
                            "enableOrderBook": True,
                            "endDateIso": "2099-06-30",
                        }
                    ],
                }
            ],
            snapshot_ts="2099-06-30T23:59:59+00:00",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["expiration_ts"], "2099-06-30")
        self.assertIn(rows[0]["time_horizon"], {"short_term", "medium_term", "long_term"})

    def test_contracts_csv_keeps_polymarket_reference_rows_but_execution_filter_is_kalshi_only(self) -> None:
        data_dir = self.with_temp_data_dir()

        with ExitStack() as stack:
            for manager in self.patch_dataset_paths(data_dir):
                stack.enter_context(manager)
            stack.enter_context(patch.object(dataset_module, "fetch_open_events", return_value=KALSHI_RAW))
            stack.enter_context(patch.object(dataset_module, "fetch_active_events", return_value=POLYMARKET_RAW))
            stack.enter_context(
                patch.object(dataset_module, "utc_now_iso", return_value="2099-06-30T23:59:59+00:00")
            )
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
