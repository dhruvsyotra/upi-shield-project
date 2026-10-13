from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"


class LocalRepository:
    def __init__(self) -> None:
        self.accounts = pd.read_csv(DATA / "account_profiles.csv")
        self.transactions = pd.read_csv(DATA / "upi_transactions.csv")
        self.reports = pd.read_csv(DATA / "scam_reports.csv")
        self.transactions["timestamp"] = pd.to_datetime(self.transactions["timestamp"], utc=True)

    def account(self, upi: str) -> dict:
        row = self.accounts.loc[self.accounts["upi"].eq(upi)]
        if row.empty:
            return {"upi": upi, "account_age_days": 30, "report_count": 0, "verified": False, "is_mule": 0}
        return row.iloc[0].to_dict()

    def reports_for(self, upi: str) -> pd.DataFrame:
        return self.reports.loc[self.reports["target_upi"].eq(upi)].copy()

    def receiver_recent_stats(self, upi: str, minutes: int = 5) -> dict:
        now = self.transactions["timestamp"].max()
        recent = self.transactions[self.transactions["timestamp"] >= now - pd.Timedelta(minutes=minutes)]
        inflow = recent.loc[recent["receiver_upi"].eq(upi)]
        outflow = recent.loc[recent["sender_upi"].eq(upi)]
        historical_in = self.transactions.loc[self.transactions["receiver_upi"].eq(upi), "amount"]
        historical_avg = float(historical_in.mean()) if len(historical_in) else 0.0
        return {
            "inflow_5m": float(inflow["amount"].sum()),
            "outflow_5m": float(outflow["amount"].sum()),
            "tx_count_5m": int(len(inflow)),
            "distinct_downstream_5m": int(outflow["receiver_upi"].nunique()),
            "avg_tx_amount": float(inflow["amount"].mean()) if len(inflow) else historical_avg,
        }
