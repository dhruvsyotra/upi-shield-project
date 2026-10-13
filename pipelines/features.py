from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "inflow_5m",
    "outflow_5m",
    "outflow_ratio_5m",
    "distinct_downstream_5m",
    "tx_count_5m",
    "avg_tx_amount",
    "account_age_days",
    "report_density",
]


def build_account_features(accounts: pd.DataFrame, tx: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"], utc=True)
    now = tx["timestamp"].max()

    inbound = tx.groupby("receiver_upi").agg(
        inflow_total=("amount", "sum"),
        tx_count=("tx_id", "count"),
        avg_tx_amount=("amount", "mean"),
    )
    downstream_proxy = tx.groupby("receiver_upi")["sender_upi"].nunique().rename("distinct_downstream_30d")

    recent = tx[tx["timestamp"] >= now - pd.Timedelta(minutes=5)]
    recent_in = recent.groupby("receiver_upi")["amount"].sum().rename("inflow_5m")
    recent_out = recent.groupby("sender_upi")["amount"].sum().rename("outflow_5m")
    recent_count = recent.groupby("receiver_upi")["tx_id"].count().rename("tx_count_5m")
    recent_distinct = recent.groupby("receiver_upi")["sender_upi"].nunique().rename("distinct_downstream_5m")

    result = accounts.merge(inbound, left_on="upi", right_index=True, how="left")
    result = result.merge(downstream_proxy, left_on="upi", right_index=True, how="left")
    result = result.merge(recent_in, left_on="upi", right_index=True, how="left")
    result = result.merge(recent_out, left_on="upi", right_index=True, how="left")
    result = result.merge(recent_count, left_on="upi", right_index=True, how="left")
    result = result.merge(recent_distinct, left_on="upi", right_index=True, how="left")

    report_counts = reports.groupby("target_upi").size().rename("report_count_actual") if not reports.empty else pd.Series(dtype=float)
    result = result.merge(report_counts, left_on="upi", right_index=True, how="left")
    result["report_count"] = result["report_count_actual"].fillna(result["report_count"]).fillna(0)

    for c in ["inflow_total", "tx_count", "avg_tx_amount", "distinct_downstream_30d", "inflow_5m", "outflow_5m", "tx_count_5m", "distinct_downstream_5m"]:
        result[c] = result[c].fillna(0.0)
    result["outflow_ratio_5m"] = np.divide(
        result["outflow_5m"], result["inflow_5m"].clip(lower=1.0)
    ).clip(0, 10)
    result["report_density"] = result["report_count"] / (result["account_age_days"] + 30.0)
    return result
