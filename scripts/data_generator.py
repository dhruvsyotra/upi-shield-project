from __future__ import annotations

import argparse
import uuid
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

CATEGORIES = ["Digital Arrest", "Fake Collect", "KYC Scam", "Investment Scam", "Impersonation"]


def make_accounts(n_accounts: int = 800) -> pd.DataFrame:
    rows = []
    mule_count = max(20, int(n_accounts * 0.07))
    mule_ids = set(RNG.choice(np.arange(n_accounts), size=mule_count, replace=False).tolist())
    # Reserve deterministic demo mule IDs so the dashboard examples always exist.
    demo_mules = {12, 25, 41, 77, 99}
    mule_ids.update(i for i in demo_mules if i < n_accounts)
    for i in range(n_accounts):
        is_mule = i in mule_ids
        age = int(RNG.integers(2, 25)) if is_mule else int(RNG.integers(45, 2500))
        report_count = int(RNG.poisson(4.5 if is_mule else 0.15))
        rows.append(
            {
                "upi": f"mule{i:03d}@upi" if is_mule else f"user{i:04d}@upi",
                "account_age_days": age,
                "report_count": report_count,
                "verified": bool(not is_mule and RNG.random() > 0.05),
                "is_mule": int(is_mule),
            }
        )
    return pd.DataFrame(rows)


def make_transactions(accounts: pd.DataFrame, n_transactions: int = 20000) -> pd.DataFrame:
    mule_accounts = accounts.loc[accounts.is_mule.eq(1), "upi"].to_numpy()
    normal_accounts = accounts.loc[accounts.is_mule.eq(0), "upi"].to_numpy()
    rows = []
    end = pd.Timestamp.now(tz="UTC").floor("s")
    for _ in range(n_transactions):
        is_mule_receiver = RNG.random() < 0.09
        receiver = RNG.choice(mule_accounts if is_mule_receiver else normal_accounts)
        sender = f"sender{RNG.integers(1000, 9999)}@upi"
        ts = end - pd.Timedelta(minutes=int(RNG.integers(0, 60 * 24 * 30)))
        amount = float(np.clip(RNG.lognormal(mean=8.2 if is_mule_receiver else 7.2, sigma=0.7), 100, 100000))
        rows.append(
            {
                "tx_id": str(uuid.uuid4()),
                "sender_upi": sender,
                "receiver_upi": receiver,
                "amount": round(amount, 2),
                "timestamp": ts.isoformat(),
                "is_mule": int(is_mule_receiver),
            }
        )
    return pd.DataFrame(rows)


def make_reports(accounts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    report_id = 0
    for _, a in accounts.iterrows():
        for _ in range(int(a.report_count)):
            rows.append(
                {
                    "report_id": f"R{report_id:06d}",
                    "target_upi": a.upi,
                    "report_text": (
                        RNG.choice(CATEGORIES)
                        + " scam reported; recipient requested urgent transfer and impersonated an authority."
                    ),
                    "category": RNG.choice(CATEGORIES),
                }
            )
            report_id += 1
    return pd.DataFrame(rows)


def generate(num_accounts: int, num_transactions: int) -> None:
    accounts = make_accounts(num_accounts)
    transactions = make_transactions(accounts, num_transactions)
    reports = make_reports(accounts)

    # Add an explicitly generated downstream pattern for a subset of mule accounts.
    patterns = []
    mule_upis = accounts.loc[accounts.is_mule.eq(1), "upi"].head(max(10, num_accounts // 80)).tolist()
    for upi in mule_upis:
        base = end = pd.Timestamp.now(tz="UTC").floor("s")
        inbound = float(RNG.uniform(15000, 50000))
        for j in range(8):
            patterns.append(
                {
                    "tx_id": str(uuid.uuid4()),
                    "sender_upi": f"victim{RNG.integers(1000,9999)}@upi",
                    "receiver_upi": upi,
                    "amount": round(inbound / 8, 2),
                    "timestamp": (base - pd.Timedelta(minutes=j + 1)).isoformat(),
                    "is_mule": 1,
                }
            )
    transactions = pd.concat([transactions, pd.DataFrame(patterns)], ignore_index=True)

    accounts.to_csv(DATA / "account_profiles.csv", index=False)
    transactions.to_csv(DATA / "upi_transactions.csv", index=False)
    reports.to_csv(DATA / "scam_reports.csv", index=False)
    print(f"Created {len(accounts):,} accounts, {len(transactions):,} transactions, {len(reports):,} scam reports")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--accounts", type=int, default=800)
    parser.add_argument("--transactions", type=int, default=20000)
    args = parser.parse_args()
    generate(args.accounts, args.transactions)
