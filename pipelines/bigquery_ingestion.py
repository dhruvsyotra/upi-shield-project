from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def load_transactions_to_bigquery(csv_path: str | Path, table_id: str | None = None) -> int:
    """Optional BigQuery adapter; local pipeline remains usable without GCP credentials."""
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise RuntimeError("Install google-cloud-bigquery for GCP ingestion.") from exc

    table_id = table_id or os.getenv("BIGQUERY_TABLE")
    if not table_id:
        raise ValueError("Set BIGQUERY_TABLE as project.dataset.table or pass table_id.")
    df = pd.read_csv(csv_path)
    client = bigquery.Client()
    job = client.load_table_from_dataframe(df, table_id)
    job.result()
    return len(df)
