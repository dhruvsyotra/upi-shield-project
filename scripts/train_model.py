from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from pipelines.features import FEATURE_COLUMNS, build_account_features
from models.autoencoder import OutflowAnomalyAutoencoder, reconstruction_errors

BASE = ROOT
DATA = BASE / "data"
ART = BASE / "models" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


def train(epochs: int = 30) -> None:
    accounts = pd.read_csv(DATA / "account_profiles.csv")
    tx = pd.read_csv(DATA / "upi_transactions.csv")
    reports = pd.read_csv(DATA / "scam_reports.csv")
    features = build_account_features(accounts, tx, reports)

    X = features[FEATURE_COLUMNS].astype(float).replace([np.inf, -np.inf], 0).fillna(0).to_numpy()
    y = features["is_mule"].to_numpy().astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    scaler = StandardScaler().fit(X_train[y_train == 0])
    train_normal = scaler.transform(X_train[y_train == 0]).astype(np.float32)
    test_scaled = scaler.transform(X_test).astype(np.float32)

    torch.manual_seed(42)
    model = OutflowAnomalyAutoencoder(input_dim=len(FEATURE_COLUMNS))
    loader = DataLoader(TensorDataset(torch.from_numpy(train_normal)), batch_size=64, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for (batch,) in loader:
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()

    normal_errors = reconstruction_errors(model, torch.from_numpy(train_normal)).numpy()
    threshold = float(np.quantile(normal_errors, 0.995))
    test_errors = reconstruction_errors(model, torch.from_numpy(test_scaled)).numpy()
    y_pred = (test_errors > threshold).astype(int)
    auc = float(roc_auc_score(y_test, test_errors))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    torch.save(model.state_dict(), ART / "outflow_autoencoder.pt")
    joblib.dump(scaler, ART / "scaler.joblib")
    (ART / "model_meta.json").write_text(json.dumps({
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
        "metrics": {"roc_auc": auc, "precision": precision, "recall": recall, "f1": f1},
    }, indent=2))
    print(json.dumps({"roc_auc": auc, "precision": precision, "recall": recall, "f1": f1, "threshold": threshold}, indent=2))


if __name__ == "__main__":
    train()
