from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from agents.lien_guardrail import GUARDRAIL_APP
from models.autoencoder import OutflowAnomalyAutoencoder, reconstruction_errors
from pipelines.features import FEATURE_COLUMNS
from services.repositories import LocalRepository
from services.vector_store import LocalSemanticReportStore

BASE = Path(__file__).resolve().parents[1]
ART = BASE / "models" / "artifacts"


class RiskEngine:
    def __init__(self) -> None:
        self.repo = LocalRepository()
        self.report_store = LocalSemanticReportStore(self.repo.reports)
        self.model = OutflowAnomalyAutoencoder(len(FEATURE_COLUMNS))
        self.model.load_state_dict(torch.load(ART / "outflow_autoencoder.pt", map_location="cpu", weights_only=True))
        self.model.eval()
        self.scaler = joblib.load(ART / "scaler.joblib")
        self.meta = json.loads((ART / "model_meta.json").read_text())
        self.threshold = float(self.meta["threshold"])

    @staticmethod
    def _clamp(v: float) -> float:
        return float(max(0.0, min(100.0, v)))

    def _anomaly(self, features: dict) -> tuple[float, float]:
        x = np.array([[features[c] for c in FEATURE_COLUMNS]], dtype=float)
        scaled = self.scaler.transform(x).astype(np.float32)
        error = float(reconstruction_errors(self.model, torch.from_numpy(scaled))[0])
        # Convert reconstruction error into 0-1 using a soft threshold curve.
        ratio = error / max(self.threshold, 1e-9)
        probability = 1.0 - float(np.exp(-max(0.0, ratio - 0.5)))
        return error, min(1.0, probability)

    def pre_check(self, receiver_upi: str, amount: float) -> dict:
        account = self.repo.account(receiver_upi)
        reports = self.repo.reports_for(receiver_upi)
        stats = self.repo.receiver_recent_stats(receiver_upi)
        age_score = 1.0 if account["account_age_days"] < 30 else max(0.0, 1 - account["account_age_days"] / 365)
        report_score = min(1.0, len(reports) / 6)
        velocity_score = min(1.0, stats["tx_count_5m"] / 8 + stats["inflow_5m"] / 75000)
        amount_score = min(1.0, amount / 50000)
        verification_penalty = 0.65 if not account["verified"] else 0.0
        risk = 100 * (0.45 * report_score + 0.20 * age_score + 0.15 * velocity_score + 0.10 * amount_score + 0.10 * verification_penalty)
        risk = self._clamp(risk)
        level = "RED_ALERT" if risk >= 75 else "AMBER" if risk >= 45 else "SAFE"
        categories = reports["category"].value_counts().head(3).to_dict() if not reports.empty else {}
        return {
            "receiver_upi": receiver_upi, "risk_score": round(risk, 1), "warning_level": level,
            "total_reports": int(len(reports)), "verified": bool(account["verified"]),
            "account_age_days": int(account["account_age_days"]), "recent_tx_count_5m": stats["tx_count_5m"],
            "report_categories": categories,
            "message": "Multiple risk signals detected. Verify the recipient before entering your UPI PIN." if level == "RED_ALERT" else "Some risk signals are present. Review the recipient details carefully." if level == "AMBER" else "No strong crowd or behavioral warning signals detected."
        }

    def post_check(self, receiver_upi: str, observed_outflow_5m: float | None = None, downstream_accounts_5m: int | None = None) -> dict:
        account = self.repo.account(receiver_upi)
        reports = self.repo.reports_for(receiver_upi)
        stats = self.repo.receiver_recent_stats(receiver_upi)
        if observed_outflow_5m is not None:
            stats["outflow_5m"] = float(observed_outflow_5m)
        if downstream_accounts_5m is not None:
            stats["distinct_downstream_5m"] = int(downstream_accounts_5m)
        contextual_inflow = stats["inflow_5m"]
        if contextual_inflow <= 0:
            # For a demonstration-only manual post-check, use the account's typical inbound amount
            # rather than treating absence of a 5-minute inbound event as proof of layering.
            contextual_inflow = max(stats["avg_tx_amount"], 100.0)
        features = {
            "inflow_5m": contextual_inflow,
            "outflow_5m": stats["outflow_5m"],
            "outflow_ratio_5m": stats["outflow_5m"] / max(contextual_inflow, 1.0),
            "distinct_downstream_5m": stats["distinct_downstream_5m"],
            "tx_count_5m": stats["tx_count_5m"],
            "avg_tx_amount": stats["avg_tx_amount"],
            "account_age_days": account["account_age_days"],
            "report_density": len(reports) / (account["account_age_days"] + 30.0),
        }
        error, anomaly_probability = self._anomaly(features)
        # Fuse behavioral anomaly with reputation evidence for the agent.
        reputation = min(1.0, len(reports) / 6)
        risk = self._clamp(100 * (0.70 * anomaly_probability + 0.20 * reputation + 0.10 * (1 - float(account["verified"]))))
        result = GUARDRAIL_APP.invoke({
            "receiver_upi": receiver_upi,
            "risk_score": risk,
            "anomaly_score": anomaly_probability,
            "report_count": len(reports),
            "verified": bool(account["verified"]),
            "action_log": [],
        })
        return {
            "receiver_upi": receiver_upi, "risk_score": round(risk, 1),
            "anomaly_probability": round(anomaly_probability, 4), "reconstruction_error": round(error, 6),
            "observed_outflow_5m": round(stats["outflow_5m"], 2),
            "downstream_accounts_5m": stats["distinct_downstream_5m"],
            "decision": result["decision"], "lien_status": result["lien_status"],
            "action_log": result["action_log"],
        }

    def report_similarity(self, receiver_upi: str, query: str) -> list[dict]:
        return [m.__dict__ for m in self.report_store.search(query, target_upi=receiver_upi)]
