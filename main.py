from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from services.risk_engine import RiskEngine
    engine = RiskEngine()
except Exception as exc:  # Helpful startup message for students who forgot setup.
    engine = None
    startup_error = str(exc)

app = FastAPI(title="UPI Shield — Adaptive Mule Account Layering Prevention", version="1.0.0")


class TransactionCheckRequest(BaseModel):
    sender_upi: str = Field(min_length=3)
    receiver_upi: str = Field(min_length=3)
    amount: float = Field(gt=0, le=500000)


class PostCheckRequest(BaseModel):
    receiver_upi: str = Field(min_length=3)
    observed_outflow_5m: float | None = Field(default=None, ge=0)
    downstream_accounts_5m: int | None = Field(default=None, ge=0, le=1000)


@app.get("/health")
def health():
    return {"status": "ok" if engine else "setup_required", "error": None if engine else startup_error}


@app.post("/api/v1/pre-check")
def pre_transaction_check(req: TransactionCheckRequest):
    if engine is None:
        raise HTTPException(503, detail=f"Model not initialized: {startup_error}")
    return engine.pre_check(req.receiver_upi, req.amount)


@app.post("/api/v1/post-check")
def post_transaction_check(req: PostCheckRequest):
    if engine is None:
        raise HTTPException(503, detail=f"Model not initialized: {startup_error}")
    return engine.post_check(req.receiver_upi, req.observed_outflow_5m, req.downstream_accounts_5m)


@app.get("/api/v1/reports/{receiver_upi}")
def reports(receiver_upi: str):
    if engine is None:
        raise HTTPException(503, detail=f"Model not initialized: {startup_error}")
    rows = engine.repo.reports_for(receiver_upi)
    return {"receiver_upi": receiver_upi, "count": len(rows), "reports": rows.to_dict(orient="records")}
