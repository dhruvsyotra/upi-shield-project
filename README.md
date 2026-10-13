# UPI Shield — Adaptive Mule Account Layering Prevention

A student-friendly, end-to-end prototype for the proposed:

**Adaptive Mule Account Layering Prevention & Crowd-Sourced UPI Risk Scoring Engine**

This version improves the original design by replacing hard-coded demo decisions with:

- Synthetic transaction + crowd-report generation
- Feature engineering for payee velocity, transfer frequency, age, report density and downstream fan-out
- A PyTorch autoencoder trained on normal account behavior
- Hybrid recipient risk scoring (behavior + reputation + report evidence)
- A LangGraph guardrail state machine with SAFE / REVIEW / HOLD states
- FastAPI pre-check and post-payment endpoints
- Streamlit payment-warning and admin-monitoring dashboard
- Optional Pinecone and BigQuery adapters (the app works locally without cloud credentials)

> Safety note: this is a simulation. It does **not** connect to a real UPI rail or actually freeze bank funds. “Lien/HOLD” is represented as a decision/state in the prototype.

## Project structure

```text
upi_shield/
├── app.py
├── main.py
├── requirements.txt
├── .env.example
├── README.md
├── data/
├── models/
│   ├── autoencoder.py
│   └── artifacts/                 # generated model + thresholds
├── agents/
│   └── lien_guardrail.py
├── pipelines/
│   ├── features.py
│   └── vector_indexer.py
├── scripts/
│   ├── data_generator.py
│   └── train_model.py
├── services/
│   ├── risk_engine.py
│   ├── repositories.py
│   └── vector_store.py
└── artifacts/
```

## 1. Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Generate synthetic data

```bash
python scripts/data_generator.py
```

This creates:

- `data/upi_transactions.csv`
- `data/scam_reports.csv`
- `data/account_profiles.csv`

## 3. Train the anomaly detector

```bash
python scripts/train_model.py
```

The script trains on normal account-level behavior and stores:

- `models/artifacts/outflow_autoencoder.pt`
- `models/artifacts/model_meta.json`
- `models/artifacts/scaler.joblib`

## 4. Start the API

```bash
uvicorn main:app --reload --port 8000
```

Open Swagger at `http://localhost:8000/docs`.

## 5. Start the Streamlit UI

In another terminal:

```bash
streamlit run app.py
```

## Example workflow

1. Enter a recipient such as `mule12@upi` in the dashboard.
2. The API combines account profile, crowd report reputation, and behavioral indicators into a 0–100 score.
3. A warning is shown **before simulated PIN entry**.
4. Run a post-payment check for the same receiver.
5. The anomaly detector estimates downstream layering risk.
6. LangGraph routes the account to `SAFE`, `REVIEW`, or `HOLD`.

## API

### POST `/api/v1/pre-check`

Example:

```json
{
  "sender_upi": "victim@upi",
  "receiver_upi": "mule12@upi",
  "amount": 15000
}
```

### POST `/api/v1/post-check`

Example:

```json
{
  "receiver_upi": "mule12@upi",
  "observed_outflow_5m": 48500,
  "downstream_accounts_5m": 7
}
```

## Evaluation

Training prints:

- ROC-AUC
- Precision
- Recall
- F1
- anomaly threshold

Because the detector is unsupervised, the threshold is learned from the normal training reconstruction-error distribution. Evaluation is then performed against held-out synthetic labels.

## Optional cloud integrations

The local demo deliberately runs without requiring cloud credentials.

Set these when you want to connect adapters:

```text
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
GOOGLE_CLOUD_PROJECT=
BIGQUERY_DATASET=
BIGQUERY_TABLE=
```

The included `vector_store.py` and `repositories.py` have local fallbacks and are structured so Pinecone/BigQuery can be plugged in later.

## 6. Smoke test

```bash
python tests_smoke.py
```

## 7. Cloud adapters

`pipelines/bigquery_ingestion.py` shows how the generated transaction table can be loaded into BigQuery, while `pipelines/vector_indexer.py` creates Pinecone-ready transformer vectors. These are intentionally optional so the student demo does not require cloud billing/credentials.
