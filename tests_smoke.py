from fastapi.testclient import TestClient
from main import app

c = TestClient(app)
assert c.get('/health').status_code == 200
pre = c.post('/api/v1/pre-check', json={"sender_upi": "victim@upi", "receiver_upi": "mule012@upi", "amount": 15000})
assert pre.status_code == 200
assert pre.json()["warning_level"] == "RED_ALERT"
post = c.post('/api/v1/post-check', json={"receiver_upi": "mule012@upi", "observed_outflow_5m": 48500, "downstream_accounts_5m": 7})
assert post.status_code == 200
assert post.json()["decision"] == "HOLD"
print('Smoke tests passed.')
